"""
01_etl.py — Ingesta y consolidacion de fuentes para el piloto Prediccion 1P vs Brent

Lee datos/raw/ y produce datos/staging/tablon_unico.parquet (+ .csv espejo).
Granularidad: CAMPO x AÑO x ESCENARIO.

Re-arquitectura 2026-06-04 (ver docs/MAESTRO.md §6):
  - VIGENCIA: año (BASE) o quarter del deck (CONSOLIDADO_YYYY_Qq).
  - ES_BASELINE: True para filas de serie historica oficial (ESCENARIO=BASE).
  - CHECKPOINT_1P_MBPE: cierre OFICIAL del MISMO año A (referencia/auditoría del confound).
  - BASELINE_1P_MBPE: cierre OFICIAL del año ANTERIOR (A−1). La Sensibilidad del deck se
    construye sobre el último libro certificado DISPONIBLE al armar el deck = cierre A−1
    (hallazgo 2026-07-09, validado vs Hoja1 del Consolidado; MAESTRO §10).
  - DELTA_SENS_MBPE: ΔReservas = SENSIBILIDAD − BASELINE_1P_MBPE (cierre A−1). Target del
    modelo. Restar el cierre A (semántica pre-2026-07-09) inyectaba el salto de
    recertificación A−1→A en el target = el confound de vigencia.
  - NIVEL_DEFINICIONAL: flag para quiebres definicionales (ej. 2026_REGALIAS).
  - VIGENCIA_BREAKEVEN: cierre FC del que viene el breakeven ("2024", "2025", ...).
    Fuente primaria: breakeven_resultados.parquet (generado por 05_breakeven.py).
    ALERTA=BREAKEVEN_VIGENCIA_STALE si el BK usado es anterior al AÑO de la fila.

Bugs corregidos:
  B2: Consolidado descuentos normalizados a signo negativo con -abs() (fuente puede
      tener cualquier signo; -abs() garantiza convencion unica en el tablon).
  B4: Identidad Brent=Neto−Cal−Tra validada por fila en HIST → ALERTA=IDENTIDAD_PRECIO.
  B5: Auditoria de homologacion en Consolidado sobre el set COMPLETO antes de filtrar piloto.
  B6: Filas con neto OK pero reservas NaN se incluyen con ALERTA=TARGET_NULO (no se descartan).
  B7: CASTILLA ESTE sin vol para ponderar → ALERTA=BREAKEVEN_FALLBACK (no usar en CAPEX).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from homologacion import Homologador, reporte_homologacion, normalizar_base
from track import sufijo_track

warnings.filterwarnings("ignore", category=UserWarning)

# ── Rutas ──────────────────────────────────────────────────────────────────────
_SUF = sufijo_track()   # '' Produccion; '_calidad' si PRED_TRACK=calidad
BASE_DIR = Path(__file__).parent
RAW      = BASE_DIR / "datos" / "raw"
STAGING  = BASE_DIR / "datos" / f"staging{_SUF}"
RESULTADOS = BASE_DIR / f"resultados{_SUF}"
STAGING.mkdir(parents=True, exist_ok=True)
RESULTADOS.mkdir(parents=True, exist_ok=True)

# Coherencia BK↔deck (directriz DIRECTRIZ_ESCALERA_DECK.md §2): tolerancias del flag
# de revisión de Finanzas. NO altera curvas; solo marca campos donde el deck real
# contradice el umbral del breakeven.
BK_COHERENCIA_MARGEN_USD   = 2.0    # margen sobre BK_ANCLA_FIN para "libro vivo bajo el BK"
BK_COHERENCIA_DELTA_FRAC   = 0.05   # delta > −5%·baseline = el libro sigue vivo
BK_COHERENCIA_BANDA_USD    = 5.0    # "precipicio pegado a la banda": banda a <$5 del BK
BK_COHERENCIA_BASELINE_MIN = 5.0    # materialidad: micro-campos no van a revisión Finanzas

HIST_1P     = RAW / "HIST 1P.xlsx"
CONSOLIDADO = RAW / "Consolidado Bases de Datos.xlsx"   # renombrado por el usuario 2026-07-09

# Fase 1: universo completo (todos los campos Incluir=Si del Consolidado).
# Para volver a modo piloto, descomentar la línea siguiente:
# CAMPOS_PILOTO = {"CASTILLA", "CASTILLA NORTE", "CASTILLA ESTE", "RUBIALES"}
CAMPOS_PILOTO: set[str] = set()   # vacío = sin filtro de piloto

# Tolerancia para filtro excepciones: |Brent_campo − Brent_oficial| > TOL → excluir
TOL_BRENT = 2.0
# Tolerancia identidad Brent = Neto − Cal − Tra
TOL_IDENTIDAD = 0.1
# Cierre FC del Breakeven actual (archivos Jan-2025 = Cierre 2024)
VIGENCIA_BREAKEVEN_ACTUAL = "2024"

H = Homologador()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _col_anio(df: pd.DataFrame) -> str:
    """Detecta la columna año robustamente (AÑO / Año / A\xd1o / AÑO)."""
    for c in df.columns:
        cc = str(c).strip().upper()
        if cc.startswith("A") and "O" in cc and len(cc) <= 4:
            return c
    raise KeyError(f"No se encontro columna de año en {list(df.columns)}")


def _homologar_columna(df: pd.DataFrame, col_raw: str, origen: str = "") -> pd.DataFrame:
    """Homologa una columna completa y audita NO_HOMOLOGADOS sobre el set completo."""
    df = df.copy()
    df["CAMPO_ORIGEN_RAW"] = df[col_raw].astype(str)
    hom = H.homologar_serie(df[col_raw])
    df["CAMPO"]        = hom["CAMPO"]
    df["HOMOLOG_FLAG"] = hom["HOMOLOG_FLAG"]
    if origen:
        reporte_homologacion(df, origen)
    return df


def _prom_ponderado(valores: pd.Series, pesos: pd.Series) -> float:
    """Promedio ponderado por 'pesos' (volumen) ignorando NaN en 'valores'.
    Sin peso válido (todo NaN/0) → promedio simple; todo valor NaN → NaN.

    Agregación v3 (2026-07-09): al fusionar componentes físicos de un UNIFICADO
    (APIAY+GAVAN+GUATIQUIA, LISAMA+NUTRIA+TESORO, CHICHIMENE+CHICHIMENE SW) los
    volúmenes se SUMAN pero los precios/descuentos deben ponderarse por el volumen
    de cada componente — un promedio simple sobredimensiona los campos pequeños.
    """
    m = valores.notna()
    if not m.any():
        return np.nan
    w = pesos.where(m).fillna(0.0)
    sw = float(w.sum())
    if sw > 0:
        return float((valores.fillna(0.0) * w).sum() / sw)
    return float(valores[m].mean())


# ── Lectores de fuente ──────────────────────────────────────────────────────────

def leer_brent_oficial() -> pd.DataFrame:
    """Lee HIST 1P / Precio Brent: Brent oficial de cierre por año."""
    df = pd.read_excel(HIST_1P, sheet_name="Precio Brent", engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    col_a = _col_anio(df)
    col_b = [c for c in df.columns if "BRENT" in str(c).upper()][0]
    out = df[[col_a, col_b]].rename(columns={col_a: "AÑO", col_b: "BRENT_OFICIAL_USD_BBL"})
    out["AÑO"]                 = pd.to_numeric(out["AÑO"], errors="coerce").astype("Int64")
    out["BRENT_OFICIAL_USD_BBL"] = pd.to_numeric(out["BRENT_OFICIAL_USD_BBL"], errors="coerce")
    return out.dropna(subset=["AÑO"])


def leer_hist1p_reservas() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lee HIST 1P / Reservas (MBPE). Trae columna CAMPO granular directa.
    Decimales con coma ("186,04") → float.

    Retorna (df_agregado, pesos):
      df_agregado — CAMPO(UNIFICADO)×AÑO con volúmenes SUMADOS de sus componentes.
      pesos       — CAMPO_ORIGEN_RAW×AÑO con RAW_NORM y 1P por componente, para
                    ponderar los precios de HIST (misma clave que la hoja Precio Net).
    """
    df = pd.read_excel(HIST_1P, sheet_name="Reservas (MBPE)", engine="openpyxl", dtype=str)
    df.columns = [str(c).strip().upper() for c in df.columns]
    col_a = _col_anio(df)

    cols_num = ["PDP", "PNP", "PND", "1P", "PROD", "PRB", "PS", "3P",
                "DIF RP", "DIF RP (%)", "MAX OF LIMIT_1P", "VAR 1P", "R/P", "IRR"]
    for col in cols_num:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                       .str.replace(",", ".", regex=False).str.strip()
                       .replace({"nan": np.nan, "": np.nan, "NONE": np.nan})
                       .str.replace("%", "", regex=False)
                       .apply(lambda x: pd.to_numeric(x, errors="coerce")))

    df["AÑO"] = pd.to_numeric(df[col_a], errors="coerce").astype("Int64")
    df = _homologar_columna(df, "CAMPO", "Reservas/HIST")
    if CAMPOS_PILOTO:
        df = df[df["CAMPO"].isin(CAMPOS_PILOTO)]
    df = df.dropna(subset=["AÑO", "CAMPO"]).copy()

    cols = ["CAMPO", "CAMPO_ORIGEN_RAW", "AÑO", "PDP", "PNP", "PND", "1P", "PROD"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].rename(columns={
        "PDP": "VOLUMEN_PDP_MBPE", "PNP": "VOLUMEN_PNP_MBPE",
        "PND": "VOLUMEN_PND_MBPE", "1P":  "VOLUMEN_1P_OFICIAL_MBPE",
        "PROD": "PRODUCCION_MBPE",
    })
    vol_cols = [c for c in ["VOLUMEN_PDP_MBPE", "VOLUMEN_PNP_MBPE", "VOLUMEN_PND_MBPE",
                            "VOLUMEN_1P_OFICIAL_MBPE", "PRODUCCION_MBPE"] if c in df.columns]

    # Agregacion v3 (2026-07-09): un UNIFICADO agrupa CAMPOS FISICOS distintos que se
    # certifican juntos (APIAY=APIAY+APIAY ESTE+GAVAN+GUATIQUIA; LISAMA=LISAMA+NUTRIA+
    # TESORO; CHICHIMENE+CHICHIMENE SW tras la fusion en DIM). Sus reservas SON barriles
    # aditivos -> SE SUMAN. El coalesce .first() anterior conservaba solo el componente
    # mayor y botaba el resto: la serie quedaba truncada y saltaba de forma artificial
    # al año en que la fuente ya reporta la fila unificada -> confound de vigencia FALSO
    # (APIAY 2020=4.98 coalesce vs 9.69 suma -> salto ficticio a 10.27 en 2021).
    # Paso A: colapsar duplicados EXACTOS del MISMO componente (RAW×AÑO) con el mayor
    #         valor no-nulo — cubre el renombre historico (una fila trae el dato del año
    #         del cambio de nombre, la otra viene vacia). NUNCA suma un componente consigo.
    df = (df.sort_values("VOLUMEN_1P_OFICIAL_MBPE", ascending=False, na_position="last")
            .groupby(["CAMPO", "CAMPO_ORIGEN_RAW", "AÑO"], as_index=False).first())

    # Pesos por componente para ponderar los precios de HIST (clave = nombre crudo tal
    # como aparece tambien en la hoja Precio Net, normalizado).
    pesos = df[["CAMPO_ORIGEN_RAW", "AÑO", "VOLUMEN_1P_OFICIAL_MBPE"]].copy()
    pesos["RAW_NORM"] = pesos["CAMPO_ORIGEN_RAW"].apply(normalizar_base)
    pesos = pesos.rename(columns={"VOLUMEN_1P_OFICIAL_MBPE": "PESO_1P"})

    # Paso B: SUMAR los componentes fisicos distintos del mismo UNIFICADO×AÑO.
    # min_count=1: todo-NaN -> NaN (ausencia real), no 0.
    n_comp = df.groupby(["CAMPO", "AÑO"])["CAMPO_ORIGEN_RAW"].nunique()
    multi = n_comp[n_comp > 1]
    df = df.groupby(["CAMPO", "AÑO"], as_index=False)[vol_cols].sum(min_count=1)
    if len(multi) > 0:
        campos_sum = sorted(multi.index.get_level_values("CAMPO").unique())
        print(f"      [INFO] {len(multi)} CAMPO×AÑO con >1 componente fisico SUMADOS "
              f"(agregacion v3): {campos_sum[:8]}{'...' if len(campos_sum) > 8 else ''}")
    return df, pesos


def leer_hist1p_precio(df_brent_of: pd.DataFrame, pesos: pd.DataFrame = None) -> pd.DataFrame:
    """
    Lee HIST 1P / 'Precio Net ' (espacio final).

    CORRECCIÓN CRITICA: 'Precio Aceite' = PRECIO NETO, no el Brent.
      BRENT_FLAT   ← Brent oficial por año (hoja Precio Brent).
      PRECIO_NETO  ← 'Precio Aceite (USD/bbl)'.
      DESCUENTO_*  ← descuentos (signo nativo negativo en HIST).

    Filtro de excepciones: |Brent_campo − Brent_oficial| > TOL_BRENT → excluir.
    B4: Identidad Brent = Neto − Cal − Tra validada por fila → ALERTA_PRECIO por fila.
    """
    df = pd.read_excel(HIST_1P, sheet_name="Precio Net ", engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    col_a = _col_anio(df)

    ren = {
        "Precio Aceite (USD/bbl)":       "PRECIO_NETO_USD_BBL",
        "Descuento Calidad (USD/bbl)":   "DESCUENTO_CALIDAD_USD_BBL",
        "Descuento Transporte (USD/bbl)":"DESCUENTO_TRANSPORTE_USD_BBL",
        "Precio Brent (USD/bbl)":        "BRENT_CAMPO_USD_BBL",
        col_a: "AÑO",
    }
    df = df.rename(columns=ren)
    for c in ["PRECIO_NETO_USD_BBL", "DESCUENTO_CALIDAD_USD_BBL",
              "DESCUENTO_TRANSPORTE_USD_BBL", "BRENT_CAMPO_USD_BBL"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce").astype("Int64")

    df = _homologar_columna(df, "CAMPO", "Precio/HIST")
    if CAMPOS_PILOTO:
        df = df[df["CAMPO"].isin(CAMPOS_PILOTO)]
    df = df.dropna(subset=["AÑO", "CAMPO"]).copy()

    df = df.merge(df_brent_of, on="AÑO", how="left")

    # Filtro excepciones: filiales/marcadores extranjeros con Brent distinto
    dev = (df["BRENT_CAMPO_USD_BBL"] - df["BRENT_OFICIAL_USD_BBL"]).abs()
    n_exc = int((dev > TOL_BRENT).sum())
    if n_exc > 0:
        print(f"      [INFO] {n_exc} filas excluidas (|Brent campo - oficial| > {TOL_BRENT})")
    df = df[~(dev > TOL_BRENT)].copy()

    # BRENT_FLAT = oficial por año (fuente correcta, no el campo)
    df["BRENT_FLAT_USD_BBL"] = df["BRENT_OFICIAL_USD_BBL"]

    # B2: Forzar signo negativo en descuentos (convención única: Brent = Neto − Cal − Tra)
    # HIST debería ser nativo-negativo, pero hay filas con signo invertido en la fuente
    for _dc in ["DESCUENTO_CALIDAD_USD_BBL", "DESCUENTO_TRANSPORTE_USD_BBL"]:
        df[_dc] = -df[_dc].abs()

    # B4: Identidad Brent = Neto − Cal − Tra (descuentos negativos en HIST)
    ident = (df["PRECIO_NETO_USD_BBL"]
             - df["DESCUENTO_CALIDAD_USD_BBL"]
             - df["DESCUENTO_TRANSPORTE_USD_BBL"]
             - df["BRENT_CAMPO_USD_BBL"]).abs()
    n_fail = int((ident > TOL_IDENTIDAD).sum())
    if n_fail > 0:
        print(f"      [INFO] {n_fail} filas con identidad violada -> ALERTA=IDENTIDAD_PRECIO")
    df["ALERTA_PRECIO"] = np.where(ident > TOL_IDENTIDAD, "IDENTIDAD_PRECIO", "")

    df = df[["CAMPO", "CAMPO_ORIGEN_RAW", "AÑO", "BRENT_FLAT_USD_BBL",
             "DESCUENTO_CALIDAD_USD_BBL", "DESCUENTO_TRANSPORTE_USD_BBL",
             "PRECIO_NETO_USD_BBL", "ALERTA_PRECIO"]].copy()

    # Agregacion v3 (2026-07-09): al fusionar componentes fisicos de un UNIFICADO, el
    # precio neto y los descuentos se ponderan por el volumen 1P de cada componente
    # (pesos de leer_hist1p_reservas). Sin pesos (o sin match) -> promedio simple.
    # BRENT_FLAT es oficial por año (igual para todos los componentes): se conserva.
    df["RAW_NORM"] = df["CAMPO_ORIGEN_RAW"].apply(normalizar_base)
    if pesos is not None and not pesos.empty:
        df = df.merge(pesos[["RAW_NORM", "AÑO", "PESO_1P"]],
                      on=["RAW_NORM", "AÑO"], how="left")
    else:
        df["PESO_1P"] = np.nan

    def _agg_precio(g: pd.DataFrame) -> pd.Series:
        w = g["PESO_1P"]
        out = {
            "BRENT_FLAT_USD_BBL": (g["BRENT_FLAT_USD_BBL"].dropna().iloc[0]
                                   if g["BRENT_FLAT_USD_BBL"].notna().any() else np.nan),
            "PRECIO_NETO_USD_BBL":         _prom_ponderado(g["PRECIO_NETO_USD_BBL"], w),
            "DESCUENTO_CALIDAD_USD_BBL":   _prom_ponderado(g["DESCUENTO_CALIDAD_USD_BBL"], w),
            "DESCUENTO_TRANSPORTE_USD_BBL":_prom_ponderado(g["DESCUENTO_TRANSPORTE_USD_BBL"], w),
            "ALERTA_PRECIO": "|".join(dict.fromkeys(
                p for p in g["ALERTA_PRECIO"].fillna("").astype(str) if p and p != "nan")),
        }
        return pd.Series(out)

    df = (df.groupby(["CAMPO", "AÑO"])
            .apply(_agg_precio, include_groups=False)
            .reset_index())
    return df


def leer_consolidado() -> pd.DataFrame:
    """
    Lee Consolidado / hoja 'Consolidado' — fuente PRIMARIA multi-estado.

    Layout: fila 5 = quarters, datos desde fila 7. Bloques de 9 quarters:
      cols 3-11 Precio Neto | 12-20 Desc Cal | 21-29 Desc Tra | 30-38 Reservas 1P.

    B2: Descuentos → -abs() para garantizar signo negativo independiente de la fuente.
    B5: Auditoria de homologacion sobre el set COMPLETO antes de filtrar a piloto.
    B6: Filas con neto OK y res NaN → incluir con ALERTA=TARGET_NULO.
    """
    raw = pd.read_excel(CONSOLIDADO, sheet_name="Consolidado", header=None, engine="openpyxl")
    # Layout 2026-07 (detección dinámica): se insertó columna ID y los bloques
    # dejaron de tener igual ancho — Precio Neto/Desc Calidad/Desc Transporte traen
    # 10 quarters (…2026_Q2) mientras Reservas 1P trae 9 (…2026_Q1, sin deck Q2 aún).
    # Se localizan columnas por etiqueta en vez de offsets fijos para no romper con
    # cada quarter nuevo:
    #   fila 4  → títulos "Precio Neto|Descuento Calidad|Descuento Transporte YYYY_QN"
    #   fila 5  → quarter del bloque Reservas 1P
    #   fila 6  → cabeceras cortas (Incluir, ID, Nombre, ..., "Reservas 1P")
    #   datos desde fila 7.
    hdr = raw.iloc[6].astype(str).str.strip()
    incluir_col = int(hdr[hdr.str.upper() == "INCLUIR"].index[0])
    nombre_col  = int(hdr[hdr.str.upper() == "NOMBRE"].index[0])

    # Mapas {quarter: col} por métrica desde los títulos de fila 4
    import re as _re
    fila4 = raw.iloc[4]
    map_neto, map_cal, map_tra = {}, {}, {}
    pat = _re.compile(r"^(Precio Neto|Descuento Calidad|Descuento Transporte)\s+(\d{4}_Q\d)$")
    for c in range(raw.shape[1]):
        m = pat.match(str(fila4.iloc[c]).strip())
        if not m:
            continue
        metrica, q = m.group(1), m.group(2)
        {"Precio Neto": map_neto, "Descuento Calidad": map_cal,
         "Descuento Transporte": map_tra}[metrica][q] = c

    # Reservas 1P: cabecera corta en fila 6, quarter en fila 5 (bloque más angosto)
    fila5 = raw.iloc[5]
    map_res = {}
    for c in range(raw.shape[1]):
        if str(hdr.iloc[c]).strip().upper() == "RESERVAS 1P":
            q = str(fila5.iloc[c]).strip()
            if _re.match(r"^\d{4}_Q\d$", q):
                map_res[q] = c

    # Recorrer los quarters con Precio Neto (los completos); Reservas puede faltar
    quarters = sorted(map_neto.keys())
    datos = raw.iloc[7:].reset_index(drop=True)

    # Brent de referencia por quarter (fila BRENT del deck — columna Nombre)
    fila_brent = datos[datos[nombre_col].astype(str).str.strip().str.upper() == "BRENT"]
    brent_q = {}
    if not fila_brent.empty:
        r = fila_brent.iloc[0]
        for q in quarters:
            brent_q[q] = pd.to_numeric(r.iloc[map_neto[q]], errors="coerce")

    # B5: Paso 1 — homologar TODOS los campos Incluir=Si para auditoria completa
    todos_hom = []
    for _, r in datos.iterrows():
        nombre = str(r.iloc[nombre_col]).strip()
        if not nombre or nombre.upper() in ("BRENT", "NAN", "NOMBRE"):
            continue
        incluir = str(r.iloc[incluir_col]).strip().upper()
        if incluir == "NO":
            continue
        campo, flag = H.homologar(nombre)
        todos_hom.append({"CAMPO": campo, "CAMPO_ORIGEN_RAW": nombre, "HOMOLOG_FLAG": flag})

    if todos_hom:
        reporte_homologacion(pd.DataFrame(todos_hom), "Consolidado")

    # Paso 2: procesar campos piloto con Incluir=Si
    filas = []
    for _, r in datos.iterrows():
        nombre = str(r.iloc[nombre_col]).strip()
        if not nombre or nombre.upper() in ("BRENT", "NAN", "NOMBRE"):
            continue
        incluir = str(r.iloc[incluir_col]).strip().upper()
        if incluir == "NO":
            continue
        campo, flag = H.homologar(nombre)
        if CAMPOS_PILOTO and campo not in CAMPOS_PILOTO:
            continue

        for q in quarters:
            neto = pd.to_numeric(r.iloc[map_neto[q]], errors="coerce")
            if pd.isna(neto) or neto == 0:
                continue

            cal_raw = pd.to_numeric(r.iloc[map_cal[q]], errors="coerce") if q in map_cal else np.nan
            tra_raw = pd.to_numeric(r.iloc[map_tra[q]], errors="coerce") if q in map_tra else np.nan
            res     = pd.to_numeric(r.iloc[map_res[q]], errors="coerce") if q in map_res else np.nan
            brent   = brent_q.get(q, np.nan)
            anio    = int(q.split("_")[0])

            # B2: normalizar siempre a signo negativo (-abs garantiza convencion unica)
            cal = -abs(float(cal_raw)) if pd.notna(cal_raw) else np.nan
            tra = -abs(float(tra_raw)) if pd.notna(tra_raw) else np.nan

            # B6: flag filas con precio pero sin volumen
            alerta = "TARGET_NULO" if pd.isna(res) else ""

            filas.append({
                "CAMPO":                      campo,
                "CAMPO_ORIGEN_RAW":           nombre,
                "HOMOLOG_FLAG":               flag,
                "AÑO":                        anio,
                "VIGENCIA":                   q,
                "ESCENARIO":                  f"CONSOLIDADO_{q}",
                "BRENT_FLAT_USD_BBL":         brent,
                "DESCUENTO_CALIDAD_USD_BBL":  cal,
                "DESCUENTO_TRANSPORTE_USD_BBL": tra,
                "PRECIO_NETO_USD_BBL":        neto,
                "VOLUMEN_1P_SENSIBILIDAD_MBPE": res if pd.notna(res) else np.nan,
                "ALERTA":                     alerta,
            })

    out = pd.DataFrame(filas)
    if out.empty:
        return out

    # Agregacion v3 (2026-07-09): CAMPO×ESCENARIO. Los componentes fisicos de un
    # UNIFICADO (CHICHIMENE+CHICHIMENE SW tras la fusion en DIM) SUMAN su sensibilidad
    # de reservas; el precio neto y descuentos se ponderan por esa sensibilidad. Para
    # campos con un solo componente el resultado es identico al coalesce previo.
    def _agg_cons(g: pd.DataFrame) -> pd.Series:
        res = g["VOLUMEN_1P_SENSIBILIDAD_MBPE"]
        w = res.where(res > 0)
        out_g = {
            "CAMPO_ORIGEN_RAW": g["CAMPO_ORIGEN_RAW"].iloc[0],
            "HOMOLOG_FLAG":     g["HOMOLOG_FLAG"].iloc[0],
            "AÑO":              g["AÑO"].iloc[0],
            "VIGENCIA":         g["VIGENCIA"].iloc[0],
            "BRENT_FLAT_USD_BBL": (g["BRENT_FLAT_USD_BBL"].dropna().iloc[0]
                                   if g["BRENT_FLAT_USD_BBL"].notna().any() else np.nan),
            "DESCUENTO_CALIDAD_USD_BBL":    _prom_ponderado(g["DESCUENTO_CALIDAD_USD_BBL"], w),
            "DESCUENTO_TRANSPORTE_USD_BBL": _prom_ponderado(g["DESCUENTO_TRANSPORTE_USD_BBL"], w),
            "PRECIO_NETO_USD_BBL":          _prom_ponderado(g["PRECIO_NETO_USD_BBL"], w),
            "VOLUMEN_1P_SENSIBILIDAD_MBPE": res.sum(min_count=1),
            "ALERTA": "|".join(dict.fromkeys(
                p for p in g["ALERTA"].fillna("").astype(str) if p and p != "nan")),
        }
        return pd.Series(out_g)

    out = (out.groupby(["CAMPO", "ESCENARIO"])
              .apply(_agg_cons, include_groups=False)
              .reset_index())

    # Alerta calidad de datos (2026-06-11): el diferencial implicito (neto − brent
    # del deck) debe ser estable entre quarters consecutivos del mismo campo. Un
    # salto > TOL_DIF_SALTO marca DIF_NETO_SALTO — caso RUBIALES 2025_Q4: dif pasa
    # de −8 a +0.03 (neto ≈ brent) mientras sus descuentos dicen −12.8; pendiente
    # validar el neto con planeacion. Se marca, no se corrige ni descarta.
    TOL_DIF_SALTO = 5.0
    out["_dif"] = out["PRECIO_NETO_USD_BBL"] - out["BRENT_FLAT_USD_BBL"]
    n_saltos = 0
    for campo, g in out.groupby("CAMPO"):
        g = g.sort_values("VIGENCIA")
        salto = g["_dif"].diff().abs() > TOL_DIF_SALTO
        idx_salto = g.index[salto.fillna(False)]
        if len(idx_salto) > 0:
            n_saltos += len(idx_salto)
            out.loc[idx_salto, "ALERTA"] = out.loc[idx_salto, "ALERTA"].apply(
                lambda a: "|".join(p for p in [str(a or ""), "DIF_NETO_SALTO"] if p))
    out.drop(columns=["_dif"], inplace=True)
    if n_saltos > 0:
        campos_salto = sorted(out.loc[out["ALERTA"].str.contains("DIF_NETO_SALTO", na=False),
                                      "CAMPO"].unique())
        print(f"      [WARN] {n_saltos} quarters con salto de diferencial neto-brent "
              f"> {TOL_DIF_SALTO} USD (ALERTA=DIF_NETO_SALTO) en {len(campos_salto)} campos")
        print(f"             {campos_salto[:10]}")
    return out


def _swap_breakeven_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Frontera de nombres MOTOR -> EQUIPO (renombre 2026-07-10, ver docs/DICCIONARIO.md).

    El motor (motor_breakeven.py, intocable) calcula:
      "financiero"  = precio NET donde NPV=0            -> piso INFERIOR (abandono;
                      abajo NO hay reservas; mata PDP+PNP+PND).
      "operacional" = precio NET donde FC del ultimo año con aceite = 0 (Limite
                      Economico) -> piso SUPERIOR (delta=0; mantener reservas).

    Convencion de equipo (validada finanzas 2026-06-09; renombrada 2026-07-10):
      PRECIO_EQUILIBRIO = piso INFERIOR (VPN=0). Antes "Breakeven Operacional".
      BREAKEVEN         = piso SUPERIOR (L.E.; mantener el L.E./reservas del FC).
                          Antes "Breakeven Financiero".

    Se renombra aqui, en la frontera de lectura, sin tocar el motor ni sus tests
    golden (tests/test_05_breakeven.py, que siguen con los nombres internos).
    """
    return df.rename(columns={
        "BREAKEVEN_FINANCIERO_USD_BBL":  "PRECIO_EQUILIBRIO_USD_BBL",  # motor NPV=0 -> piso inferior
        "BREAKEVEN_OPERACIONAL_USD_BBL": "BREAKEVEN_USD_BBL",          # motor FC ult.oil=0 (L.E.) -> piso superior
    })


def leer_breakeven() -> pd.DataFrame:
    """Lee el tablón de breakeven generado por 05_breakeven.py (Fase 0.5).
    Fallback: Breakeven.xlsm / Resultados Solver (solo vigencia 2024, VBA GoalSeek).
    Retorna CAMPO, VIGENCIA_BREAKEVEN, CLASE_RESERVA, BK_FIN, BK_OP, SOLVER_S1_FALLO,
    BRENT_INSENSITIVE.
    Las etiquetas FINANCIERO/OPERACIONAL ya vienen revertidas a convencion
    finanzas (ver _swap_breakeven_labels)."""
    pq = STAGING / "breakeven_resultados.parquet"
    if pq.exists():
        # Fuente primaria: motor Python (soporta 2024 y 2025, brentq robusto)
        df = pd.read_parquet(pq)
        df = df[df["CLASE_RESERVA"].isin({"D-PDP", "D-PNP", "D-PND"})].copy()
        if CAMPOS_PILOTO:
            df = df[df["CAMPO"].isin(CAMPOS_PILOTO)].copy()
        # BRENT_INSENSITIVE puede no existir en parquets generados con la versión antigua
        if "BRENT_INSENSITIVE" not in df.columns:
            df["BRENT_INSENSITIVE"] = False
        # BK_P* (perfil de salida) solo existen si 05 corrio con PRED_PERFIL_SALIDA
        _cols_perfil = [c for c in df.columns if c.startswith("BK_P")]
        df = df[["CAMPO", "VIGENCIA_BREAKEVEN", "CLASE_RESERVA",
                 "BREAKEVEN_OPERACIONAL_USD_BBL", "BREAKEVEN_FINANCIERO_USD_BBL",
                 "SOLVER_S1_FALLO", "BRENT_INSENSITIVE"] + _cols_perfil].copy()
        return _swap_breakeven_labels(df)

    # --- Fallback: Breakeven.xlsm (VBA, solo Cierre 2024, falla en ~80% de campos) ---
    print("  [WARN] breakeven_resultados.parquet no encontrado. "
          "Usando fallback Breakeven.xlsm (solo vigencia 2024).")
    df = pd.read_excel(RAW / "Breakeven.xlsm", sheet_name="Resultados Solver", engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    col_bk_op  = df.columns[5]   # Var. Final S1  = BREAKEVEN OPERACIONAL
    col_msg_s1 = df.columns[7]
    col_bk_fin = df.columns[11]  # Var. Final B.O. = BREAKEVEN FINANCIERO NPV=0
    col_msg_bo = df.columns[13]

    df = df.rename(columns={
        df.columns[0]: "ARCHIVO_ORIGEN",  df.columns[1]: "CLASE_RESERVA",
        col_bk_op:     "BREAKEVEN_OPERACIONAL_USD_BBL",
        col_msg_s1:    "MENSAJE_S1",
        col_bk_fin:    "BREAKEVEN_FINANCIERO_USD_BBL",
        col_msg_bo:    "MENSAJE_BO",
    })

    df = _homologar_columna(df, "ARCHIVO_ORIGEN", "Breakeven")
    df = df[df["CLASE_RESERVA"].isin({"D-PDP", "D-PNP", "D-PND"})].copy()
    if CAMPOS_PILOTO:
        df = df[df["CAMPO"].isin(CAMPOS_PILOTO)].copy()

    df["SOLVER_S1_FALLO"] = ~df["MENSAJE_S1"].astype(str).str.contains("OK", case=False, na=False)
    df["BREAKEVEN_OPERACIONAL_USD_BBL"] = pd.to_numeric(df["BREAKEVEN_OPERACIONAL_USD_BBL"], errors="coerce")
    df["BREAKEVEN_FINANCIERO_USD_BBL"]  = pd.to_numeric(df["BREAKEVEN_FINANCIERO_USD_BBL"],  errors="coerce")
    df["VIGENCIA_BREAKEVEN"] = VIGENCIA_BREAKEVEN_ACTUAL
    # El xlsm no genera BRENT_INSENSITIVE; por defecto False (VBA no lo detectaba)
    df["BRENT_INSENSITIVE"] = False
    df = df[["CAMPO", "VIGENCIA_BREAKEVEN", "CLASE_RESERVA",
             "BREAKEVEN_OPERACIONAL_USD_BBL", "BREAKEVEN_FINANCIERO_USD_BBL",
             "SOLVER_S1_FALLO", "BRENT_INSENSITIVE"]].copy()
    return _swap_breakeven_labels(df)


def calcular_breakeven_ponderado(df_bk: pd.DataFrame, df_res: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula, por CAMPO×VIGENCIA_BREAKEVEN, los precios de anclaje físico para el modelo.

    Convención equipo (post-swap, ver _swap_breakeven_labels; renombre 2026-07-10):
      BREAKEVEN         = piso SUPERIOR (L.E.; mantener reservas). Antes "financiero".
      PRECIO_EQUILIBRIO = piso INFERIOR (VPN=0; abandono). Antes "operacional".

    Columnas de salida:
      BREAKEVEN_USD_BBL         — ponderado por vol 1P (todas las clases), trazabilidad.
      PRECIO_EQUILIBRIO_USD_BBL — ponderado por vol PDP, trazabilidad.
      BK_ANCLA_FIN_USD_BBL  — piso SUPERIOR del anclaje. Criterio CLASE DE MAYOR
                               INCERTIDUMBRE (2026-06-11): BK financiero de la clase mas
                               riesgosa presente con volumen y solver valido (PND>PNP>PDP).
                               Sobre este precio, delta=0 (full reservas); abajo arranca
                               la escalera.
      BK_ANCLA_CLASE        — clase que fijo BK_ANCLA_FIN (D-PND/D-PNP/D-PDP) o
                               FALLBACK_GLOBAL si ninguna tenia volumen+solver.
      BK_ANCLA_PDP_USD_BBL  — piso INFERIOR de abandono (operacional D-PDP).
                               Bajo este precio, NINGUNA reserva 1P es económica.
      BRENT_INSENSITIVE      — True si TODAS las clases del campo son Brent-insensibles
                               (ingreso de gas/GLP fijo domina; el campo no tiene un
                               breakeven de Brent significativo).
      ALERTA_BK              — advertencias (BREAKEVEN_FALLBACK si se usó promedio simple;
                               ESCALERA_DEGENERADA si BK_ANCLA_FIN <= BK_ANCLA_PDP, caso en
                               el que se colapsa a un solo piso = BK_ANCLA_PDP).

    Economía por clase (escalera de 02_synthetic.py):
      Precio >= BK_ANCLA_FIN (financiero, superior):
        Predicción libre del modelo (delta=0 como piso, puede subir con Brent).
      BK_ANCLA_PDP <= Precio < BK_ANCLA_FIN (escalera):
        Solo sobrevive PDP (capex hundido); PNP+PND dejan de ser viables (NPV<0).
      Precio < BK_ANCLA_PDP (operacional, inferior):
        Abandono — ninguna reserva 1P es económica (vol=0).
    """
    # Compatibilidad hacia atrás: parquets generados antes de la columna de vigencia
    if "VIGENCIA_BREAKEVEN" not in df_bk.columns:
        df_bk = df_bk.copy()
        df_bk["VIGENCIA_BREAKEVEN"] = VIGENCIA_BREAKEVEN_ACTUAL
    if "BRENT_INSENSITIVE" not in df_bk.columns:
        df_bk = df_bk.copy()
        df_bk["BRENT_INSENSITIVE"] = False

    ultimo = (df_res.groupby("CAMPO")["AÑO"].max().reset_index()
              .rename(columns={"AÑO": "AÑO_MAX"}))
    df_ult = df_res.merge(ultimo, on="CAMPO")
    df_ult = df_ult[df_ult["AÑO"] == df_ult["AÑO_MAX"]]

    mapeo = {"D-PDP": "VOLUMEN_PDP_MBPE",
             "D-PNP": "VOLUMEN_PNP_MBPE",
             "D-PND": "VOLUMEN_PND_MBPE"}

    filas = []
    for (campo, vigencia), bk_cv in df_bk.groupby(["CAMPO", "VIGENCIA_BREAKEVEN"]):
        if CAMPOS_PILOTO and campo not in CAMPOS_PILOTO:
            continue
        vol_c = df_ult[df_ult["CAMPO"] == campo]

        # Acumuladores para el ponderado global (trazabilidad / BK_REFERENCIA)
        sum_v = sum_fin = sum_op = 0.0
        # BK financiero y volumen POR CLASE (para el criterio de mayor incertidumbre)
        bk_fin_por_clase = {}
        vol_por_clase = {}
        # PDP operacional para BK_ANCLA_PDP
        bk_ancla_pdp = np.nan
        # Salidas POR CLASE para la escalera multi-clase (Path D, 2026-06-11):
        # limite economico propio de PNP/PND — debajo, la clase deja de ser
        # bookeable. Si el solver no lo separa, la clase degrada a BK_ANCLA_FIN
        # (ponderado) en 02_synthetic.
        salidas_clase = {"D-PNP": np.nan, "D-PND": np.nan}
        # Perfil de salida volumetrico D-PDP (track Calidad, directriz §4bis.6): solo
        # presente si 05_breakeven corrio con PRED_PERFIL_SALIDA (columnas BK_P*).
        perfil_pdp = {}
        # Conteo de insensibles para flag de campo
        n_clases = n_insensibles = 0

        for clase, col_v in mapeo.items():
            fila_bk = bk_cv[bk_cv["CLASE_RESERVA"] == clase]
            if fila_bk.empty or col_v not in vol_c.columns:
                continue
            n_clases += 1
            es_insens = bool(fila_bk["BRENT_INSENSITIVE"].values[0])
            if es_insens:
                n_insensibles += 1
                continue

            vol_vals = vol_c[col_v].values
            v = float(vol_vals[0]) if len(vol_vals) > 0 and pd.notna(vol_vals[0]) else 0.0
            # Tras el swap (nombres de equipo): BREAKEVEN = FC ultimo año con aceite
            # (motor "operacional", solver S1) -> gateado por SOLVER_S1_FALLO.
            # PRECIO_EQUILIBRIO = NPV=0 (motor "financiero", brentq, casi siempre
            # converge) -> sin gate. bk_fin_c = superior (BREAKEVEN); bk_op_c = inferior.
            bk_fin_c = (float(fila_bk["BREAKEVEN_USD_BBL"].values[0])
                        if not fila_bk["SOLVER_S1_FALLO"].values[0] else np.nan)
            bk_op_c  = float(fila_bk["PRECIO_EQUILIBRIO_USD_BBL"].values[0])

            # Ponderado global (trazabilidad / BK_REFERENCIA en PBI)
            if not np.isnan(bk_fin_c) and v > 0:
                sum_v   += v
                sum_fin += bk_fin_c * v
                if not np.isnan(bk_op_c):
                    sum_op += bk_op_c * v

            # BK financiero y volumen por clase (insumo del criterio de incertidumbre)
            bk_fin_por_clase[clase] = bk_fin_c
            vol_por_clase[clase]    = v

            # BK_ANCLA_PDP: operacional D-PDP (piso de abandono; fallback al financiero
            # si el NPV=0 no esta disponible)
            if clase == "D-PDP":
                if not np.isnan(bk_op_c):
                    bk_ancla_pdp = bk_op_c
                elif not np.isnan(bk_fin_c):
                    bk_ancla_pdp = bk_fin_c

            # Salida por clase (Path D): limite economico propio, sin ponderar
            if clase in salidas_clase and not np.isnan(bk_fin_c):
                salidas_clase[clase] = bk_fin_c

            # Perfil de salida D-PDP: capturar BK_P* de la fila D-PDP (si existen)
            if clase == "D-PDP":
                for pc in [c for c in fila_bk.columns if c.startswith("BK_P")]:
                    val = fila_bk[pc].values[0]
                    if pd.notna(val):
                        perfil_pdp[pc] = float(val)

        # Campo Brent-insensible: todas las clases presentes son BRENT_INSENSITIVE
        brent_insens = (n_clases > 0) and (n_insensibles == n_clases)

        alerta_bk = ""
        if sum_v > 0:
            bk_fin_pond = sum_fin / sum_v
            bk_op_pond  = sum_op  / sum_v if sum_op > 0 else np.nan
        else:
            # Fallback: promedio simple cuando no hay volumen para ponderar
            # (ej. CASTILLA ESTE: solo D-PDP disponible, sin PNP/PND)
            bk_vals_fin = bk_cv[~bk_cv["SOLVER_S1_FALLO"]]["BREAKEVEN_USD_BBL"].dropna().values
            bk_vals_op  = bk_cv["PRECIO_EQUILIBRIO_USD_BBL"].dropna().values
            bk_fin_pond = float(np.mean(bk_vals_fin)) if len(bk_vals_fin) > 0 else np.nan
            bk_op_pond  = float(np.mean(bk_vals_op))  if len(bk_vals_op)  > 0 else np.nan
            if not np.isnan(bk_fin_pond) and not brent_insens:
                print(f"  [INFO] {campo} [{vigencia}]: breakeven fallback (sin vol) "
                      f"-> {bk_fin_pond:.2f}  [BREAKEVEN_FALLBACK - NO usar en soporte CAPEX]")
                alerta_bk = "BREAKEVEN_FALLBACK"

        # BK_ANCLA_FIN: criterio de CLASE DE MAYOR INCERTIDUMBRE (2026-06-11, supersede
        # ponderado PNP+PND). La reserva mas riesgosa presente con volumen y solver valido
        # (orden PND > PNP > PDP) fija el piso superior: es la primera que el auditor
        # desincorpora al bajar el precio. Mas robusto que el ponderado por volumen (que se
        # contamina con clases de breakeven extremo/degenerado, ej. anclas >120 USD/bbl) y
        # mas conservador en el gate dorado. Ver 06_comparativa_bk.py y MAESTRO §10.
        bk_ancla_fin = np.nan
        bk_ancla_clase = None
        for clase_inc in ("D-PND", "D-PNP", "D-PDP"):
            v_inc = vol_por_clase.get(clase_inc, 0.0)
            bk_inc = bk_fin_por_clase.get(clase_inc, np.nan)
            if v_inc > 0 and not np.isnan(bk_inc):
                bk_ancla_fin   = bk_inc
                bk_ancla_clase = clase_inc
                break
        if np.isnan(bk_ancla_fin):
            # Ninguna clase con volumen+solver valido → financiero global (mismo
            # comportamiento que el fallback previo; ej. CASTILLA ESTE solo D-PDP sin vol)
            bk_ancla_fin   = bk_fin_pond
            bk_ancla_clase = "FALLBACK_GLOBAL"

        # Guard: la convención finanzas exige FIN (superior) > PDP (inferior). Si el
        # motor no separa ambos pisos para este campo (escalera degenerada), colapsar
        # a un solo piso = BK_ANCLA_PDP (sin zona PDP intermedia) y alertar.
        if (pd.notna(bk_ancla_fin) and pd.notna(bk_ancla_pdp)
                and bk_ancla_fin <= bk_ancla_pdp):
            bk_ancla_fin = bk_ancla_pdp
            alerta_bk = "|".join(p for p in [alerta_bk, "ESCALERA_DEGENERADA"] if p)

        filas.append({
            "CAMPO":                          campo,
            "VIGENCIA_BREAKEVEN":             vigencia,
            "BREAKEVEN_USD_BBL":              bk_fin_pond,
            "PRECIO_EQUILIBRIO_USD_BBL":     bk_op_pond,
            "BK_ANCLA_FIN_USD_BBL":           bk_ancla_fin,
            "BK_ANCLA_PDP_USD_BBL":           bk_ancla_pdp,
            "BK_ANCLA_CLASE":                 bk_ancla_clase,
            "BK_SALIDA_PNP_USD_BBL":          salidas_clase["D-PNP"],
            "BK_SALIDA_PND_USD_BBL":          salidas_clase["D-PND"],
            "BRENT_INSENSITIVE":              brent_insens,
            "ALERTA_BK":                      alerta_bk,
            **perfil_pdp,   # BK_P10..BK_P90 solo en track Calidad
        })
    return pd.DataFrame(filas)


def construir_tablon(df_res: pd.DataFrame, df_px: pd.DataFrame,
                     df_bk_pond: pd.DataFrame, df_cons: pd.DataFrame) -> pd.DataFrame:
    """
    Ensambla el tablon CAMPO x AÑO x ESCENARIO con todas las columnas canonicas.

    Nuevas columnas (2026-06-04; re-semantizadas 2026-07-09):
      VIGENCIA, ES_BASELINE, NIVEL_DEFINICIONAL
      CHECKPOINT_1P_MBPE — cierre OFICIAL del MISMO año A (referencia del confound)
      BASELINE_1P_MBPE   — cierre OFICIAL del año ANTERIOR (A−1): la base real del deck
      DELTA_SENS_MBPE    — ΔReservas = SENSIBILIDAD − BASELINE_1P_MBPE (target del modelo)
      VIGENCIA_BREAKEVEN — cierre FC del breakeven
    """
    # Lookup de cierres A−1: el deck de Sensibilidad de una vigencia A se construye sobre
    # el último libro certificado DISPONIBLE en ese momento (cierre A−1), no el cierre A.
    bline_prev = (df_res[["CAMPO", "AÑO", "VOLUMEN_1P_OFICIAL_MBPE"]]
                  .assign(AÑO=lambda d: d["AÑO"] + 1)
                  .rename(columns={"VOLUMEN_1P_OFICIAL_MBPE": "BASELINE_1P_MBPE"}))

    # ── 1. Serie historica BASE ────────────────────────────────────────────────
    hist = df_res.merge(df_px, on=["CAMPO", "AÑO"], how="left")
    hist["ESCENARIO"]                    = "BASE"
    hist["VIGENCIA"]                     = hist["AÑO"].apply(
        lambda x: str(int(x)) if pd.notna(x) else "")
    hist["ES_BASELINE"]                  = True
    hist["ES_SINTETICO"]                 = False
    hist["VOLUMEN_1P_SENSIBILIDAD_MBPE"] = np.nan
    hist["CAMPO_ORIGEN_RAW"]             = hist["CAMPO"]
    hist["HOMOLOG_FLAG"]                 = "OK"
    hist["CHECKPOINT_1P_MBPE"]           = hist["VOLUMEN_1P_OFICIAL_MBPE"]
    hist = hist.merge(bline_prev, on=["CAMPO", "AÑO"], how="left")
    hist["DELTA_SENS_MBPE"]              = np.nan   # sin punto de sensibilidad en filas BASE

    # Incorporar ALERTA de la hoja Precio Net
    alerta_col = hist.pop("ALERTA_PRECIO") if "ALERTA_PRECIO" in hist.columns \
        else pd.Series("", index=hist.index)
    hist["ALERTA"] = alerta_col.fillna("")

    # ── 2. Consolidado ─────────────────────────────────────────────────────────
    cons = df_cons.copy()
    for c in ["VOLUMEN_PDP_MBPE", "VOLUMEN_PNP_MBPE", "VOLUMEN_PND_MBPE",
              "PRODUCCION_MBPE", "VOLUMEN_1P_OFICIAL_MBPE"]:
        cons[c] = np.nan
    cons["ES_BASELINE"] = False
    cons["ES_SINTETICO"] = False

    # Lookups de cierres: CHECKPOINT = OFICIAL del mismo CAMPO × AÑO (referencia del
    # confound); BASELINE = OFICIAL de A−1 (la base real del deck → target del modelo).
    # Guard: si df_cons venía vacío (0 columnas porque no hay datos de Consolidado),
    # el merge fallaría con KeyError 'CAMPO'. En ese caso agregamos columnas NaN.
    if not cons.empty and "CAMPO" in cons.columns:
        chk = (df_res[["CAMPO", "AÑO", "VOLUMEN_1P_OFICIAL_MBPE"]]
               .rename(columns={"VOLUMEN_1P_OFICIAL_MBPE": "CHECKPOINT_1P_MBPE"}))
        cons = cons.merge(chk, on=["CAMPO", "AÑO"], how="left")
        cons = cons.merge(bline_prev, on=["CAMPO", "AÑO"], how="left")
        cons["DELTA_SENS_MBPE"] = np.where(
            cons["VOLUMEN_1P_SENSIBILIDAD_MBPE"].notna() & cons["BASELINE_1P_MBPE"].notna(),
            cons["VOLUMEN_1P_SENSIBILIDAD_MBPE"] - cons["BASELINE_1P_MBPE"],
            np.nan
        )
        # Quarter con sensibilidad pero sin cierre A−1 (campo sin historia previa):
        # no entrena — visible, nunca descartado en silencio.
        sin_base = (cons["VOLUMEN_1P_SENSIBILIDAD_MBPE"].notna()
                    & cons["BASELINE_1P_MBPE"].isna())
        if "ALERTA" not in cons.columns:
            cons["ALERTA"] = ""
        cons.loc[sin_base, "ALERTA"] = cons.loc[sin_base, "ALERTA"].fillna("").apply(
            lambda a: "|".join(dict.fromkeys(p for p in [str(a), "SIN_BASELINE_ANTERIOR"] if p and p != "nan")))
    else:
        # Consolidado vacío: agregar columnas requeridas para la concatenación posterior
        cons["CHECKPOINT_1P_MBPE"] = pd.Series(dtype=float)
        cons["BASELINE_1P_MBPE"]   = pd.Series(dtype=float)
        cons["DELTA_SENS_MBPE"]    = pd.Series(dtype=float)

    # ── 3. Concatenar ──────────────────────────────────────────────────────────
    tablon = pd.concat([hist, cons], ignore_index=True)

    # ── 4. Nivel definicional ──────────────────────────────────────────────────
    # 2026+: incluye Monetizacion por Regalias → salto de nivel no correlacionado con precio
    tablon["NIVEL_DEFINICIONAL"] = np.where(
        tablon["AÑO"].fillna(0).astype(int) >= 2026, "2026_REGALIAS", "")

    # ── 5. Breakevens (merge vintage-keyed por CAMPO×AÑO) ─────────────────────
    # Para cada fila, elegir la VIGENCIA_BREAKEVEN disponible para ese campo que sea
    # la más reciente ≤ AÑO. Si no hay ninguna ≤ AÑO, usar la mínima disponible.
    bk_vigs_por_campo = (df_bk_pond.groupby("CAMPO")["VIGENCIA_BREAKEVEN"]
                         .apply(lambda s: sorted(s.unique())).to_dict())

    def _elegir_vig_bk(campo, ano):
        """Retorna la VIGENCIA_BREAKEVEN óptima para (campo, año)."""
        vigs = bk_vigs_por_campo.get(campo, [])
        if not vigs:
            return None
        ano_str = str(int(ano)) if pd.notna(ano) else "0"
        candidatas = [v for v in vigs if v <= ano_str]
        return candidatas[-1] if candidatas else vigs[0]

    tablon["_VIG_BK"] = tablon.apply(lambda r: _elegir_vig_bk(r["CAMPO"], r["AÑO"]), axis=1)

    # BK_P* (perfil de salida) solo existen si 05 corrio con PRED_PERFIL_SALIDA (Calidad)
    _cols_perfil = [c for c in df_bk_pond.columns if c.startswith("BK_P")]
    tablon = tablon.merge(
        df_bk_pond[["CAMPO", "VIGENCIA_BREAKEVEN",
                    "BREAKEVEN_USD_BBL", "PRECIO_EQUILIBRIO_USD_BBL",
                    "BK_ANCLA_FIN_USD_BBL", "BK_ANCLA_PDP_USD_BBL", "BK_ANCLA_CLASE",
                    "BK_SALIDA_PNP_USD_BBL", "BK_SALIDA_PND_USD_BBL",
                    "BRENT_INSENSITIVE", "ALERTA_BK"] + _cols_perfil].rename(
                        columns={"VIGENCIA_BREAKEVEN": "_VIG_BK"}),
        on=["CAMPO", "_VIG_BK"], how="left"
    )
    tablon.rename(columns={"_VIG_BK": "VIGENCIA_BREAKEVEN"}, inplace=True)

    # STALE: se usó un BK más antiguo que el AÑO de la fila (falta FC de ese cierre)
    ano_str  = tablon["AÑO"].fillna(0).astype(int).astype(str)
    vig_str  = tablon["VIGENCIA_BREAKEVEN"].fillna("0")
    stale    = ano_str > vig_str

    def _add_alert(row_a, new_a):
        # Normalizar: NaN/None/float → string vacío
        row_a = "" if (row_a is None or (isinstance(row_a, float) and row_a != row_a)) else str(row_a)
        new_a = "" if (new_a is None or (isinstance(new_a, float) and new_a != new_a)) else str(new_a)
        parts = [p for p in [row_a, new_a] if p]
        return "|".join(dict.fromkeys(parts))  # dedupe orden-preserving

    tablon.loc[stale, "ALERTA_BK"] = tablon.loc[stale, "ALERTA_BK"].apply(
        lambda a: _add_alert(a, "BREAKEVEN_VIGENCIA_STALE"))

    # ── 6. Combinar todas las alertas ──────────────────────────────────────────
    tablon["ALERTA"]    = tablon["ALERTA"].fillna("")
    tablon["ALERTA_BK"] = tablon["ALERTA_BK"].fillna("")
    no_hom = tablon["HOMOLOG_FLAG"].fillna("") == "NO_HOMOLOGADO"
    tablon.loc[no_hom, "ALERTA"] = tablon.loc[no_hom, "ALERTA"].apply(
        lambda a: _add_alert(a, "NO_HOMOLOGADO"))
    tablon["ALERTA"] = tablon.apply(
        lambda r: _add_alert(r["ALERTA"], r["ALERTA_BK"]), axis=1)
    tablon.drop(columns=["ALERTA_BK"], inplace=True)

    # ── 7. Columnas de prediccion (se llenan en 03_modelo.py) ─────────────────
    # Motor PRIMARIO = Isotonica; motor VALIDACION = Suave (PCHIP). XGBoost retirado
    # (ver 03_modelo.py: en 1D era el peor del benchmark y su restriccion invertia signo).
    for col in ["PRED_ISOTONICA_MBPE", "PRED_SUAVE_MBPE",
                "DELTA_ISOTONICA_VS_OFICIAL", "DELTA_SUAVE_VS_OFICIAL"]:
        tablon[col] = np.nan

    # ── 8. Orden canonico de columnas ─────────────────────────────────────────
    orden = [
        "CAMPO", "CAMPO_ORIGEN_RAW", "AÑO", "VIGENCIA", "ESCENARIO",
        "ES_BASELINE", "ES_SINTETICO", "NIVEL_DEFINICIONAL",
        "BRENT_FLAT_USD_BBL", "DESCUENTO_CALIDAD_USD_BBL",
        "DESCUENTO_TRANSPORTE_USD_BBL", "PRECIO_NETO_USD_BBL",
        "VOLUMEN_PDP_MBPE", "VOLUMEN_PNP_MBPE", "VOLUMEN_PND_MBPE",
        "VOLUMEN_1P_OFICIAL_MBPE", "CHECKPOINT_1P_MBPE", "BASELINE_1P_MBPE",
        "VOLUMEN_1P_SENSIBILIDAD_MBPE", "DELTA_SENS_MBPE",
        "BREAKEVEN_USD_BBL", "PRECIO_EQUILIBRIO_USD_BBL",
        "BK_ANCLA_FIN_USD_BBL", "BK_ANCLA_PDP_USD_BBL", "BK_ANCLA_CLASE",
        "BK_SALIDA_PNP_USD_BBL", "BK_SALIDA_PND_USD_BBL",
        "BRENT_INSENSITIVE",
        "VIGENCIA_BREAKEVEN",
        "PRED_ISOTONICA_MBPE", "PRED_SUAVE_MBPE",
        "DELTA_ISOTONICA_VS_OFICIAL", "DELTA_SUAVE_VS_OFICIAL",
        "ALERTA",
    ]
    # Perfil de salida (Calidad): conservar BK_P* si vinieron del merge, sin romper
    # el orden canonico de Produccion (no aparecen cuando el flag esta apagado).
    orden = orden + sorted(c for c in tablon.columns if c.startswith("BK_P"))
    for col in orden:
        if col not in tablon.columns:
            tablon[col] = np.nan
    return tablon[orden].sort_values(
        ["CAMPO", "AÑO", "ESCENARIO"]).reset_index(drop=True)


def flag_bk_contradicho_por_deck(tablon: pd.DataFrame) -> pd.DataFrame:
    """
    Marca ALERTA=BK_CONTRADICHO_POR_DECK en campos donde la evidencia del deck real
    contradice el umbral del breakeven (directriz DIRECTRIZ_ESCALERA_DECK.md §2). NO
    cambia curvas — alimenta la lista de revisión de Finanzas. Dos disparadores:

      (1) Libro vivo bajo el BK: existe punto real con Precio Neto <= BK_ANCLA_FIN + margen
          y delta > −5%·baseline (el deck dice que el campo vive donde la escalera lo mata).
      (2) Precipicio pegado a la banda (caso Caño Limón): la banda real entera está SOBRE
          el BK_ANCLA_FIN y el primer dato real cae a < BK_COHERENCIA_BANDA_USD del BK —
          la escalera sintética sostiene un precipicio profundo pegado a datos vivos.

    Retorna el DataFrame de revisión (una fila por campo flaggeado) y muta `tablon`
    agregando la ALERTA en las filas del campo.
    """
    real = tablon[(~tablon["ES_SINTETICO"]) & tablon["DELTA_SENS_MBPE"].notna()]
    filas_rev = []
    for campo, sub in real.groupby("CAMPO"):
        bk_fin = sub["BK_ANCLA_FIN_USD_BBL"].dropna()
        base   = sub["BASELINE_1P_MBPE"].dropna()
        if bk_fin.empty or base.empty:
            continue
        bk_fin = float(bk_fin.iloc[-1])
        baseline = float(base.iloc[-1])
        # Materialidad: micro-campos con BK absurdo (FC garbage) ya salen por ALERTA_BK;
        # esta lista es de revisión de Finanzas sobre campos con volumen real.
        if baseline < BK_COHERENCIA_BASELINE_MIN:
            continue

        pneto = sub["PRECIO_NETO_USD_BBL"]
        delta = sub["DELTA_SENS_MBPE"]
        banda_lo = float(pneto.min())
        banda_hi = float(pneto.max())

        # Disparador (1): punto real bajo/junto al BK que sigue vivo. Requiere que el BK
        # esté dentro del alcance de los datos (bk_fin <= banda_hi); si el BK está muy por
        # encima de la banda (FC absurdo) el problema es el BK, no una contradicción sutil.
        vivo_bajo_bk = bk_fin <= banda_hi and (
            (pneto <= bk_fin + BK_COHERENCIA_MARGEN_USD) &
            (delta > -BK_COHERENCIA_DELTA_FRAC * baseline)
        ).any()

        # Disparador (2): banda entera sobre el BK y pegada a él (precipicio a <$5)
        banda_sobre_bk   = banda_lo > bk_fin
        banda_pegada     = (banda_lo - bk_fin) < BK_COHERENCIA_BANDA_USD
        precipicio_banda = banda_sobre_bk and banda_pegada

        if not (vivo_bajo_bk or precipicio_banda):
            continue

        motivos = []
        if vivo_bajo_bk:
            motivos.append("libro-vivo-bajo-bk")
        if precipicio_banda:
            motivos.append("precipicio-pegado-a-banda")
        filas_rev.append({
            "CAMPO":              campo,
            "BK_ANCLA_FIN_USD_BBL": round(bk_fin, 2),
            "BANDA_REAL_LO_NETO": round(banda_lo, 2),
            "GAP_BANDA_BK_USD":   round(banda_lo - bk_fin, 2),
            "DELTA_MIN_BANDA_MBPE": round(float(delta.min()), 3),
            "DELTA_MIN_BANDA_PCT":  round(float(delta.min()) / baseline * 100, 1),
            "BASELINE_MBPE":      round(baseline, 2),
            "VIGENCIA_BREAKEVEN": sub["VIGENCIA_BREAKEVEN"].dropna().iloc[-1]
                                  if sub["VIGENCIA_BREAKEVEN"].notna().any() else "",
            "MOTIVO":             " | ".join(motivos),
        })
        mask_campo = tablon["CAMPO"] == campo
        tablon.loc[mask_campo, "ALERTA"] = tablon.loc[mask_campo, "ALERTA"].apply(
            lambda a: _add_alert(a, "BK_CONTRADICHO_POR_DECK"))

    return pd.DataFrame(filas_rev)


def _add_alert(row_a, new_a):
    """Fusiona ALERTA existente con una nueva etiqueta (dedupe, tolerante a NaN)."""
    row_a = "" if (row_a is None or (isinstance(row_a, float) and row_a != row_a)) else str(row_a)
    new_a = "" if (new_a is None or (isinstance(new_a, float) and new_a != new_a)) else str(new_a)
    parts = [p for p in [row_a, new_a] if p]
    return "|".join(dict.fromkeys(parts))


def validar_tablon(df: pd.DataFrame) -> None:
    """Resumen ejecutivo + validacion de invariantes del tablon."""
    sep = "-" * 70
    print(f"\n{sep}\n  Resumen tablon unico\n{sep}")
    print(f"  Filas totales               : {len(df)}")
    print(f"  Campos presentes            : {sorted(df['CAMPO'].unique())}")
    anios = df["AÑO"].dropna()
    print(f"  Años (rango)                : {int(anios.min())} - {int(anios.max())}")
    print(f"  Escenarios                  : {sorted(df['ESCENARIO'].unique())}")
    print(f"  BREAKEVEN_FALLBACK          : {df['ALERTA'].str.contains('BREAKEVEN_FALLBACK', na=False).sum()}")
    print(f"  BREAKEVEN_VIGENCIA_STALE    : {df['ALERTA'].str.contains('BREAKEVEN_VIGENCIA_STALE', na=False).sum()}")
    print(f"  TARGET_NULO                 : {df['ALERTA'].str.contains('TARGET_NULO', na=False).sum()}")
    print(f"  SIN_BASELINE_ANTERIOR       : {df['ALERTA'].str.contains('SIN_BASELINE_ANTERIOR', na=False).sum()}")
    print(f"  IDENTIDAD_PRECIO            : {df['ALERTA'].str.contains('IDENTIDAD_PRECIO', na=False).sum()}")
    print()

    for campo in sorted(df["CAMPO"].unique()):
        sub     = df[df["CAMPO"] == campo]
        bk_vals = sub["BREAKEVEN_USD_BBL"].dropna().values
        bk_str  = f"{bk_vals[0]:.2f}" if len(bk_vals) > 0 else "N/A"
        vbk     = sub["VIGENCIA_BREAKEVEN"].dropna().values
        vbk_str = vbk[0] if len(vbk) > 0 else "N/A"
        alertas = sub.loc[sub["ALERTA"] != "", "ALERTA"].unique()
        print(f"  {campo:<16} | filas={len(sub):3d} | vol={sub['VOLUMEN_1P_OFICIAL_MBPE'].notna().sum():3d} | "
              f"sens={sub['VOLUMEN_1P_SENSIBILIDAD_MBPE'].notna().sum():3d} | "
              f"delta={sub['DELTA_SENS_MBPE'].notna().sum():3d} | "
              f"BK={bk_str}[{vbk_str}]"
              + (f" | {set(alertas)}" if len(alertas) > 0 else ""))
    print()

    # Validaciones de invariantes
    # Solo verificar BRENT_FLAT en filas CONSOLIDADO (las entrenables); las filas BASE
    # de HIST sin hoja "Precio Net" son esperadas y no afectan el entrenamiento.
    cons_mask = df["ES_BASELINE"] == False
    warns = []
    n_null_brent = df.loc[cons_mask, "BRENT_FLAT_USD_BBL"].isnull().sum()
    n_null_bk    = df.loc[cons_mask, "BREAKEVEN_USD_BBL"].isnull().sum()
    n_pos_cal    = (df["DESCUENTO_CALIDAD_USD_BBL"].dropna()    > 0).sum()
    n_pos_tra    = (df["DESCUENTO_TRANSPORTE_USD_BBL"].dropna() > 0).sum()
    n_vig_nula   = (df["VIGENCIA"].isna() | (df["VIGENCIA"] == "")).sum()
    if n_null_brent:  warns.append(f"BRENT_FLAT nulos (CONSOLIDADO): {n_null_brent}")
    if n_null_bk:     warns.append(f"BREAKEVEN nulos (CONSOLIDADO): {n_null_bk}")
    if n_pos_cal:     warns.append(f"DESCUENTO_CALIDAD positivos: {n_pos_cal}")
    if n_pos_tra:     warns.append(f"DESCUENTO_TRANSPORTE positivos: {n_pos_tra}")
    if n_vig_nula:    warns.append(f"VIGENCIA nula: {n_vig_nula}")
    for w in warns:
        print(f"  [WARN] {w}")
    if not warns:
        print("  [OK] Todas las invariantes del tablon pasan.")
    print(sep + "\n")


if __name__ == "__main__":
    print("=== 01_etl.py — Iniciando ingesta ===\n")

    print("[1/6] Leyendo Brent oficial (HIST 1P / Precio Brent)...")
    df_brent_of = leer_brent_oficial()
    print(f"      {len(df_brent_of)} años | rango "
          f"{df_brent_of['BRENT_OFICIAL_USD_BBL'].min():.1f}–{df_brent_of['BRENT_OFICIAL_USD_BBL'].max():.1f}")

    print("[2/6] Leyendo HIST 1P / Reservas (MBPE)...")
    df_res, pesos_hist = leer_hist1p_reservas()
    print(f"      {len(df_res)} filas | campos: {sorted(df_res['CAMPO'].unique())}")

    print("[3/6] Leyendo HIST 1P / Precio Net (Brent remap + filtro excepciones)...")
    df_px = leer_hist1p_precio(df_brent_of, pesos_hist)
    print(f"      {len(df_px)} filas | años: {int(df_px['AÑO'].min())}–{int(df_px['AÑO'].max())}")

    print("[4/6] Leyendo Consolidado (multi-trimestre, fuente primaria)...")
    df_cons = leer_consolidado()
    print(f"      {len(df_cons)} filas | escenarios: "
          f"{sorted(df_cons['ESCENARIO'].unique()) if len(df_cons) else '[]'}")

    print("[5/6] Leyendo Breakeven y calculando ponderado...")
    df_bk = leer_breakeven()
    df_bk_pond = calcular_breakeven_ponderado(df_bk, df_res)
    for _, row in df_bk_pond.iterrows():
        alert_str = f"  [{row['ALERTA_BK']}]" if row.get("ALERTA_BK") else ""
        print(f"        {row['CAMPO']:<16} BK={row['BREAKEVEN_USD_BBL']:.2f}"
              f"  [{row['VIGENCIA_BREAKEVEN']}]{alert_str}")

    print("\n[6/6] Construyendo tablon unico...")
    tablon = construir_tablon(df_res, df_px, df_bk_pond, df_cons)

    # Coherencia BK↔deck (directriz §2): flag de revisión de Finanzas (no altera curvas)
    df_bk_rev = flag_bk_contradicho_por_deck(tablon)
    ruta_rev = RESULTADOS / "bk_revision_finanzas.csv"
    df_bk_rev.sort_values("GAP_BANDA_BK_USD").to_csv(
        ruta_rev, index=False, encoding="utf-8-sig") if not df_bk_rev.empty else \
        pd.DataFrame(columns=["CAMPO", "MOTIVO"]).to_csv(ruta_rev, index=False, encoding="utf-8-sig")
    print(f"  [BK↔deck] {len(df_bk_rev)} campos marcados BK_CONTRADICHO_POR_DECK -> {ruta_rev.name}")

    validar_tablon(tablon)

    tablon.to_parquet(STAGING / "tablon_unico.parquet", index=False)
    tablon.to_csv(STAGING / "tablon_unico.csv", index=False, encoding="utf-8-sig")
    print(f"Tablon guardado:\n  {STAGING / 'tablon_unico.parquet'}\n  {STAGING / 'tablon_unico.csv'}")
    print("\n=== 01_etl.py — Completado ===")
