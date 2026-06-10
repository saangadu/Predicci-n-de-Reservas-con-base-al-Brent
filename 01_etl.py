"""
01_etl.py — Ingesta y consolidacion de fuentes para el piloto Prediccion 1P vs Brent

Lee datos/raw/ y produce datos/staging/tablon_unico.parquet (+ .csv espejo).
Granularidad: CAMPO x AÑO x ESCENARIO.

Re-arquitectura 2026-06-04 (ver docs/MAESTRO.md §6):
  - VIGENCIA: año (BASE) o quarter del deck (CONSOLIDADO_YYYY_Qq).
  - ES_BASELINE: True para filas de serie historica oficial (ESCENARIO=BASE).
  - BASELINE_1P_VIGENCIA_MBPE: reservas OFICIAL del mismo CAMPO×AÑO (ancla de nivel).
  - DELTA_SENS_MBPE: ΔReservas = SENSIBILIDAD − BASELINE. Cancela saltos de nivel entre
    vigencias (deplecion, revisiones, Monetizacion por Regalias 2026). Target del modelo.
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

from homologacion import Homologador, reporte_homologacion

warnings.filterwarnings("ignore", category=UserWarning)

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RAW      = BASE_DIR / "datos" / "raw"
STAGING  = BASE_DIR / "datos" / "staging"
STAGING.mkdir(parents=True, exist_ok=True)

HIST_1P     = RAW / "HIST 1P.xlsx"
CONSOLIDADO = RAW / "Consolidado Bases de Datos Pronostico de Precios.xlsx"

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


def leer_hist1p_reservas() -> pd.DataFrame:
    """
    Lee HIST 1P / Reservas (MBPE). Trae columna CAMPO granular directa.
    Decimales con coma ("186,04") → float.
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

    cols = ["CAMPO", "AÑO", "PDP", "PNP", "PND", "1P", "PROD"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].rename(columns={
        "PDP": "VOLUMEN_PDP_MBPE", "PNP": "VOLUMEN_PNP_MBPE",
        "PND": "VOLUMEN_PND_MBPE", "1P":  "VOLUMEN_1P_OFICIAL_MBPE",
        "PROD": "PRODUCCION_MBPE",
    })


def leer_hist1p_precio(df_brent_of: pd.DataFrame) -> pd.DataFrame:
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

    return df[["CAMPO", "AÑO", "BRENT_FLAT_USD_BBL",
               "DESCUENTO_CALIDAD_USD_BBL", "DESCUENTO_TRANSPORTE_USD_BBL",
               "PRECIO_NETO_USD_BBL", "ALERTA_PRECIO"]].copy()


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
    # Layout 2026: col B(1)=Incluir, col D(3)=Nombre, Precio Neto E-M(4-12),
    # Desc Cal N-V(13-21), Desc Tra W-AE(22-30), Reservas 1P AF-AN(31-39)
    quarters = [str(raw.iloc[5, c]).strip() for c in range(4, 13)]

    NETO0, CAL0, TRA0, RES0 = 4, 13, 22, 31
    datos = raw.iloc[7:].reset_index(drop=True)

    # Brent de referencia por quarter (fila BRENT del deck — col D, índice 3)
    fila_brent = datos[datos[3].astype(str).str.strip().str.upper() == "BRENT"]
    brent_q = {}
    if not fila_brent.empty:
        r = fila_brent.iloc[0]
        for qi, q in enumerate(quarters):
            brent_q[q] = pd.to_numeric(r.iloc[NETO0 + qi], errors="coerce")

    # B5: Paso 1 — homologar TODOS los campos Incluir=Si para auditoria completa
    todos_hom = []
    for _, r in datos.iterrows():
        nombre = str(r.iloc[3]).strip()
        if not nombre or nombre.upper() in ("BRENT", "NAN", "NOMBRE"):
            continue
        incluir = str(r.iloc[1]).strip().upper()
        if incluir == "NO":
            continue
        campo, flag = H.homologar(nombre)
        todos_hom.append({"CAMPO": campo, "CAMPO_ORIGEN_RAW": nombre, "HOMOLOG_FLAG": flag})

    if todos_hom:
        reporte_homologacion(pd.DataFrame(todos_hom), "Consolidado")

    # Paso 2: procesar campos piloto con Incluir=Si
    filas = []
    for _, r in datos.iterrows():
        nombre = str(r.iloc[3]).strip()
        if not nombre or nombre.upper() in ("BRENT", "NAN", "NOMBRE"):
            continue
        incluir = str(r.iloc[1]).strip().upper()
        if incluir == "NO":
            continue
        campo, flag = H.homologar(nombre)
        if CAMPOS_PILOTO and campo not in CAMPOS_PILOTO:
            continue

        for qi, q in enumerate(quarters):
            neto = pd.to_numeric(r.iloc[NETO0 + qi], errors="coerce")
            if pd.isna(neto) or neto == 0:
                continue

            cal_raw = pd.to_numeric(r.iloc[CAL0 + qi], errors="coerce")
            tra_raw = pd.to_numeric(r.iloc[TRA0 + qi], errors="coerce")
            res     = pd.to_numeric(r.iloc[RES0 + qi], errors="coerce")
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

    # Dedupe: conservar curva con mayor volumen por CAMPO×ESCENARIO
    out["_orden"] = out["VOLUMEN_1P_SENSIBILIDAD_MBPE"].fillna(-1)
    out = (out.sort_values("_orden", ascending=False)
              .drop_duplicates(["CAMPO", "ESCENARIO"], keep="first")
              .drop(columns=["_orden"])
              .reset_index(drop=True))
    return out


def _swap_breakeven_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convencion finanzas (validada con el equipo financiero, 2026-06-09):
      FINANCIERO = piso SUPERIOR (delta=0; limite economico de PNP/PND).
      OPERACIONAL = piso INFERIOR (abandono; mata PDP+PNP+PND).
    El motor (motor_breakeven.py) los calcula al reves: su "financiero" es
    NPV=0 (mas bajo) y su "operacional" es FC del ultimo año con aceite (mas
    alto). Se revierten las etiquetas aqui, en la frontera de lectura, sin
    tocar el motor ni sus tests golden (tests/test_05_breakeven.py).
    """
    df = df.rename(columns={
        "BREAKEVEN_FINANCIERO_USD_BBL":  "_BK_TMP_USD_BBL",
        "BREAKEVEN_OPERACIONAL_USD_BBL": "BREAKEVEN_FINANCIERO_USD_BBL",
    })
    return df.rename(columns={"_BK_TMP_USD_BBL": "BREAKEVEN_OPERACIONAL_USD_BBL"})


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
        df = df[["CAMPO", "VIGENCIA_BREAKEVEN", "CLASE_RESERVA",
                 "BREAKEVEN_OPERACIONAL_USD_BBL", "BREAKEVEN_FINANCIERO_USD_BBL",
                 "SOLVER_S1_FALLO", "BRENT_INSENSITIVE"]].copy()
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

    Convención finanzas (post-swap, ver _swap_breakeven_labels):
      FINANCIERO = piso SUPERIOR (delta=0; límite económico de PNP/PND).
      OPERACIONAL = piso INFERIOR (abandono; mata PDP+PNP+PND).

    Columnas de salida:
      BREAKEVEN_FINANCIERO_USD_BBL  — ponderado por vol 1P (todas las clases), trazabilidad.
      BREAKEVEN_OPERACIONAL_USD_BBL — ponderado por vol PDP, trazabilidad.
      BK_ANCLA_FIN_USD_BBL  — piso SUPERIOR del anclaje (financiero PNP+PND ponderado).
                               Sobre este precio, delta=0 (full reservas); abajo arranca
                               la escalera.
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

        # Acumuladores para el ponderado global (trazabilidad)
        sum_v = sum_fin = sum_op = 0.0
        # Acumuladores para BK_ANCLA_FIN (financiero solo PNP+PND)
        sum_v_pnd = sum_fin_pnd = 0.0
        # PDP operacional para BK_ANCLA_PDP
        bk_ancla_pdp = np.nan
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
            # Tras el swap, FINANCIERO = FC ultimo año con aceite (motor "operacional",
            # solver S1) -> gateado por SOLVER_S1_FALLO. OPERACIONAL = NPV=0 (motor
            # "financiero", brentq, casi siempre converge) -> sin gate.
            bk_fin_c = (float(fila_bk["BREAKEVEN_FINANCIERO_USD_BBL"].values[0])
                        if not fila_bk["SOLVER_S1_FALLO"].values[0] else np.nan)
            bk_op_c  = float(fila_bk["BREAKEVEN_OPERACIONAL_USD_BBL"].values[0])

            # Ponderado global (trazabilidad)
            if not np.isnan(bk_fin_c) and v > 0:
                sum_v   += v
                sum_fin += bk_fin_c * v
                if not np.isnan(bk_op_c):
                    sum_op += bk_op_c * v

            # BK_ANCLA_PDP: operacional D-PDP (piso de abandono; fallback al financiero
            # si el NPV=0 no esta disponible)
            if clase == "D-PDP":
                if not np.isnan(bk_op_c):
                    bk_ancla_pdp = bk_op_c
                elif not np.isnan(bk_fin_c):
                    bk_ancla_pdp = bk_fin_c

            # BK_ANCLA_FIN: financiero (piso superior, delta=0) ponderado de PNP y PND
            if clase in ("D-PNP", "D-PND") and not np.isnan(bk_fin_c) and v > 0:
                sum_v_pnd   += v
                sum_fin_pnd += bk_fin_c * v

        # Campo Brent-insensible: todas las clases presentes son BRENT_INSENSITIVE
        brent_insens = (n_clases > 0) and (n_insensibles == n_clases)

        alerta_bk = ""
        if sum_v > 0:
            bk_fin_pond = sum_fin / sum_v
            bk_op_pond  = sum_op  / sum_v if sum_op > 0 else np.nan
        else:
            # Fallback: promedio simple cuando no hay volumen para ponderar
            # (ej. CASTILLA ESTE: solo D-PDP disponible, sin PNP/PND)
            bk_vals_fin = bk_cv[~bk_cv["SOLVER_S1_FALLO"]]["BREAKEVEN_FINANCIERO_USD_BBL"].dropna().values
            bk_vals_op  = bk_cv["BREAKEVEN_OPERACIONAL_USD_BBL"].dropna().values
            bk_fin_pond = float(np.mean(bk_vals_fin)) if len(bk_vals_fin) > 0 else np.nan
            bk_op_pond  = float(np.mean(bk_vals_op))  if len(bk_vals_op)  > 0 else np.nan
            if not np.isnan(bk_fin_pond) and not brent_insens:
                print(f"  [INFO] {campo} [{vigencia}]: breakeven fallback (sin vol) "
                      f"-> {bk_fin_pond:.2f}  [BREAKEVEN_FALLBACK - NO usar en soporte CAPEX]")
                alerta_bk = "BREAKEVEN_FALLBACK"

        # BK_ANCLA_FIN: financiero ponderado PNP+PND (o fallback al global si no hay PNP/PND)
        if sum_v_pnd > 0:
            bk_ancla_fin = sum_fin_pnd / sum_v_pnd
        else:
            # Solo hay PDP (sin PNP/PND con volumen) → usar el financiero global como piso
            bk_ancla_fin = bk_fin_pond

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
            "BREAKEVEN_FINANCIERO_USD_BBL":   bk_fin_pond,
            "BREAKEVEN_OPERACIONAL_USD_BBL":  bk_op_pond,
            "BK_ANCLA_FIN_USD_BBL":           bk_ancla_fin,
            "BK_ANCLA_PDP_USD_BBL":           bk_ancla_pdp,
            "BRENT_INSENSITIVE":              brent_insens,
            "ALERTA_BK":                      alerta_bk,
        })
    return pd.DataFrame(filas)


def construir_tablon(df_res: pd.DataFrame, df_px: pd.DataFrame,
                     df_bk_pond: pd.DataFrame, df_cons: pd.DataFrame) -> pd.DataFrame:
    """
    Ensambla el tablon CAMPO x AÑO x ESCENARIO con todas las columnas canonicas.

    Nuevas columnas (2026-06-04):
      VIGENCIA, ES_BASELINE, NIVEL_DEFINICIONAL
      BASELINE_1P_VIGENCIA_MBPE — reservas OFICIAL del año de la vigencia
      DELTA_SENS_MBPE           — ΔReservas = SENSIBILIDAD − BASELINE (target del modelo)
      VIGENCIA_BREAKEVEN        — cierre FC del breakeven
    """
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
    hist["BASELINE_1P_VIGENCIA_MBPE"]    = hist["VOLUMEN_1P_OFICIAL_MBPE"]
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

    # Lookup baseline: OFICIAL del mismo CAMPO × AÑO.
    # Guard: si df_cons venía vacío (0 columnas porque no hay datos de Consolidado),
    # el merge fallaría con KeyError 'CAMPO'. En ese caso agregamos columnas NaN.
    if not cons.empty and "CAMPO" in cons.columns:
        bline = (df_res[["CAMPO", "AÑO", "VOLUMEN_1P_OFICIAL_MBPE"]]
                 .rename(columns={"VOLUMEN_1P_OFICIAL_MBPE": "BASELINE_1P_VIGENCIA_MBPE"}))
        cons = cons.merge(bline, on=["CAMPO", "AÑO"], how="left")
        cons["DELTA_SENS_MBPE"] = np.where(
            cons["VOLUMEN_1P_SENSIBILIDAD_MBPE"].notna() & cons["BASELINE_1P_VIGENCIA_MBPE"].notna(),
            cons["VOLUMEN_1P_SENSIBILIDAD_MBPE"] - cons["BASELINE_1P_VIGENCIA_MBPE"],
            np.nan
        )
    else:
        # Consolidado vacío: agregar columnas requeridas para la concatenación posterior
        cons["BASELINE_1P_VIGENCIA_MBPE"] = pd.Series(dtype=float)
        cons["DELTA_SENS_MBPE"]           = pd.Series(dtype=float)

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

    tablon = tablon.merge(
        df_bk_pond[["CAMPO", "VIGENCIA_BREAKEVEN",
                    "BREAKEVEN_FINANCIERO_USD_BBL", "BREAKEVEN_OPERACIONAL_USD_BBL",
                    "BK_ANCLA_FIN_USD_BBL", "BK_ANCLA_PDP_USD_BBL",
                    "BRENT_INSENSITIVE", "ALERTA_BK"]].rename(
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
    for col in ["PRED_XGBOOST_MBPE", "PRED_ISOTONICA_MBPE",
                "DELTA_XGBOOST_VS_OFICIAL", "DELTA_ISOTONICA_VS_OFICIAL"]:
        tablon[col] = np.nan

    # ── 8. Orden canonico de columnas ─────────────────────────────────────────
    orden = [
        "CAMPO", "CAMPO_ORIGEN_RAW", "AÑO", "VIGENCIA", "ESCENARIO",
        "ES_BASELINE", "ES_SINTETICO", "NIVEL_DEFINICIONAL",
        "BRENT_FLAT_USD_BBL", "DESCUENTO_CALIDAD_USD_BBL",
        "DESCUENTO_TRANSPORTE_USD_BBL", "PRECIO_NETO_USD_BBL",
        "VOLUMEN_PDP_MBPE", "VOLUMEN_PNP_MBPE", "VOLUMEN_PND_MBPE",
        "VOLUMEN_1P_OFICIAL_MBPE", "BASELINE_1P_VIGENCIA_MBPE",
        "VOLUMEN_1P_SENSIBILIDAD_MBPE", "DELTA_SENS_MBPE",
        "BREAKEVEN_FINANCIERO_USD_BBL", "BREAKEVEN_OPERACIONAL_USD_BBL",
        "BK_ANCLA_FIN_USD_BBL", "BK_ANCLA_PDP_USD_BBL",
        "BRENT_INSENSITIVE",
        "VIGENCIA_BREAKEVEN",
        "PRED_XGBOOST_MBPE", "PRED_ISOTONICA_MBPE",
        "DELTA_XGBOOST_VS_OFICIAL", "DELTA_ISOTONICA_VS_OFICIAL",
        "ALERTA",
    ]
    for col in orden:
        if col not in tablon.columns:
            tablon[col] = np.nan
    return tablon[orden].sort_values(
        ["CAMPO", "AÑO", "ESCENARIO"]).reset_index(drop=True)


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
    print(f"  IDENTIDAD_PRECIO            : {df['ALERTA'].str.contains('IDENTIDAD_PRECIO', na=False).sum()}")
    print()

    for campo in sorted(df["CAMPO"].unique()):
        sub     = df[df["CAMPO"] == campo]
        bk_vals = sub["BREAKEVEN_FINANCIERO_USD_BBL"].dropna().values
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
    n_null_bk    = df.loc[cons_mask, "BREAKEVEN_FINANCIERO_USD_BBL"].isnull().sum()
    n_pos_cal    = (df["DESCUENTO_CALIDAD_USD_BBL"].dropna()    > 0).sum()
    n_pos_tra    = (df["DESCUENTO_TRANSPORTE_USD_BBL"].dropna() > 0).sum()
    n_vig_nula   = (df["VIGENCIA"].isna() | (df["VIGENCIA"] == "")).sum()
    if n_null_brent:  warns.append(f"BRENT_FLAT nulos (CONSOLIDADO): {n_null_brent}")
    if n_null_bk:     warns.append(f"BREAKEVEN_FINANCIERO nulos (CONSOLIDADO): {n_null_bk}")
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
    df_res = leer_hist1p_reservas()
    print(f"      {len(df_res)} filas | campos: {sorted(df_res['CAMPO'].unique())}")

    print("[3/6] Leyendo HIST 1P / Precio Net (Brent remap + filtro excepciones)...")
    df_px = leer_hist1p_precio(df_brent_of)
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
        print(f"        {row['CAMPO']:<16} BK_fin={row['BREAKEVEN_FINANCIERO_USD_BBL']:.2f}"
              f"  [{row['VIGENCIA_BREAKEVEN']}]{alert_str}")

    print("\n[6/6] Construyendo tablon unico...")
    tablon = construir_tablon(df_res, df_px, df_bk_pond, df_cons)
    validar_tablon(tablon)

    tablon.to_parquet(STAGING / "tablon_unico.parquet", index=False)
    tablon.to_csv(STAGING / "tablon_unico.csv", index=False, encoding="utf-8-sig")
    print(f"Tablon guardado:\n  {STAGING / 'tablon_unico.parquet'}\n  {STAGING / 'tablon_unico.csv'}")
    print("\n=== 01_etl.py — Completado ===")
