"""
04_pbi_export.py — Matrices de prediccion para Power BI (Modelo 2 + Modelo 1 + cadena)

Pre-calcula las combinaciones (Brent x Campo x Motor) como CSV estatico.
PBI consume el CSV con DAX + What-If slider de Brent.

ARQUITECTURA (2026-06-11; reformada 2026-06-12 — sin escenarios, con re-anclaje):
  Para cada Brent de la grilla:
    1. Modelo 2 (03b_correlacion_brent.py): Brent -> PRECIO ACEITE por campo
       (recta unica BASE; escenarios BAJO/ALTO retirados).
    2. Modelo 1 (03_modelo.py): PRECIO ACEITE -> DELTA Reservas. Motores 1D:
       - Isotonica (PRIMARIO), Suave/PCHIP (VALIDACION). XGBoost retirado.
    3. RE-ANCLAJE: VOLUMEN = max(BASELINE + [f(p) − f(p_ref)], 0), piso duro
       p < BK_ANCLA_PDP -> 0. p_ref/delta_ref vienen de metricas.csv (03).
       Garantia: en Brent=BRENT_REF (ultimo quarter conocido) Vol = baseline exacto.

  ES_VIABLE        = precio_aceite >= piso EFECTIVO (PRECIO_EQUILIBRIO/abandono capado
                     bajo p_ref cuando >= p_ref con baseline > 0; auditoria 2026-07-07 H1).
  ES_FULL_RESERVAS = precio_aceite >= BREAKEVEN  (piso superior; mantener reservas).
  ALERTA_BK        = BK_SUPERA_PRECIO_REF (BK falsificado por la certificacion,
                     piso capado) | SIN_REANCLAJE (campo sin p_ref, curva sin shift).
  ES_EXTRAPOLADO   = Brent fuera de la banda observada del Consolidado por campo ± margen.
  ES_CLIPPED       = Brent por ENCIMA del techo de la banda del deck: la isotonica esta
                     saturada (out_of_bounds='clip') y el volumen es techo de recuperacion.
  M2_ES_FALLBACK   = el campo no tiene relacion Aceite~Brent propia (k de portafolio).

TRES MATRICES (aislar el origen de errores, directriz 2026-06-12):
  output_matriz_modelo1.csv   grilla Precio Aceite -> delta anclado/volumen (M1 puro)
  output_matriz_modelo2.csv   grilla Brent -> Precio Aceite + metricas (M2 puro)
  output_matriz_prediccion.csv  cadena completa Brent -> Aceite -> Volumen

Confianza por campo: usa MAE_LOO del PRIMARIO (Isotonica) y la divergencia Isotonica↔Suave.
"""

import importlib
import shutil
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from motores_modelo1 import piso_efectivo, volumen_anclado
from track import sufijo_track, flag

_SUF = sufijo_track()   # '' Produccion; '_calidad' si PRED_TRACK=calidad
BASE_DIR    = Path(__file__).parent
STAGING     = BASE_DIR / "datos" / f"staging{_SUF}"
MODELOS_DIR = STAGING / "modelos"
RESULTADOS  = BASE_DIR / f"resultados{_SUF}"
RESULTADOS.mkdir(parents=True, exist_ok=True)

# Modelo 2: predictor Brent -> Neto (import dinamico: el nombre empieza por digito)
m2 = importlib.import_module("03b_correlacion_brent")

BRENT_PASO = 1
MARGEN_BRENT_USD  = 10    # margen sobre la banda del Consolidado para el meshgrid
MARGEN_EXTRAP_USD = 5.0   # margen para marcar ES_EXTRAPOLADO por campo

# ── Umbrales de confianza (ajustables) ───────────────────────────────────────
CONFIANZA_N_REAL_MIN    = 6
CONFIANZA_BASELINE_MIN  = 0.5
CONFIANZA_MAE_REL_ALTA  = 0.20
CONFIANZA_MAE_REL_MEDIA = 0.40
CONFIANZA_DIV_ALTA      = 0.30
CONFIANZA_DIV_MEDIA     = 0.50
CONFIANZA_MAE_ABS_ALTA  = 2.0
CONFIANZA_MAE_ABS_MEDIA = 5.0
CONFIANZA_OUTLIER_RATIO = 2.0
# Cap de outlier condicionado a materialidad (2026-06-11): el flag OUTLIER_LOO
# (RMSE/MAE>2) solo degrada si el error relativo es material. Caso RUBIALES: el
# ratio lo dispara un unico fold en la frontera de vigencias (revision +21.8 MBPE
# del baseline 2025 → a mismo neto conviven deltas de dos vigencias) con MAE_REL
# 0.9% sobre 323 MBPE — inmaterial para CAPEX.
CONFIANZA_MAE_REL_OUTLIER = 0.05
# Gate de skill (2026-06-11): ALTA exige ademas que el modelo SUPERE al predictor
# ingenuo (SKILL_ISO > 0) — error bajo + motores de acuerdo no implican que el
# modelo extraiga la señal de precio (caso AKACIAS: MAE_rel 12% pero SKILL=0).
# Exencion: campos "planos" (MAE_NAIVE < CONFIANZA_MAE_ABS_ALTA) — alli la media
# ingenua es imbatible por construccion y SKILL<=0 no indica un mal modelo.
CONFIANZA_SKILL_MIN = 0.0
# Materialidad para etiquetas de riesgo (2026-07-01): en campos grandes una
# etiqueta ALTA no debe leerse como "confio en el +X%" si el modelo NO extrae
# senal de precio (plano exento del gate de skill) o si M2 es fragil (sin
# historia / n<5 / R2_LOO<0). Se conserva el nivel pero se marca el motivo.
CONFIANZA_BASELINE_MATERIAL = 50.0
# Confound de vigencia (2026-07-02, informe §3ter): en varios campos el "escalon
# de precio" que aprende la isotonica es en realidad la diferencia entre los
# pronosticos de dos AÑOS de Consolidado distintos (2024 delta≈+15, 2025 ≈-30 en
# CASTILLA) cuyas bandas de precio no se solapan. Elasticidad intra-vigencia ≈ 0
# → la sensibilidad al precio esta NO IDENTIFICADA en banda. Criterio de flag:
#   - η² ≥ 0.8: el año de vigencia explica ≥80% de la varianza del delta
#   - solape < $2: las bandas de precio neto de los años casi no se tocan
#     (si solapan ampliamente, el efecto precio SI es separable del año)
#   - salto ≥ 5% del baseline: el escalon inter-año es material
CONFOUND_ETA2_MIN       = 0.8
CONFOUND_SOLAPA_MAX_USD = 2.0
CONFOUND_SALTO_REL_MIN  = 0.05
# Materialidad propia del confound (≥20 MBPE, mas estricta que los 50 de las
# otras etiquetas): el salto espurio escala con el baseline y ya es relevante
# para CAPEX en campos medianos (LA CIRA 44, CUSIANA 32, YARIGUI 32).
CONFOUND_BASELINE_MATERIAL = 20.0
# Sesgo de recuperacion de vigencia (2026-07-10, artifact 8a8f2cbc): la isotonica
# lee el DECLINO de reservas 2024->2026 como elasticidad de precio. El sintoma es
# un |d_ref|/baseline grande: el ancla p_ref cae en el fondo del valle de la ultima
# vigencia y la curva "recupera" el nivel de 2024 al subir el Brent (Casabe +40%,
# Guando +56%). A diferencia de los otros caps, este NO se condiciona a materialidad:
# el problema es mas visible justamente en maduros pequeños (Casabe 14, Caño Limon
# 12 MBPE) que hoy salen ALTA con +40%. Cap a MEDIA con motivo 'sesgo-recuperacion'.
CONFIANZA_SESGO_RECUP_MAX = 0.15

HISTORICO_DIR = RESULTADOS / "historico_predicciones"

# Motores 1D: PRIMARIO (Isotonica/hibrido) y VALIDACION (Suave u otro 2.o LOYO s12).
# Sufijo de archivo joblib y label PBI (legado: "Isotonica"/"Suave" — los nombres
# reales del motor van en METODO_REAL si hay metricas).
MOTORES = [("Isotonica", "iso"), ("Suave", "suave")]

# Grilla de Precio Aceite para la matriz M1 pura (independiente del Brent)
PNETO_GRID_MIN, PNETO_GRID_MAX, PNETO_GRID_PASO = 20.0, 110.0, 1.0


def siguiente_quarter(q: str) -> str:
    """Avanza un quarter: '2026_Q1' -> '2026_Q2', '2025_Q4' -> '2026_Q1'."""
    anio, qn = q.split("_Q")
    anio, qn = int(anio), int(qn)
    if qn == 4:
        return f"{anio + 1}_Q1"
    return f"{anio}_Q{qn + 1}"


def derivar_vigencias(df: pd.DataFrame) -> tuple[str, str]:
    """Ultimo quarter CONSOLIDADO_* CON RESERVAS -> (base, objetivo).

    El loop rodante avanza cuando llegan las RESERVAS, no cuando llega el precio:
    siempre recibimos el precio de un quarter un trimestre antes que sus reservas.
    Un quarter con precio pero sin reservas (ALERTA=TARGET_NULO) es el OBJETIVO a
    predecir, no la base. Por eso se filtra a quarters con VOLUMEN_1P_SENSIBILIDAD.
    """
    con = df[df["ESCENARIO"].astype(str).str.startswith("CONSOLIDADO_")
             & df["VOLUMEN_1P_SENSIBILIDAD_MBPE"].notna()]
    qs = [s.replace("CONSOLIDADO_", "") for s in con["ESCENARIO"].dropna().unique()]
    if not qs:
        return "DESCONOCIDA", "DESCONOCIDA"
    vigencia_base = sorted(qs)[-1]
    return vigencia_base, siguiente_quarter(vigencia_base)


def construir_puntos_reales(df, filas_m1, filas_m2, baselines, anclaje):
    """Une curvas del modelo + puntos reales + baseline + ancla para replicar en
    Power BI los dos plots de datos/staging (espacio delta y volumen absoluto) y
    el plot M2.

    `SERIE` lleva el detalle (incluye el Q de la Sensibilidad, ej. 'Sensibilidad
    2024-Q3'); `SERIE_COLOR` agrupa por año para que el color/legend del tablero
    no explote en un color por trimestre (directriz usuario 2026-07-16: Q en la
    etiqueta, color agrupado por vigencia/año).
    No incluye sinteticos ni breakeven: solo modelo + sensibilidad + baseline + ancla.
    - M1: sensibilidad = CONSOLIDADO (~ES_SINTETICO & ~ES_BASELINE & DELTA_SENS notna,
          por VIGENCIA/Q). Volumen graficado = baseline_latest + DELTA_SENS (mismo
          criterio que el panel derecho de 03_modelo: delta anclado al baseline vigente).
          Baseline = cierres oficiales anuales (ES_BASELINE), mismos puntos que M2.
    - M2: reales = HIST cierres anuales (ES_BASELINE & BRENT_FLAT/PRECIO_NETO notna),
          mismos puntos con que se ajusta la recta Theil-Sen (03b).
    """
    # ── M1: curva iso/suave + sensibilidad por Q + baseline anual + ancla ────
    reg_m1 = []
    for r in filas_m1:
        serie = "Isotónica" if r["MOTOR"] == "Isotonica" else "Suave"
        reg_m1.append({
            "CAMPO": r["CAMPO"],
            "SERIE": serie,
            "SERIE_COLOR": serie,
            "AÑO": None,
            "PRECIO_NETO_USD_BBL": r["PRECIO_ACEITE_USD_BBL"],
            "DELTA_MBPE": r["DELTA_ANCLADO_MBPE"],
            "VOLUMEN_MBPE": r["VOLUMEN_1P_PREDICHO_MBPE"],
        })
    reales = df[(~df["ES_SINTETICO"]) & (~df["ES_BASELINE"])
                & df["DELTA_SENS_MBPE"].notna() & df["PRECIO_NETO_USD_BBL"].notna()]
    for _, r in reales.iterrows():
        base = baselines.get(r["CAMPO"], np.nan)
        if pd.isna(base):
            continue
        anio = int(r["AÑO"])
        vig_q = str(r["VIGENCIA"]).replace("_Q", "-Q")
        reg_m1.append({
            "CAMPO": r["CAMPO"],
            "SERIE": f"Sensibilidad {vig_q}",
            "SERIE_COLOR": f"Sensibilidad {anio}",
            "AÑO": anio,
            "PRECIO_NETO_USD_BBL": round(float(r["PRECIO_NETO_USD_BBL"]), 2),
            "DELTA_MBPE": round(float(r["DELTA_SENS_MBPE"]), 2),
            "VOLUMEN_MBPE": round(float(base + r["DELTA_SENS_MBPE"]), 2),
        })
    # Baseline: cierres oficiales anuales (mismos puntos que alimentan M2 Theil-Sen,
    # ver bloque `hist` abajo) — el usuario quiere verlos también en el espacio M1.
    hist_baseline = df[(~df["ES_SINTETICO"]) & df["ES_BASELINE"]
                       & df["VOLUMEN_1P_OFICIAL_MBPE"].notna()
                       & df["PRECIO_NETO_USD_BBL"].notna()]
    for _, r in hist_baseline.iterrows():
        base = baselines.get(r["CAMPO"], np.nan)
        anio = int(r["AÑO"])
        vol_of = float(r["VOLUMEN_1P_OFICIAL_MBPE"])
        reg_m1.append({
            "CAMPO": r["CAMPO"],
            "SERIE": f"Baseline {anio}",
            "SERIE_COLOR": f"Baseline {anio}",
            "AÑO": anio,
            "PRECIO_NETO_USD_BBL": round(float(r["PRECIO_NETO_USD_BBL"]), 2),
            "DELTA_MBPE": round(vol_of - float(base), 2) if pd.notna(base) else None,
            "VOLUMEN_MBPE": round(vol_of, 2),
        })
    for campo, base in baselines.items():
        p_ref = anclaje.get(campo, {}).get("P_REF_USD_BBL", np.nan)
        if pd.isna(p_ref) or pd.isna(base):
            continue
        reg_m1.append({
            "CAMPO": campo, "SERIE": "Ancla", "SERIE_COLOR": "Ancla", "AÑO": None,
            "PRECIO_NETO_USD_BBL": round(float(p_ref), 2),
            "DELTA_MBPE": 0.0, "VOLUMEN_MBPE": round(float(base), 2),
        })
    df_m1 = pd.DataFrame(reg_m1)

    # ── M2: recta (grilla) + reales HIST por año ─────────────────────────────
    reg_m2 = []
    for r in filas_m2:
        reg_m2.append({
            "CAMPO": r["CAMPO"], "SERIE": "Recta", "AÑO": None,
            "BRENT_USD_BBL": r["BRENT_USD_BBL"],
            "PRECIO_NETO_USD_BBL": r["PRECIO_ACEITE_USD_BBL"],
        })
    hist = df[(~df["ES_SINTETICO"]) & df["ES_BASELINE"]
              & df["BRENT_FLAT_USD_BBL"].notna() & df["PRECIO_NETO_USD_BBL"].notna()]
    for _, r in hist.iterrows():
        reg_m2.append({
            "CAMPO": r["CAMPO"], "SERIE": f"Real {int(r['AÑO'])}", "AÑO": int(r["AÑO"]),
            "BRENT_USD_BBL": round(float(r["BRENT_FLAT_USD_BBL"]), 2),
            "PRECIO_NETO_USD_BBL": round(float(r["PRECIO_NETO_USD_BBL"]), 2),
        })
    df_m2 = pd.DataFrame(reg_m2)

    # IDX: granularidad por (campo, serie) para el scatter de Power BI (evita
    # que agregue puntos con el mismo precio; cada punto es una fila unica)
    # AÑO como entero nullable → escribe "2024"/"" (no "2024.0") para tipar limpio en M
    for _d in (df_m1, df_m2):
        if not _d.empty:
            _d.insert(0, "IDX", _d.groupby(["CAMPO", "SERIE"]).cumcount())
            _d["AÑO"] = _d["AÑO"].astype("Int64")
    return df_m1, df_m2


def cargar_correlacion() -> dict:
    """Coeficientes del Modelo 2 por campo (correlacion_brent.csv): {campo: dict_coef}."""
    ruta = STAGING / "correlacion_brent.csv"
    if not ruta.exists():
        raise FileNotFoundError("correlacion_brent.csv no existe: correr 03b_correlacion_brent.py")
    df = pd.read_csv(ruta)
    return {r["CAMPO"]: r.to_dict() for _, r in df.iterrows()}


# Umbrales de fragilidad M2 (informe 2026-07-01 WS2.2): un campo material cuya
# recta Brent->Aceite se apoya en pocos puntos, sin R2_LOO positivo, o en el
# fallback de portafolio, no sostiene con solidez el salto de precio que
# arrastra a M1. Se marca (no se descarta): la cadena sigue siendo la mejor
# aproximacion disponible, pero la confianza debe reflejarlo.
M2_FRAGIL_N_MIN = 5


def es_m2_fragil(coef: dict) -> bool:
    """True si el Modelo 2 del campo es fragil: fallback de portafolio,
    menos de M2_FRAGIL_N_MIN puntos HIST, o R2_LOO no positivo (out-of-sample)."""
    if coef is None:
        return True
    if bool(coef.get("ES_FALLBACK", False)):
        return True
    n_puntos = coef.get("N_PUNTOS")
    if pd.isna(n_puntos) or n_puntos < M2_FRAGIL_N_MIN:
        return True
    r2_loo = coef.get("R2_LOO")
    if pd.notna(r2_loo) and r2_loo < 0:
        return True
    return False


def calcular_confound_vigencia(df: pd.DataFrame, baselines: dict) -> dict:
    """Detecta por campo el confound de vigencia (informe 2026-07-02 §3ter):
    el año de Consolidado explica el delta (η² alto), las bandas de precio de
    los años no se solapan, y el salto inter-año es material vs el baseline.
    En esos campos la sensibilidad al precio esta NO IDENTIFICADA en banda:
    el "escalon" de M1 puede ser diferencia de decks de pronostico, no precio.

    Retorna {campo: {"FLAG", "ETA2", "SOLAPA_USD", "SALTO_REL"}} para los campos
    con ≥4 puntos reales y ≥2 años de vigencia; el resto no aparece (sin flag)."""
    real = df[(~df["ES_BASELINE"]) & (~df["ES_SINTETICO"])
              & df["DELTA_SENS_MBPE"].notna() & df["PRECIO_NETO_USD_BBL"].notna()].copy()
    real["_ANIO_VIG"] = real["VIGENCIA"].astype(str).str.slice(0, 4)

    out = {}
    for campo, g in real.groupby("CAMPO"):
        if len(g) < 4 or g["_ANIO_VIG"].nunique() < 2:
            continue
        # η²: fraccion de la varianza del delta explicada por el año de vigencia
        gm = g["DELTA_SENS_MBPE"].mean()
        ss_tot = ((g["DELTA_SENS_MBPE"] - gm) ** 2).sum()
        if ss_tot <= 0:
            continue
        ss_between = sum(len(sub) * (sub["DELTA_SENS_MBPE"].mean() - gm) ** 2
                         for _, sub in g.groupby("_ANIO_VIG"))
        eta2 = ss_between / ss_tot
        # Solape de bandas de precio entre años (>0 = solapan; <0 = separadas)
        rangos = {a: (sub["PRECIO_NETO_USD_BBL"].min(), sub["PRECIO_NETO_USD_BBL"].max())
                  for a, sub in g.groupby("_ANIO_VIG")}
        solapa = min(r[1] for r in rangos.values()) - max(r[0] for r in rangos.values())
        # Salto de medias inter-año relativo al baseline del campo
        medias = [sub["DELTA_SENS_MBPE"].mean() for _, sub in g.groupby("_ANIO_VIG")]
        salto = max(medias) - min(medias)
        base = baselines.get(campo, np.nan)
        salto_rel = salto / base if pd.notna(base) and base > 0 else np.nan
        flag = bool(eta2 >= CONFOUND_ETA2_MIN
                    and solapa < CONFOUND_SOLAPA_MAX_USD
                    and pd.notna(salto_rel) and salto_rel >= CONFOUND_SALTO_REL_MIN)
        out[campo] = {"FLAG": flag, "ETA2": round(float(eta2), 3),
                      "SOLAPA_USD": round(float(solapa), 2),
                      "SALTO_REL": round(float(salto_rel), 3) if pd.notna(salto_rel) else None,
                      "N_REALES": len(g)}
    return out


def cargar_anclas(df: pd.DataFrame) -> dict:
    """Anclas (BK_ANCLA_FIN superior, BK_ANCLA_PDP inferior) por campo, vigencia reciente."""
    sub = df.dropna(subset=["BK_ANCLA_FIN_USD_BBL", "BK_ANCLA_PDP_USD_BBL"], how="all")
    out = {}
    for campo, g in sub.groupby("CAMPO"):
        fila = g.sort_values("VIGENCIA_BREAKEVEN", ascending=False).iloc[0]
        bk_fin = float(fila["BK_ANCLA_FIN_USD_BBL"]) if pd.notna(fila["BK_ANCLA_FIN_USD_BBL"]) else np.nan
        bk_pdp = float(fila["BK_ANCLA_PDP_USD_BBL"]) if pd.notna(fila["BK_ANCLA_PDP_USD_BBL"]) else np.nan
        if np.isnan(bk_fin):
            bk_fin = bk_pdp
        if np.isnan(bk_pdp):
            bk_pdp = bk_fin
        out[campo] = (bk_fin, bk_pdp)
    return out


def cargar_ponderados(df: pd.DataFrame) -> dict:
    """Breakevens PONDERADOS 1P globales (D6, referencia de reporte) por campo."""
    sub = df.dropna(subset=["BREAKEVEN_USD_BBL",
                            "PRECIO_EQUILIBRIO_USD_BBL"], how="all")
    out = {}
    for campo, g in sub.groupby("CAMPO"):
        fila = g.sort_values("VIGENCIA_BREAKEVEN", ascending=False).iloc[0]
        fin = float(fila["BREAKEVEN_USD_BBL"]) \
            if pd.notna(fila["BREAKEVEN_USD_BBL"]) else np.nan
        ope = float(fila["PRECIO_EQUILIBRIO_USD_BBL"]) \
            if pd.notna(fila["PRECIO_EQUILIBRIO_USD_BBL"]) else np.nan
        out[campo] = (fin, ope)
    return out


def cargar_baselines(df: pd.DataFrame) -> dict:
    """Ultimo VOLUMEN_1P_OFICIAL_MBPE certificado por campo."""
    df_b = df[(df["ESCENARIO"] == "BASE") & df["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
    return (df_b.sort_values("AÑO")
            .groupby("CAMPO")["VOLUMEN_1P_OFICIAL_MBPE"].last().to_dict())


def cargar_anio_baseline(df: pd.DataFrame) -> dict:
    """Año del cierre oficial (carry-forward A-1) usado como VOLUMEN_1P_BASELINE_MBPE.

    Para trazabilidad: el Baseline per-campo NO es necesariamente el cierre 2025
    (campos sin reporte 2025 cargan el ultimo cierre conocido, ej. 2021) — distinto
    del Cierre 2025 (cargar_cierre_2025), que es el numero de comparacion 2026."""
    df_b = df[(df["ESCENARIO"] == "BASE") & df["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
    return (df_b.sort_values("AÑO")
            .groupby("CAMPO")["AÑO"].last().to_dict())


def cargar_cierre_2025(anio_cierre: int) -> dict:
    """1P certificado ECP S.A. (sin filiales) del cierre `anio_cierre`, por campo
    UNIFICADO — el numero de comparacion 2026 (~1685 MBPE para 2025).

    Reusa la homologacion de reconciliacion_baseline.py (misma agregacion v3: suma
    de componentes fisicos que coexisten bajo el mismo UNIFICADO) en vez de
    hardcodear el total: solo suma campos que SI certificaron ese cierre — un campo
    sin reporte ese año no debe aportar al numero contra el que se compara la
    predicción (directriz usuario 2026-07-16)."""
    from homologacion import Homologador
    from reconciliacion_baseline import cargar_hist_homologado, clasificar_scope
    hist = cargar_hist_homologado(anio_cierre)
    hom = Homologador()
    hist["SCOPE"] = hist["UNIFICADO"].apply(lambda u: clasificar_scope(u, hom))
    ok = hist[(hist["HOMOLOG_FLAG"] == "OK") & (hist["SCOPE"] == "PORTAFOLIO")]
    return ok.groupby("UNIFICADO")["1P"].sum().to_dict()


def banda_historica_brent(df: pd.DataFrame) -> dict:
    """Rango de Brent observado en CONSOLIDADO real por campo (para ES_EXTRAPOLADO)."""
    df_real = df[(~df["ES_SINTETICO"]) & (~df["ES_BASELINE"]) & df["BRENT_FLAT_USD_BBL"].notna()]
    out = {}
    for campo, sub in df_real.groupby("CAMPO"):
        out[campo] = (float(sub["BRENT_FLAT_USD_BBL"].min()),
                      float(sub["BRENT_FLAT_USD_BBL"].max()))
    return out


def clasificar_confianza(n_real, mae_rel, divergencia, baseline,
                         mae_abs=999.0, outlier_lloo=False,
                         skill=np.nan, mae_naive=np.nan,
                         m2_fragil=False, sens_no_ident=False,
                         alerta_bk="", sesgo_recup=np.nan,
                         metrica_base="LOO", skill_alt=np.nan) -> tuple[str, str]:
    """Clasifica la confianza usando criterios funcionales del piloto (MAESTRO §7.4).

    Métrica de gate (2026-07-10, contrato NORTE): cuando el campo tiene ≥2 vigencias de
    Consolidado se usan las métricas LOYO (leave-one-YEAR-out) — la pregunta honesta del
    negocio "¿puedes predecir el año que no viste?" = predecir el siguiente quarter. El
    LOO clásico deja 1 punto fuera y lo predicen los otros del MISMO año (delta plano
    intra-vigencia) → subestima el error de generalización 4-9x. `metrica_base` documenta
    cuál se usó; `skill_alt` es el skill de la otra métrica (para el motivo). Ver §3ter.

    Gate de skill (2026-06-11): ALTA exige SKILL > 0 o campo "plano"
    (MAE_NAIVE pequeño — la media ingenua es imbatible por construccion alli).
    SKILL NaN (sin reales suficientes o naive≈0) no penaliza.

    Revision hallazgos 2026-07-01: la exencion de campo plano deja pasar a ALTA
    campos grandes SIN senal de precio (ej. CAÑO SUR ESTE: MAE_NAIVE=1.18<2.0,
    skill=-0.44, baseline=140 MBPE). El nivel no baja (el error SIGUE siendo
    bajo) pero el motivo se marca "insensible-al-precio" para que el lector no
    confunda "preciso" con "sensible al Brent" en un campo material.
    m2_fragil (M2 sin historia propia / n<5 / R2_LOO<0) SI degrada el techo a
    MEDIA en campos materiales: la cadena completa depende de una recta fragil.
    sens_no_ident (confound de vigencia §3ter, 2026-07-02) tambien degrada el
    techo a MEDIA en campos materiales: el escalon de M1 puede ser diferencia
    entre decks de años distintos, no efecto precio — la pendiente que lee CAPEX
    no esta identificada aunque el error LOO luzca bajo."""
    if mae_abs < CONFIANZA_MAE_ABS_ALTA:
        mae_rel_eff = min(mae_rel, CONFIANZA_MAE_REL_ALTA * 0.99)
    elif mae_abs < CONFIANZA_MAE_ABS_MEDIA:
        mae_rel_eff = min(mae_rel, CONFIANZA_MAE_REL_MEDIA * 0.99)
    else:
        mae_rel_eff = mae_rel

    # Outlier inmaterial: el ratio RMSE/MAE dispara pero el error relativo es
    # despreciable para CAPEX → no degrada, solo queda en el motivo (auditable).
    outlier_material = outlier_lloo and mae_rel >= CONFIANZA_MAE_REL_OUTLIER

    # Divergencia NaN = no computable (sin pivote en banda): no bloquea ni degrada,
    # solo queda visible en el motivo (auditoria 2026-07-07 H3; antes 999 -> BAJA).
    div_na = pd.isna(divergencia)
    div_ok_alta  = div_na or divergencia < CONFIANZA_DIV_ALTA
    div_ok_media = div_na or divergencia < CONFIANZA_DIV_MEDIA

    partes = [f"metrica={metrica_base}",
              f"N={n_real}", f"MAE_rel={mae_rel:.2f}", f"MAE_abs={mae_abs:.2f}MBPE",
              "div=NA" if div_na else f"div={divergencia:.2f}",
              f"base={baseline:.1f}MBPE"]
    if div_na:
        partes.insert(0, "divergencia-no-computable")
    if alerta_bk:
        # BK_SUPERA_PRECIO_REF / SIN_REANCLAJE en formato motivo (minusculas-guion)
        partes.insert(0, alerta_bk.lower().replace("_", "-"))
    if pd.notna(skill):
        partes.append(f"skill={skill:.2f}")
    if pd.notna(skill_alt):
        partes.append(f"skill_{'loo' if metrica_base=='LOYO' else 'loyo'}={skill_alt:.2f}")
    if outlier_lloo:
        partes.append("OUTLIER_LOO" if outlier_material else "OUTLIER_LOO_INMATERIAL")
    # Separador " | " (no ";"): Excel es-CO interpreta ";" como delimitador de
    # columnas y parte la celda MOTIVO_CONFIANZA al abrir el CSV (ver MAESTRO §10).
    motivo = " | ".join(partes)

    if n_real == 0:
        # INSENSIBLE_PRECIO (antes SOLO_GAS/SOLO_SINTETICO, renombre 2026-07-15): campo
        # sin deck real — típicamente gas/GLP insensible al precio. La predicción es
        # plana = último cierre (sin curva sintética en el export; directriz §4).
        return "INSENSIBLE_PRECIO", f"sin deck real (gas/insensible) | {motivo}"
    if baseline < CONFIANZA_BASELINE_MIN:
        return "BAJA", f"micro-campo | {motivo}"
    if outlier_material:
        if (mae_rel_eff < CONFIANZA_MAE_REL_MEDIA and div_ok_media):
            return "MEDIA", motivo
        return "BAJA", motivo

    # Gate de skill: falla solo si el modelo NO supera al ingenuo en un campo
    # con variacion material de deltas (naive grande). Campos planos exentos.
    sin_skill = (pd.notna(skill) and skill <= CONFIANZA_SKILL_MIN
                 and pd.notna(mae_naive) and mae_naive >= CONFIANZA_MAE_ABS_ALTA)

    # Campo material que pasa el gate SOLO por la exencion de "plano" (skill<=0
    # pero mae_naive chico): el error bajo no implica que el Brent explique el
    # volumen. Se marca en el motivo, no se baja el nivel (informe 2026-07-01 WS2.1).
    insensible_material = (pd.notna(skill) and skill <= CONFIANZA_SKILL_MIN
                            and not sin_skill and baseline >= CONFIANZA_BASELINE_MATERIAL)
    if insensible_material:
        motivo = f"insensible-al-precio | {motivo}"

    # Confound de vigencia (informe 2026-07-02 §3ter) en campo material: el
    # escalon de precio de M1 puede ser diferencia entre decks de pronostico de
    # años distintos (sensibilidad NO identificada en banda) -> techo MEDIA.
    # El error LOO luce bajo (los folds caen dentro de cada nube anual) pero la
    # PENDIENTE de la curva no es confiable — que es justo lo que lee CAPEX.
    if sens_no_ident and baseline >= CONFOUND_BASELINE_MATERIAL:
        motivo = f"sensibilidad-no-identificada | {motivo}"
        if (mae_rel_eff < CONFIANZA_MAE_REL_MEDIA and div_ok_media):
            return "MEDIA", motivo
        return "BAJA", motivo

    # M2 fragil (sin historia propia / n<5 puntos / R2_LOO<0) en campo material:
    # la cadena completa depende de una recta poco confiable -> techo MEDIA
    # aunque M1 luzca preciso (informe 2026-07-01 WS2.3).
    if m2_fragil and baseline >= CONFIANZA_BASELINE_MATERIAL:
        motivo = f"M2-fragil | {motivo}"
        if (mae_rel_eff < CONFIANZA_MAE_REL_MEDIA and div_ok_media):
            return "MEDIA", motivo
        return "BAJA", motivo

    # Sesgo de recuperacion de vigencia (artifact 8a8f2cbc, 2026-07-10): un
    # |d_ref|/baseline grande delata que el ancla p_ref cae en el valle de la ultima
    # vigencia y la curva "recupera" el nivel de un deck anterior al subir el Brent
    # (declino leido como elasticidad). NO se condiciona a materialidad: el problema
    # es mas visible en maduros pequeños (Casabe/Caño Limon) que hoy salen ALTA con
    # +40%. Techo MEDIA (BAJA si ademas el error LOO es alto).
    if pd.notna(sesgo_recup) and sesgo_recup > CONFIANZA_SESGO_RECUP_MAX:
        motivo = f"sesgo-recuperacion={sesgo_recup:.0%} | {motivo}"
        if (mae_rel_eff < CONFIANZA_MAE_REL_MEDIA and div_ok_media):
            return "MEDIA", motivo
        return "BAJA", motivo

    if (n_real >= CONFIANZA_N_REAL_MIN and mae_rel_eff < CONFIANZA_MAE_REL_ALTA
            and div_ok_alta and not sin_skill):
        return "ALTA", motivo
    if (mae_rel_eff < CONFIANZA_MAE_REL_MEDIA and div_ok_media):
        if sin_skill:
            return "MEDIA", f"sin skill vs ingenuo | {motivo}"
        return "MEDIA", motivo
    return "BAJA", motivo


def calcular_divergencia_motores(df_out: pd.DataFrame) -> pd.Series:
    """Divergencia media |Suave - Isotonica| / Isotonica por campo en banda Brent $40-$80.
    Retorna Serie indexada por CAMPO."""
    banda = df_out[(df_out["BRENT_USD_BBL"] >= 40) & (df_out["BRENT_USD_BBL"] <= 80)]
    piv = (banda.pivot_table(index=["CAMPO", "BRENT_USD_BBL"], columns="MOTOR",
                             values="VOLUMEN_1P_PREDICHO_MBPE")
           .reset_index().dropna())
    if piv.empty or "Isotonica" not in piv.columns or "Suave" not in piv.columns:
        return pd.Series(dtype=float)
    piv["div"] = (piv["Suave"] - piv["Isotonica"]).abs() / piv["Isotonica"].replace(0, np.nan)
    return piv.groupby("CAMPO")["div"].mean()


def _extraer_en_brent_ref(df: pd.DataFrame, brent_ref: float) -> pd.DataFrame:
    """Para cada CAMPO x MOTOR, fila con BRENT mas cercano a brent_ref (escenario BASE)."""
    sub = df
    if "ESCENARIO_DESCUENTO" in sub.columns:
        sub = sub[sub["ESCENARIO_DESCUENTO"] == "BASE"]
    sub = sub.copy()
    sub["_dist"] = (sub["BRENT_USD_BBL"] - brent_ref).abs()
    idx = sub.groupby(["CAMPO", "MOTOR"])["_dist"].idxmin()
    out = sub.loc[idx].reset_index(drop=True)
    for c in ["NIVEL_CONFIANZA", "Q_OBJETIVO"]:
        if c not in out.columns:
            out[c] = np.nan
    return out[["CAMPO", "MOTOR", "VOLUMEN_1P_PREDICHO_MBPE", "Q_OBJETIVO", "NIVEL_CONFIANZA"]]


def generar_comparacion_vs_anterior(df_out, q_objetivo, brent_ref, ruta_actual):
    """Compara la prediccion actual (BASE, Brent~ref) vs el snapshot previo mas reciente."""
    cols_out = ["CAMPO", "MOTOR", "Q_OBJETIVO_ANTERIOR", "Q_OBJETIVO_NUEVO",
                "VOL_ANTERIOR_MBPE", "VOL_NUEVO_MBPE", "DIF_ABS_MBPE", "DIF_PCT",
                "NIVEL_CONFIANZA_ANTERIOR", "NIVEL_CONFIANZA_NUEVO"]
    ruta_comp = RESULTADOS / "comparacion_vs_anterior.csv"
    previos = sorted((p for p in HISTORICO_DIR.glob("prediccion_*.csv")
                      if p.resolve() != ruta_actual.resolve()),
                     key=lambda p: p.stat().st_mtime)
    if not previos:
        pd.DataFrame(columns=cols_out).to_csv(ruta_comp, index=False, encoding="utf-8-sig")
        print(f"\n  [INFO] Sin snapshot previo: {ruta_comp} (solo encabezados)")
        return None

    ruta_prev = previos[-1]
    df_prev = pd.read_csv(ruta_prev)
    prev_ref  = _extraer_en_brent_ref(df_prev, brent_ref)
    nuevo_ref = _extraer_en_brent_ref(df_out, brent_ref)
    merged = nuevo_ref.merge(prev_ref, on=["CAMPO", "MOTOR"], how="outer",
                             suffixes=("_NUEVO", "_ANTERIOR"))
    merged = merged.rename(columns={
        "VOLUMEN_1P_PREDICHO_MBPE_NUEVO":    "VOL_NUEVO_MBPE",
        "VOLUMEN_1P_PREDICHO_MBPE_ANTERIOR": "VOL_ANTERIOR_MBPE"})
    merged["DIF_ABS_MBPE"] = merged["VOL_NUEVO_MBPE"] - merged["VOL_ANTERIOR_MBPE"]
    merged["DIF_PCT"] = (merged["DIF_ABS_MBPE"] /
                         merged["VOL_ANTERIOR_MBPE"].replace(0, np.nan) * 100)
    merged = merged[cols_out].sort_values("DIF_ABS_MBPE",
                                          key=lambda s: s.abs(), ascending=False,
                                          na_position="last")
    merged.to_csv(ruta_comp, index=False, encoding="utf-8-sig")
    print(f"\n  Comparacion vs corrida anterior ({ruta_prev.name}): {ruta_comp}")
    print(f"    @Brent~${brent_ref}: {len(merged)} filas comparadas")
    return merged


def actualizar_changelog(q_objetivo, fecha, n_campos, comp) -> None:
    """Agrega entrada fechada a docs/CHANGELOG_PREDICCIONES.md."""
    ruta = BASE_DIR / "docs" / "CHANGELOG_PREDICCIONES.md"
    if not ruta.exists():
        ruta.write_text("# Changelog de Predicciones\n\n"
                        "Historial de corridas del pipeline 01-04.\n", encoding="utf-8")
    bloque = [f"\n## {fecha} — {q_objetivo} ({n_campos} campos)\n"]
    if comp is None or comp.empty:
        bloque.append("\nSin snapshot previo para comparar.\n")
    else:
        top = comp.head(5)
        bloque.append("\nTop 5 movimientos vs corrida anterior (|DIF_ABS_MBPE|):\n\n")
        bloque.append("| CAMPO | MOTOR | VOL_ANTERIOR | VOL_NUEVO | DIF_ABS | DIF_% |\n")
        bloque.append("|---|---|---:|---:|---:|---:|\n")
        for _, r in top.iterrows():
            bloque.append(f"| {r['CAMPO']} | {r['MOTOR']} | {r['VOL_ANTERIOR_MBPE']:.1f} | "
                          f"{r['VOL_NUEVO_MBPE']:.1f} | {r['DIF_ABS_MBPE']:.1f} | "
                          f"{r['DIF_PCT']:.1f}% |\n")
    with open(ruta, "a", encoding="utf-8") as f:
        f.writelines(bloque)
    print(f"  Changelog actualizado: {ruta}")


def emitir_cobertura_plana(df, campos_exportados, brent_range,
                           q_objetivo, vigencia_base, fecha_prediccion,
                           brent_ref, cierre_2025=None, anio_baseline_map=None) -> list:
    """
    Filas planas para el mapeo del portafolio COMPLETO (directriz §4 + requisito de
    cobertura 2026-07-10): todo campo con cierre oficial que NO tiene modelo (ni deck ni
    breakeven) entra como línea plana = último cierre, para que al seleccionar "todos los
    campos / todas las confiabilidades" el tablero muestre el portafolio entero.

    Excluye FILIAL (política: filiales nunca entran al análisis) y los campos ya
    exportados. Nivel de confianza = SIN_MODELO (asignado en el flujo principal).
    """
    from homologacion import Homologador
    h = Homologador()
    cierre_2025 = cierre_2025 or {}
    anio_baseline_map = anio_baseline_map or {}

    # Último cierre oficial por campo (serie BASE)
    base = df[(df["ESCENARIO"] == "BASE") & df["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
    cierre = (base.sort_values("AÑO").groupby("CAMPO")["VOLUMEN_1P_OFICIAL_MBPE"]
              .last().to_dict())

    filas_cov = []
    n_campos = 0
    for campo, vol_cierre in cierre.items():
        if campo in campos_exportados:
            continue
        gerencia = str(h.atributos(campo).get("NEW GERENCIA", "")).strip().upper()
        if gerencia == "FILIAL":
            continue
        # Directriz usuario 2026-07-16: si el campo SÍ certificó cierre 2025 (incluso
        # si fue 0), ese es el valor vigente — no el último cierre positivo de años
        # anteriores (2017-2021). Sin esto, campos ya agotados seguían mostrando un
        # "1P" fantasma heredado de su última certificación con volumen.
        cierre_2025_campo = cierre_2025.get(campo)
        if cierre_2025_campo is not None and pd.notna(cierre_2025_campo):
            vol_cierre = cierre_2025_campo
            anio_base_campo = 2025
        else:
            anio_base_campo = anio_baseline_map.get(campo)
        if not (pd.notna(vol_cierre) and float(vol_cierre) > 0):
            continue
        n_campos += 1
        for label in ("Isotonica", "Suave"):
            for brent in brent_range:
                filas_cov.append({
                    "CAMPO":                        campo,
                    "MOTOR":                        label,
                    "BRENT_USD_BBL":                float(brent),
                    "PRECIO_NETO_EFECTIVO_USD_BBL": None,   # sin M2
                    "DELTA_PRED_MBPE":              0.0,
                    "VOLUMEN_1P_BASELINE_MBPE":     round(float(vol_cierre), 2),
                    "AÑO_BASELINE":                 int(anio_base_campo)
                                                    if pd.notna(anio_base_campo) else None,
                    "CIERRE_2025_MBPE":             round(float(cierre_2025_campo), 2)
                                                    if cierre_2025_campo is not None
                                                    and pd.notna(cierre_2025_campo) else None,
                    "VOLUMEN_1P_PREDICHO_MBPE":     round(float(vol_cierre), 2),
                    "DELTA_VS_BASE_MBPE":           0.0,
                    "DELTA_VS_BASE_PCT":            0.0,
                    "BRENT_REF_USD_BBL":            round(brent_ref, 2) if pd.notna(brent_ref) else None,
                    "ES_VIABLE":                    True,
                    "ES_FULL_RESERVAS":             True,
                    "ES_EXTRAPOLADO":               False,
                    "ES_CLIPPED":                   False,
                    "TIPO_MODELO":                  "SIN_MODELO",
                    "TIPO_DATO":                    "SIN_MODELO",
                    "Q_OBJETIVO":                   q_objetivo,
                    "VIGENCIA_BASE":                vigencia_base,
                    "FECHA_PREDICCION":             fecha_prediccion,
                })
    print(f"\n  [Cobertura] {n_campos} campos sin modelo (cierre 2025, no-filial) "
          f"agregados como línea plana SIN_MODELO")
    return filas_cov


if __name__ == "__main__":
    print("=== 04_pbi_export.py — Meshgrid Brent->Neto->Delta (Modelo 2 + Modelo 1) ===\n")

    ruta = STAGING / "tablon_unico.parquet"
    if not ruta.exists():
        raise FileNotFoundError("Ejecutar scripts 01-03b primero.")
    df = pd.read_parquet(ruta)

    correlacion = cargar_correlacion()
    anclas      = cargar_anclas(df)
    ponderados  = cargar_ponderados(df)
    baselines   = cargar_baselines(df)
    bandas      = banda_historica_brent(df)
    confound    = calcular_confound_vigencia(df, baselines)

    # Re-anclaje (2026-06-12): p_ref/delta_ref por campo x motor desde metricas.csv (03)
    ruta_met = STAGING / "metricas.csv"
    if not ruta_met.exists():
        raise FileNotFoundError("metricas.csv no existe: correr 03_modelo.py primero.")
    _met_full = pd.read_csv(ruta_met)
    anclaje = _met_full.set_index("CAMPO")[
        ["BRENT_REF_USD_BBL", "P_REF_USD_BBL", "DELTA_REF_ISO", "DELTA_REF_SUAVE"]
    ].to_dict("index")
    # Deck plano (directriz §4): campos insensibles al precio -> curva plana (sin rampa
    # isotonica interpolada en la franja sin datos). Detectado en 03_modelo.
    deck_plano_map = (_met_full.set_index("CAMPO")["DECK_PLANO"].fillna(False).astype(bool).to_dict()
                      if "DECK_PLANO" in _met_full.columns else {})
    n_real_map = (_met_full.set_index("CAMPO")["N_REAL_DELTA"].fillna(0).astype(int).to_dict()
                  if "N_REAL_DELTA" in _met_full.columns else {})
    # s12: nombres reales del par primario/validacion (2.o LOYO dinamico)
    metodo_prim_map = (_met_full.set_index("CAMPO")["METODO_PRIMARIO"].to_dict()
                       if "METODO_PRIMARIO" in _met_full.columns else {})
    metodo_valid_map = (_met_full.set_index("CAMPO")["METODO_VALIDACION"].to_dict()
                        if "METODO_VALIDACION" in _met_full.columns else {})
    # Bandas de incertidumbre LOYO (track Calidad, s9): presentes solo si 03 corrio con
    # PRED_BANDAS_LOYO. Guard por PRESENCIA de columnas (mismo patron que BK_P*): en
    # Produccion no existen y el export queda identico.
    resid_p10 = (_met_full.set_index("CAMPO")["LOYO_RESID_P10"].to_dict()
                 if "LOYO_RESID_P10" in _met_full.columns else {})
    resid_p90 = (_met_full.set_index("CAMPO")["LOYO_RESID_P90"].to_dict()
                 if "LOYO_RESID_P90" in _met_full.columns else {})
    BRENT_REF = float(_met_full["BRENT_REF_USD_BBL"].dropna().iloc[0]) \
        if _met_full["BRENT_REF_USD_BBL"].notna().any() else np.nan

    vigencia_base, q_objetivo = derivar_vigencias(df)
    fecha_prediccion = str(date.today())
    print(f"\n{'='*55}")
    print(f"  PREDICCION para {q_objetivo}  (datos base: {vigencia_base}, "
          f"generada: {fecha_prediccion})")
    print(f"{'='*55}\n")

    # Cierre de comparación 2026 (directriz usuario 2026-07-16): 1P certificado
    # ECP S.A. (sin filiales) del año A-1 respecto a vigencia_base — DISTINTO del
    # Baseline per-campo (carry-forward, puede venir de años anteriores a A-1 en
    # campos sin reporte reciente). anio_baseline_map documenta ese carry-forward.
    anio_baseline_map = cargar_anio_baseline(df)
    anio_cierre_2025 = (int(vigencia_base.split("_")[0]) - 1
                        if vigencia_base != "DESCONOCIDA" else 2025)
    cierre_2025 = cargar_cierre_2025(anio_cierre_2025)
    print(f"  Cierre {anio_cierre_2025} certificado ECP S.A. (sin filiales, "
          f"base de comparación 2026): {sum(cierre_2025.values()):,.1f} MBPE "
          f"({len(cierre_2025)} campos con cierre {anio_cierre_2025})\n")

    # Meshgrid dinamico: banda Consolidado real ± MARGEN_BRENT_USD
    df_cons_real = df[(~df["ES_SINTETICO"]) & (~df["ES_BASELINE"]) & df["BRENT_FLAT_USD_BBL"].notna()]
    if not df_cons_real.empty:
        brent_obs_min = float(df_cons_real["BRENT_FLAT_USD_BBL"].min())
        brent_obs_max = float(df_cons_real["BRENT_FLAT_USD_BBL"].max())
    else:
        brent_obs_min, brent_obs_max = 58.0, 92.0
    brent_min_din = int(np.floor(brent_obs_min)) - MARGEN_BRENT_USD
    brent_max_din = int(np.ceil(brent_obs_max)) + MARGEN_BRENT_USD
    brent_range = np.arange(brent_min_din, brent_max_din + BRENT_PASO, BRENT_PASO,
                            dtype=float)
    # Incluir el punto EXACTO de re-anclaje: la fila Brent=BRENT_REF debe existir
    # en la matriz para que Vol=baseline sea verificable sin interpolar
    if pd.notna(BRENT_REF):
        brent_range = np.unique(np.append(brent_range, round(BRENT_REF, 2)))
    campos = sorted(df["CAMPO"].unique())

    print(f"Banda Consolidado observada: [${brent_obs_min:.1f}, ${brent_obs_max:.1f}]")
    print(f"Grilla Brent (+-{MARGEN_BRENT_USD}): [${brent_min_din}, ${brent_max_din}] "
          f"paso ${BRENT_PASO} = {len(brent_range)} pts")
    print(f"Campos: {len(campos)} | Motores: Isotonica (primario), Suave (validacion)")
    print(f"Re-anclaje: BRENT_REF={BRENT_REF:.2f} -> Vol(BRENT_REF) = baseline exacto\n")

    pneto_grid = np.arange(PNETO_GRID_MIN, PNETO_GRID_MAX + PNETO_GRID_PASO,
                           PNETO_GRID_PASO)
    filas    = []   # cadena completa Brent -> Aceite -> Volumen
    filas_m1 = []   # Modelo 1 puro (grilla en Precio Aceite)
    filas_m2 = []   # Modelo 2 puro (grilla en Brent)
    alertas_bk = {} # CAMPO -> ALERTA_BK (H1/H3) para el motivo de confianza
    for campo in campos:
        modelos = {}
        for label, suf in MOTORES:
            rmodel = MODELOS_DIR / f"{campo.replace(' ', '_')}_{suf}.joblib"
            if rmodel.exists():
                modelos[label] = joblib.load(rmodel)
        if not modelos:
            print(f"  [WARN] {campo}: sin modelos, omitiendo")
            continue
        coef = correlacion.get(campo)
        if coef is None:
            print(f"  [WARN] {campo}: sin correlacion Brent->Aceite, omitiendo")
            continue

        bk_fin, bk_pdp = anclas.get(campo, (np.nan, np.nan))
        bk_ref_fin, bk_ref_ope = ponderados.get(campo, (np.nan, np.nan))
        baseline = baselines.get(campo, np.nan)
        anio_baseline_campo = anio_baseline_map.get(campo)
        cierre_2025_campo = cierre_2025.get(campo)
        bk_min_hist, bk_max_hist = bandas.get(campo, (40.0, 80.0))
        anc = anclaje.get(campo, {})
        p_ref = anc.get("P_REF_USD_BBL", np.nan)
        delta_refs = {"Isotonica": anc.get("DELTA_REF_ISO", 0.0) or 0.0,
                      "Suave":     anc.get("DELTA_REF_SUAVE", 0.0) or 0.0}
        m2_metodo = coef.get("METODO", "")
        m2_fallback = bool(coef.get("ES_FALLBACK", False))
        m2_fragil = es_m2_fragil(coef)
        conf_campo = confound.get(campo, {})
        sens_no_ident = bool(conf_campo.get("FLAG", False))
        _bk_dura = bk_pdp if pd.notna(bk_pdp) else None
        _p_ref_f = float(p_ref) if pd.notna(p_ref) else None

        # Auditoria 2026-07-07 (H1/H3): coherencia BK vs certificacion y re-anclaje.
        # BK_SUPERA_PRECIO_REF: el BK de abandono queda >= p_ref con baseline > 0 —
        #   la certificacion vigente falsifica ese BK; volumen_anclado capa el piso
        #   (piso_efectivo) para que Vol(p_ref)=baseline en vez de un 0 espurio.
        # SIN_REANCLAJE: campo sin p_ref en metricas — la curva sale SIN shift.
        alerta_bk = ""
        if _p_ref_f is None:
            print(f"  [WARN] {campo}: sin p_ref — re-anclaje DESACTIVADO (delta_ref=0)")
            alerta_bk = "SIN_REANCLAJE"
        elif (_bk_dura is not None and pd.notna(baseline) and baseline > 0
              and _bk_dura >= _p_ref_f):
            alerta_bk = "BK_SUPERA_PRECIO_REF"
        _bk_eff = piso_efectivo(_bk_dura, _p_ref_f,
                                float(baseline) if pd.notna(baseline) else np.nan)
        alertas_bk[campo] = alerta_bk

        # Deck plano (directriz §4): curva = baseline sobre el piso efectivo, 0 debajo.
        # Reemplaza la rampa isotonica interpolada en la franja sin datos por el
        # comportamiento honesto "plano hasta el abandono". Re-anclaje trivialmente
        # satisfecho (Vol(p_ref)=baseline). TIPO_MODELO documenta la sustitucion.
        # INSENSIBLE_PRECIO (n_real=0): campo sin deck real → también curva plana = último cierre.
        n_real0 = (n_real_map.get(campo, 0) == 0)
        es_plano = (bool(deck_plano_map.get(campo, False)) or n_real0) and pd.notna(baseline)
        tipo_modelo = ("INSENSIBLE_PRECIO" if n_real0 else
                       "PLANO_DECK" if bool(deck_plano_map.get(campo, False)) else "ISOTONICA")
        _piso_plano = _bk_eff if _bk_eff is not None else -np.inf

        def _vol_curva(modelo, neto, d_ref):
            """Volumen 1P reconstruido: plano (deck insensible) o anclado (isotonica)."""
            if not pd.notna(baseline):
                return np.full(len(neto), np.nan)
            if es_plano:
                return np.where(np.asarray(neto, dtype=float) >= _piso_plano,
                                float(baseline), 0.0)
            return volumen_anclado(modelo, neto, baseline, d_ref, _bk_dura, p_ref=_p_ref_f)

        # ── Matriz M2 pura: Brent -> Precio Aceite ────────────────────────────
        aceite_grid = m2.neto_desde_brent(coef, brent_range)
        for i, brent in enumerate(brent_range):
            filas_m2.append({
                "CAMPO":                  campo,
                "BRENT_USD_BBL":          float(brent),
                "PRECIO_ACEITE_USD_BBL":  round(float(aceite_grid[i]), 2),
                "ALPHA":                  coef.get("ALPHA"),
                "BETA":                   coef.get("BETA"),
                "M2_METODO":              m2_metodo,
                "M2_ES_FALLBACK":         m2_fallback,
                "N_PUNTOS":               coef.get("N_PUNTOS"),
                "R2":                     coef.get("R2"),
                "R2_LOO":                 coef.get("R2_LOO"),
                "MAE_LOO":                coef.get("MAE_LOO"),
                "ALERTA":                 coef.get("ALERTA", ""),
                "M2_FRAGIL":              m2_fragil,
            })

        # ── Matriz M1 pura: Precio Aceite -> Delta anclado / Volumen ─────────
        _q10m1, _q90m1 = resid_p10.get(campo), resid_p90.get(campo)
        for label, modelo in modelos.items():
            d_ref = float(delta_refs[label])
            vol_anc = _vol_curva(modelo, pneto_grid, d_ref)
            # Delta plano = vol−baseline (0 sobre el piso, −baseline debajo); isotonica
            # usa el modelo crudo re-anclado.
            delta_anc = (vol_anc - float(baseline)) if es_plano \
                else (modelo.predict(pneto_grid) - d_ref)
            for i, pn in enumerate(pneto_grid):
                vp = float(vol_anc[i]) if pd.notna(vol_anc[i]) else np.nan
                # Banda LOYO en espacio Precio Neto (mismo criterio que la matriz de
                # prediccion en espacio Brent, lineas ~925-933): solo motor primario,
                # clamp P10<=vol<=P90 sin negativos, colapsa a 0 bajo el piso.
                if (label == "Isotonica" and _q10m1 is not None and pd.notna(_q10m1)
                        and pd.notna(vp) and vp > 0):
                    vol_p10_m1 = round(max(0.0, min(vp, vp + float(_q10m1))), 2)
                    vol_p90_m1 = round(max(vp, vp + float(_q90m1)), 2)
                else:
                    vol_p10_m1 = vol_p90_m1 = None
                filas_m1.append({
                    "CAMPO":                       campo,
                    "MOTOR":                       label,
                    "PRECIO_ACEITE_USD_BBL":       float(pn),
                    "DELTA_ANCLADO_MBPE":          round(float(delta_anc[i]), 2)
                                                   if pd.notna(delta_anc[i]) else None,
                    "VOLUMEN_1P_PREDICHO_MBPE":    round(vp, 2) if pd.notna(vp) else None,
                    "VOL_P10_MBPE":                vol_p10_m1,
                    "VOL_P90_MBPE":                vol_p90_m1,
                    "VOLUMEN_1P_BASELINE_MBPE":    round(float(baseline), 2)
                                                   if pd.notna(baseline) else None,
                    "P_REF_USD_BBL":               round(float(p_ref), 2)
                                                   if pd.notna(p_ref) else None,
                    "TIPO_MODELO":                 tipo_modelo,
                    "METODO_PRIMARIO":             metodo_prim_map.get(campo, "Isotonica"),
                    "METODO_VALIDACION":           metodo_valid_map.get(campo, "Suave"),
                    # METODO_REAL: nombre real del motor de esta fila (joblib iso/suave)
                    "METODO_REAL":                 (metodo_prim_map.get(campo, "Isotonica")
                                                   if label == "Isotonica"
                                                   else metodo_valid_map.get(campo, "Suave")),
                    "BREAKEVEN_USD_BBL":  round(bk_fin, 2) if pd.notna(bk_fin) else None,
                    "PRECIO_EQUILIBRIO_USD_BBL": round(bk_pdp, 2) if pd.notna(bk_pdp) else None,
                })

        # ── Cadena completa: Brent -> Aceite -> Volumen anclado ──────────────
        for label, modelo in modelos.items():
            d_ref = float(delta_refs[label])
            # Sin baseline no hay volumen reconstruible: NaN, no un delta disfrazado
            # de volumen (auditoria 2026-07-07 H3). Plano: curva plana (directriz §4).
            vol = _vol_curva(modelo, aceite_grid, d_ref)
            delta_anc = (vol - float(baseline)) if es_plano \
                else (modelo.predict(aceite_grid) - d_ref)

            for i, brent in enumerate(brent_range):
                pn   = float(aceite_grid[i])
                pn_r = round(pn, 2)
                # ES_VIABLE contra el piso EFECTIVO: debe coincidir con el hard-zero
                # que aplica volumen_anclado (H1) — si no, el tablero mostraria
                # ES_VIABLE=False con volumen > 0
                es_viable = (pn_r >= round(_bk_eff, 2)) if _bk_eff is not None else True
                es_full   = (pn_r >= round(bk_fin, 2)) if pd.notna(bk_fin) else True
                es_extrap = (float(brent) < bk_min_hist - MARGEN_EXTRAP_USD or
                             float(brent) > bk_max_hist + MARGEN_EXTRAP_USD)
                # ES_CLIPPED (artifact 8a8f2cbc, 2026-07-10): el punto queda por ENCIMA
                # del techo de la banda del deck -> la isotonica (out_of_bounds='clip')
                # esta saturada y entrega el "techo de recuperacion" de una vigencia
                # anterior. Mas estricto que ES_EXTRAPOLADO (sin margen, solo lado alto):
                # marca justo la zona donde el volumen predicho es un artefacto del clip.
                es_clipped = float(brent) > bk_max_hist

                vol_pred = float(vol[i])
                vol_base = float(baseline) if pd.notna(baseline) else np.nan
                delta_vs = round(vol_pred - vol_base, 2) if pd.notna(baseline) else np.nan
                pct_vs   = round((vol_pred - vol_base) / vol_base * 100, 2) \
                    if pd.notna(baseline) and vol_base > 0 else np.nan

                # Banda de incertidumbre LOYO (s9): curva + cuantiles de residuales.
                # Clamps: P10 <= vol <= P90, sin negativos; bajo el piso (vol=0, afirmacion
                # dura de abandono) la banda colapsa a 0 — la incertidumbre ahi es del BK,
                # no de la curva. Solo motor primario (Isotonica); NaN sin residuales.
                _q10, _q90 = resid_p10.get(campo), resid_p90.get(campo)
                if (label == "Isotonica" and _q10 is not None and pd.notna(_q10)
                        and vol_pred > 0):
                    vol_p10 = round(max(0.0, min(vol_pred, vol_pred + float(_q10))), 2)
                    vol_p90 = round(max(vol_pred, vol_pred + float(_q90)), 2)
                else:
                    vol_p10 = vol_p90 = None
                ancho_banda = round(vol_p90 - vol_p10, 2) \
                    if vol_p10 is not None and vol_p90 is not None else None
                _extra_banda = ({"VOL_P10_MBPE": vol_p10, "VOL_P90_MBPE": vol_p90,
                                 "ANCHO_BANDA_LOYO_MBPE": ancho_banda}
                                if resid_p10 else {})

                filas.append({
                    **_extra_banda,
                    "CAMPO":                          campo,
                    "MOTOR":                          label,
                    "BRENT_USD_BBL":                  float(brent),
                    "PRECIO_NETO_EFECTIVO_USD_BBL":   round(pn, 2),
                    "DELTA_PRED_MBPE":                round(float(delta_anc[i]), 2),
                    "VOLUMEN_1P_BASELINE_MBPE":        round(vol_base, 2) if pd.notna(vol_base) else None,
                    "AÑO_BASELINE":                    int(anio_baseline_campo)
                                                       if pd.notna(anio_baseline_campo) else None,
                    # Cierre 2025 ECP S.A. (sin filiales) — numero de comparacion 2026,
                    # DISTINTO del Baseline per-campo (carry-forward A-1, arriba): un
                    # campo sin reporte 2025 no debe sumar aqui (directriz usuario).
                    "CIERRE_2025_MBPE":                round(float(cierre_2025_campo), 2)
                                                       if cierre_2025_campo is not None
                                                       and pd.notna(cierre_2025_campo) else None,
                    "VOLUMEN_1P_PREDICHO_MBPE":        round(vol_pred, 2),
                    "DELTA_VS_BASE_MBPE":              delta_vs,
                    "DELTA_VS_BASE_PCT":               pct_vs,
                    # Re-anclaje: punto actual donde Vol=baseline por construccion
                    "BRENT_REF_USD_BBL":               round(BRENT_REF, 2) if pd.notna(BRENT_REF) else None,
                    "P_REF_USD_BBL":                   round(float(p_ref), 2) if pd.notna(p_ref) else None,
                    # Tag M2: campos sin relacion propia Aceite~Brent (k de portafolio)
                    "M2_METODO":                       m2_metodo,
                    "M2_ES_FALLBACK":                  m2_fallback,
                    "M2_N_PUNTOS":                     coef.get("N_PUNTOS"),
                    "M2_R2_LOO":                       coef.get("R2_LOO"),
                    "M2_FRAGIL":                       m2_fragil,
                    # Confound de vigencia (§3ter): la pendiente de precio de M1
                    # no es separable del cambio de año de Consolidado
                    "SENSIBILIDAD_NO_IDENTIFICADA":    sens_no_ident,
                    "CONFOUND_ETA2":                   conf_campo.get("ETA2"),
                    "BREAKEVEN_USD_BBL":    round(bk_fin, 2) if pd.notna(bk_fin) else None,
                    "PRECIO_EQUILIBRIO_USD_BBL":   round(bk_pdp, 2) if pd.notna(bk_pdp) else None,
                    "BREAKEVEN_REF_USD_BBL":       round(bk_ref_fin, 2) if pd.notna(bk_ref_fin) else None,
                    "PRECIO_EQUILIBRIO_REF_USD_BBL":       round(bk_ref_ope, 2) if pd.notna(bk_ref_ope) else None,
                    "ES_VIABLE":                       es_viable,
                    "ES_FULL_RESERVAS":                es_full,
                    "ES_EXTRAPOLADO":                  es_extrap,
                    "ES_CLIPPED":                      es_clipped,
                    # Auditoria 2026-07-07 H1/H3: BK_SUPERA_PRECIO_REF (piso capado,
                    # revisar BK con finanzas) o SIN_REANCLAJE (curva sin shift)
                    "ALERTA_BK":                       alerta_bk,
                    # Piso duro realmente aplicado (= BK abandono, o p_ref-1 si capado)
                    "PISO_EFECTIVO_USD_BBL":           round(_bk_eff, 2)
                                                       if _bk_eff is not None else None,
                    # Motor efectivo: ISOTONICA o PLANO_DECK (directriz §4)
                    "TIPO_MODELO":                     tipo_modelo,
                    "TIPO_DATO":                       "PREDICCIÓN",
                    "Q_OBJETIVO":                      q_objetivo,
                    "VIGENCIA_BASE":                   vigencia_base,
                    "FECHA_PREDICCION":                fecha_prediccion,
                })

        # Resumen impreso (motor primario, prediccion anclada)
        if "Isotonica" in modelos and pd.notna(baseline):
            d_ref = float(delta_refs["Isotonica"])
            nb = m2.neto_desde_brent(coef, np.array([brent_min_din, brent_max_din]))
            v = volumen_anclado(modelos["Isotonica"], nb, baseline, d_ref, _bk_dura,
                                p_ref=_p_ref_f)
            print(f"  {campo:<20} | baseline={baseline:.1f} MBPE | "
                  f"Iso@${brent_min_din}={v[0]:.0f} -> @${brent_max_din}={v[1]:.0f}")

    # Cobertura total del portafolio (directriz §4): campos con cierre sin modelo
    # entran como línea plana SIN_MODELO (filiales excluidas por política).
    campos_exportados = {f["CAMPO"] for f in filas}
    filas.extend(emitir_cobertura_plana(
        df, campos_exportados, brent_range,
        q_objetivo, vigencia_base, fecha_prediccion, BRENT_REF,
        cierre_2025, anio_baseline_map))

    df_out = pd.DataFrame(filas)

    # ── Clasificacion de confianza por campo (primario = Isotonica) ───────────
    if True:
        _met_raw = _met_full.copy()
        for _c in ["ALERTA_LOO_OUTLIER_ISO"]:
            if _c not in _met_raw.columns:
                _met_raw[_c] = False
        for _c in ["SKILL_ISO", "MAE_NAIVE", "MAE_LOYO_ISO", "SKILL_LOYO_ISO",
                   "N_VIGENCIAS_LOYO", "MAE_NAIVE_LOYO"]:
            if _c not in _met_raw.columns:
                _met_raw[_c] = np.nan
        met = _met_raw[["CAMPO", "N_REAL_DELTA", "MAE_LOO_ISO", "BASELINE_LATEST",
                        "ALERTA_LOO_OUTLIER_ISO", "SKILL_ISO", "MAE_NAIVE",
                        "MAE_LOYO_ISO", "SKILL_LOYO_ISO", "N_VIGENCIAS_LOYO",
                        "MAE_NAIVE_LOYO"]].copy()
        met["MAE_REL_LOO"] = (met["MAE_LOO_ISO"] /
                              met["BASELINE_LATEST"].replace(0, np.nan)).fillna(999.0)
        # Métrica de gate LOYO (contrato NORTE 2026-07-10): honesta cross-vigencia. Solo
        # donde hay ≥2 años de Consolidado; si no, LOO clásico (fallback). El gate lee
        # MAE_REL_EFF/MAE_ABS_EFF/SKILL_EFF; el LOO queda en el motivo como referencia.
        met["MAE_REL_LOYO"] = (met["MAE_LOYO_ISO"] /
                               met["BASELINE_LATEST"].replace(0, np.nan))
        _usa_loyo = (met["N_VIGENCIAS_LOYO"].fillna(0) >= 2) & met["MAE_LOYO_ISO"].notna()
        met["METRICA_BASE"]  = np.where(_usa_loyo, "LOYO", "LOO")
        met["MAE_ABS_EFF"]   = np.where(_usa_loyo, met["MAE_LOYO_ISO"], met["MAE_LOO_ISO"])
        met["MAE_REL_EFF"]   = np.where(_usa_loyo, met["MAE_REL_LOYO"], met["MAE_REL_LOO"])
        met["MAE_REL_EFF"]   = met["MAE_REL_EFF"].fillna(999.0)
        met["SKILL_EFF"]     = np.where(_usa_loyo, met["SKILL_LOYO_ISO"], met["SKILL_ISO"])
        met["SKILL_ALT"]     = np.where(_usa_loyo, met["SKILL_ISO"], met["SKILL_LOYO_ISO"])
        met["MAE_NAIVE_EFF"] = np.where(_usa_loyo, met["MAE_NAIVE_LOYO"], met["MAE_NAIVE"])
        met["N_REAL_DELTA"] = met["N_REAL_DELTA"].fillna(0).astype(int)
        met["ALERTA_LOO_OUTLIER_ISO"] = met["ALERTA_LOO_OUTLIER_ISO"].fillna(False).astype(bool)

        div_serie = calcular_divergencia_motores(df_out).rename("DIVERGENCIA_MOTORES_PCT")
        met = met.merge(div_serie, on="CAMPO", how="left")
        # Divergencia NO computable (sin pivote en banda 40-80) queda NaN: antes se
        # rellenaba con 999 y forzaba BAJA silenciosamente (auditoria 2026-07-07 H3).
        # clasificar_confianza trata NaN como "no evaluable", no como divergente.

        m2_fragil_map = {campo: es_m2_fragil(coef) for campo, coef in correlacion.items()}

        # Sesgo de recuperacion por campo = |d_ref ISO| / baseline (artifact 8a8f2cbc):
        # profundidad del valle donde cae el ancla p_ref. Solo Isotonica (motor primario).
        sesgo_recup_map = {}
        for _campo, _bl in baselines.items():
            if pd.notna(_bl) and _bl > 0:
                _dref = anclaje.get(_campo, {}).get("DELTA_REF_ISO", np.nan)
                if pd.notna(_dref):
                    sesgo_recup_map[_campo] = abs(float(_dref)) / float(_bl)

        rows_conf = []
        for _, r in met.iterrows():
            nivel, motivo = clasificar_confianza(
                n_real=int(r["N_REAL_DELTA"]),
                mae_rel=float(r["MAE_REL_EFF"]),
                divergencia=float(r["DIVERGENCIA_MOTORES_PCT"]),
                baseline=float(r["BASELINE_LATEST"]) if pd.notna(r["BASELINE_LATEST"]) else 0.0,
                mae_abs=float(r["MAE_ABS_EFF"]) if pd.notna(r["MAE_ABS_EFF"]) else 999.0,
                outlier_lloo=bool(r["ALERTA_LOO_OUTLIER_ISO"]),
                skill=float(r["SKILL_EFF"]) if pd.notna(r["SKILL_EFF"]) else np.nan,
                mae_naive=float(r["MAE_NAIVE_EFF"]) if pd.notna(r["MAE_NAIVE_EFF"]) else np.nan,
                m2_fragil=m2_fragil_map.get(r["CAMPO"], False),
                sens_no_ident=bool(confound.get(r["CAMPO"], {}).get("FLAG", False)),
                alerta_bk=alertas_bk.get(r["CAMPO"], ""),
                sesgo_recup=sesgo_recup_map.get(r["CAMPO"], np.nan),
                metrica_base=str(r["METRICA_BASE"]),
                skill_alt=float(r["SKILL_ALT"]) if pd.notna(r["SKILL_ALT"]) else np.nan)
            rows_conf.append({"CAMPO": r["CAMPO"], "N_REAL_DELTA": int(r["N_REAL_DELTA"]),
                              "MAE_REL_LOO": round(float(r["MAE_REL_LOO"]), 4),
                              "DIVERGENCIA_MOTORES_PCT": round(float(r["DIVERGENCIA_MOTORES_PCT"]), 4),
                              "ALERTA_LOO_OUTLIER_ISO": bool(r["ALERTA_LOO_OUTLIER_ISO"]),
                              "NIVEL_CONFIANZA": nivel, "MOTIVO_CONFIANZA": motivo,
                              # Confiabilidad (pagina "Confiabilidad del Modelo"): metricas
                              # LOYO crudas + sesgo de recuperacion, hoy solo visibles en el
                              # motivo como texto — aqui van como columnas propias.
                              "MAE_LOYO_MBPE": round(float(r["MAE_LOYO_ISO"]), 3)
                                               if pd.notna(r["MAE_LOYO_ISO"]) else None,
                              "SKILL_LOYO": round(float(r["SKILL_LOYO_ISO"]), 3)
                                            if pd.notna(r["SKILL_LOYO_ISO"]) else None,
                              "N_VIGENCIAS_LOYO": int(r["N_VIGENCIAS_LOYO"])
                                                  if pd.notna(r["N_VIGENCIAS_LOYO"]) else None,
                              "SESGO_RECUPERACION": round(sesgo_recup_map.get(r["CAMPO"], np.nan), 4)
                                                    if pd.notna(sesgo_recup_map.get(r["CAMPO"], np.nan)) else None})

        df_conf = pd.DataFrame(rows_conf)
        df_out = df_out.merge(
            df_conf[["CAMPO", "N_REAL_DELTA", "MAE_REL_LOO", "DIVERGENCIA_MOTORES_PCT",
                     "ALERTA_LOO_OUTLIER_ISO", "NIVEL_CONFIANZA", "MOTIVO_CONFIANZA",
                     "MAE_LOYO_MBPE", "SKILL_LOYO", "N_VIGENCIAS_LOYO", "SESGO_RECUPERACION"]],
            on="CAMPO", how="left")

        # Campos de cobertura (sin modelo): NIVEL=SIN_MODELO, predicción = cierre plano.
        _cov = df_out["TIPO_DATO"] == "SIN_MODELO"
        df_out.loc[_cov, "NIVEL_CONFIANZA"]  = "SIN_MODELO"
        df_out.loc[_cov, "MOTIVO_CONFIANZA"] = "sin deck ni breakeven — predicción = último cierre plano"
        df_out.loc[_cov, "N_REAL_DELTA"]     = 0

        conteo = df_conf.groupby("NIVEL_CONFIANZA")["CAMPO"].nunique()
        print(f"\n{'='*55}\n  Clasificacion de confianza por campo\n{'='*55}")
        for nivel in ["ALTA", "MEDIA", "BAJA", "INSENSIBLE_PRECIO", "SIN_MODELO"]:
            print(f"  {nivel:<15}: {conteo.get(nivel, 0):3d} campos")

        materiales = met[met["BASELINE_LATEST"].fillna(0) >= 50].merge(
            df_conf[["CAMPO", "NIVEL_CONFIANZA"]], on="CAMPO")
        if not materiales.empty:
            print("\n  Campos materiales (baseline >= 50 MBPE):")
            for _, r in materiales.sort_values("BASELINE_LATEST", ascending=False).iterrows():
                print(f"    {r['CAMPO']:<25} baseline={r['BASELINE_LATEST']:.0f} MBPE  "
                      f"MAE_rel={r['MAE_REL_LOO']:.0%}  -> {r['NIVEL_CONFIANZA']}")
        print(f"{'='*55}\n")

    ruta_csv = RESULTADOS / "output_matriz_prediccion.csv"
    df_out.to_csv(ruta_csv, index=False, encoding="utf-8-sig")

    # Matrices aisladas por modelo (directriz 2026-06-12)
    ruta_m1 = RESULTADOS / "output_matriz_modelo1.csv"
    ruta_m2_csv = RESULTADOS / "output_matriz_modelo2.csv"
    pd.DataFrame(filas_m1).to_csv(ruta_m1, index=False, encoding="utf-8-sig")
    pd.DataFrame(filas_m2).to_csv(ruta_m2_csv, index=False, encoding="utf-8-sig")
    print(f"  Matriz Modelo 1 (Aceite->Volumen, puro): {ruta_m1}")
    print(f"  Matriz Modelo 2 (Brent->Aceite, puro):   {ruta_m2_csv}")

    # Puntos para replicar los plots de staging en Power BI (modelo + reales + ancla)
    df_pm1, df_pm2 = construir_puntos_reales(df, filas_m1, filas_m2, baselines, anclaje)
    ruta_pm1 = RESULTADOS / "output_puntos_m1.csv"
    ruta_pm2 = RESULTADOS / "output_puntos_m2.csv"
    df_pm1.to_csv(ruta_pm1, index=False, encoding="utf-8-sig")
    df_pm2.to_csv(ruta_pm2, index=False, encoding="utf-8-sig")
    print(f"  Puntos M1 (curva+reales+ancla): {ruta_pm1} ({len(df_pm1)} filas)")
    print(f"  Puntos M2 (recta+reales HIST):  {ruta_pm2} ({len(df_pm2)} filas)")

    # ── Dimensiones para Power BI (estrella): atributos constantes por campo ──
    # Las matrices repiten ~35 columnas constantes en cada fila de Brent; el
    # modelo semantico las descarta al cargar y consulta esta dim (1 fila/campo).
    _cols_dim = ["CAMPO", "TIPO_MODELO", "VOLUMEN_1P_BASELINE_MBPE", "AÑO_BASELINE",
                 "CIERRE_2025_MBPE", "P_REF_USD_BBL", "PISO_EFECTIVO_USD_BBL",
                 "BREAKEVEN_USD_BBL", "PRECIO_EQUILIBRIO_USD_BBL",
                 "BREAKEVEN_REF_USD_BBL", "PRECIO_EQUILIBRIO_REF_USD_BBL",
                 "M2_METODO", "M2_ES_FALLBACK", "M2_N_PUNTOS", "M2_R2_LOO",
                 "M2_FRAGIL", "SENSIBILIDAD_NO_IDENTIFICADA", "CONFOUND_ETA2",
                 "ALERTA_BK", "NIVEL_CONFIANZA", "MOTIVO_CONFIANZA", "N_REAL_DELTA",
                 "MAE_REL_LOO", "DIVERGENCIA_MOTORES_PCT", "ALERTA_LOO_OUTLIER_ISO",
                 "MAE_LOYO_MBPE", "SKILL_LOYO", "N_VIGENCIAS_LOYO",
                 "SESGO_RECUPERACION"]
    df_dim = (df_out[[c for c in _cols_dim if c in df_out.columns]]
              .drop_duplicates("CAMPO").reset_index(drop=True))
    # ALPHA/BETA viven en la matriz M2; el metodo real del motor Suave (hibridos
    # s11/s12) en la matriz M1 — el de Isotonica es trivialmente el propio.
    df_dim = df_dim.merge(
        pd.DataFrame(filas_m2).drop_duplicates("CAMPO")
          [["CAMPO", "ALPHA", "BETA", "R2", "MAE_LOO", "ALERTA"]]
          .rename(columns={"R2": "M2_R2", "MAE_LOO": "M2_MAE_LOO",
                           "ALERTA": "M2_ALERTA"}),
        on="CAMPO", how="left")
    df_dim = df_dim.merge(
        pd.DataFrame(filas_m1).query("MOTOR == 'Suave'")
          .drop_duplicates("CAMPO")[["CAMPO", "METODO_REAL"]]
          .rename(columns={"METODO_REAL": "METODO_REAL_SUAVE"}),
        on="CAMPO", how="left")
    ruta_dim = RESULTADOS / "dim_campo_modelo.csv"
    df_dim.to_csv(ruta_dim, index=False, encoding="utf-8-sig")
    print(f"  Dim campo-modelo (1 fila/campo): {ruta_dim} ({len(df_dim)} filas)")

    # Metadata de la corrida (1 fila): evita repetir constantes globales por fila.
    ruta_corrida = RESULTADOS / "dim_corrida.csv"
    pd.DataFrame([{
        "Q_OBJETIVO":        q_objetivo,
        "VIGENCIA_BASE":     vigencia_base,
        "FECHA_PREDICCION":  fecha_prediccion,
        "BRENT_REF_USD_BBL": round(BRENT_REF, 2) if pd.notna(BRENT_REF) else None,
        "N_CAMPOS":          int(df_dim["CAMPO"].nunique()),
        "FECHA_EXPORT":      pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }]).to_csv(ruta_corrida, index=False, encoding="utf-8-sig")
    print(f"  Dim corrida (metadata): {ruta_corrida}")

    HISTORICO_DIR.mkdir(parents=True, exist_ok=True)
    nombre_snapshot = f"prediccion_{q_objetivo}_generada_{fecha_prediccion}.csv"
    ruta_snapshot = HISTORICO_DIR / nombre_snapshot
    shutil.copy2(ruta_csv, ruta_snapshot)
    print(f"  Snapshot fechado: {ruta_snapshot}")

    print(f"\nMatriz exportada: {ruta_csv}")
    print(f"  Filas: {len(df_out)}")
    print(f"  Extrapolados: {df_out['ES_EXTRAPOLADO'].sum()} "
          f"({df_out['ES_EXTRAPOLADO'].mean():.0%} del total)")
    print(f"  Clipped: {df_out['ES_CLIPPED'].sum()} "
          f"({df_out['ES_CLIPPED'].mean():.0%} del total)")

    # Sanity final: extremos del grid
    brent_lo = int(brent_range[0]); brent_hi = int(brent_range[-1])
    check_lo = df_out[df_out["BRENT_USD_BBL"] == brent_lo].groupby(
        ["CAMPO", "MOTOR"])["VOLUMEN_1P_PREDICHO_MBPE"].mean()
    check_hi = df_out[df_out["BRENT_USD_BBL"] == brent_hi].groupby(
        ["CAMPO", "MOTOR"])["VOLUMEN_1P_PREDICHO_MBPE"].mean()
    print(f"\n  Sanity Brent=${brent_lo} (extremo bajo): {len(check_lo)} series")
    print(f"  Sanity Brent=${brent_hi} (extremo alto): {len(check_hi)} series")

    # Sanity re-anclaje: en Brent=BRENT_REF el volumen debe ser el baseline exacto
    # (salvo p_ref < BK abandono: alli el piso duro manda y Vol=0)
    if pd.notna(BRENT_REF):
        ref = df_out.copy()
        ref["_dist"] = (ref["BRENT_USD_BBL"] - BRENT_REF).abs()
        idx = ref.groupby(["CAMPO", "MOTOR"])["_dist"].idxmin()
        ref = ref.loc[idx]
        ref = ref[ref["VOLUMEN_1P_BASELINE_MBPE"].notna()]
        # Tolerancia 0.5 MBPE = criterio C5 (CLAUDE.md). Con el guard H1 (piso capado
        # bajo p_ref) y BRENT_REF redondeado ANTES de p_ref (H2), el ancla debe ser
        # exacta; la unica excepcion legitima es SIN_REANCLAJE.
        dif = (ref["VOLUMEN_1P_PREDICHO_MBPE"] - ref["VOLUMEN_1P_BASELINE_MBPE"]).abs()
        sin_anclaje = (ref["ALERTA_BK"] == "SIN_REANCLAJE") \
            if "ALERTA_BK" in ref.columns else False
        ok = ((dif <= 0.5) | sin_anclaje).sum()
        print(f"\n  Sanity re-anclaje @Brent~{BRENT_REF:.1f}: {ok}/{len(ref)} series "
              f"con Vol=baseline (tol 0.5 MBPE, C5) o SIN_REANCLAJE")
        if ok < len(ref):
            peores = ref.loc[~((dif <= 0.5) | sin_anclaje)].assign(_dif=dif)
            print(peores.nlargest(5, "_dif")[["CAMPO", "MOTOR", "VOLUMEN_1P_PREDICHO_MBPE",
                                              "VOLUMEN_1P_BASELINE_MBPE"]].to_string(index=False))

    # ── Ledger de cobertura (transparencia para decisiones CAPEX) ────────────────
    # Cada campo del baseline aparece con su estado y motivo de ausencia.
    # Motivos: sin_consolidado (sin tabla sensibilidad forward, mayoria activos US/gas),
    #          filial_migracion (padre perdio cert 2024/25 a entidad " FILIAL"),
    #          consolidado_sin_sens (tiene CONSOLIDADO pero sin vol sensibilidad),
    #          presente (incluido en la matriz de prediccion).
    presentes_en_matriz = set(df_out["CAMPO"].unique())
    # nombre de la columna AÑO (puede tener encoding raro en el parquet)
    _col_anio = next((c for c in df.columns if c in ("AÑO",) or
                      (c.startswith("A") and c.endswith("O") and len(c) <= 4)), None)

    rows_cob = []
    for campo, vol in baselines.items():
        if campo in presentes_en_matriz:
            motivo = "presente"
        else:
            sub = df[df["CAMPO"] == campo]
            cons = sub[sub["ESCENARIO"].str.startswith("CONSOLIDADO", na=False)]
            n_cons = len(cons)
            n_sens = int(cons["VOLUMEN_1P_SENSIBILIDAD_MBPE"].notna().sum())
            # Tiene certificado reciente 2024/2025 en el padre?
            b2425 = (sub[(sub["ESCENARIO"] == "BASE") &
                         sub[_col_anio].isin([2024, 2025])]
                     if _col_anio else sub.iloc[0:0])
            tiene_base_reciente = bool(b2425["VOLUMEN_1P_OFICIAL_MBPE"].notna().any())
            if n_cons == 0:
                motivo = "sin_consolidado"
            elif n_sens > 0 and not tiene_base_reciente:
                motivo = "filial_migracion"
            else:
                motivo = "consolidado_sin_sens"
        _c25 = cierre_2025.get(campo)
        rows_cob.append({"CAMPO": campo,
                         "BASELINE_1P_MBPE": round(float(vol), 2),
                         "CIERRE_2025_MBPE": round(float(_c25), 2)
                                             if _c25 is not None and pd.notna(_c25) else None,
                         "EN_PREDICCION": campo in presentes_en_matriz,
                         "MOTIVO_AUSENCIA": motivo})

    df_cob = pd.DataFrame(rows_cob).sort_values(
        ["EN_PREDICCION", "BASELINE_1P_MBPE"], ascending=[True, False])
    ruta_cob = RESULTADOS / "cobertura_portafolio.csv"
    df_cob.to_csv(ruta_cob, index=False, encoding="utf-8-sig")

    # Rollup de cobertura (informe 2026-07-01 WS2.4): MBPE total y conteo por
    # motivo de ausencia, para declarar explicitamente ante finanzas cuanto
    # 1P del portafolio (ej. filiales sin_consolidado) queda fuera del piloto.
    df_resumen = (df_cob.groupby("MOTIVO_AUSENCIA")
                  .agg(N_CAMPOS=("CAMPO", "nunique"),
                       BASELINE_1P_MBPE=("BASELINE_1P_MBPE", "sum"))
                  .reset_index()
                  .sort_values("BASELINE_1P_MBPE", ascending=False))
    df_resumen["BASELINE_1P_MBPE"] = df_resumen["BASELINE_1P_MBPE"].round(2)
    ruta_resumen = RESULTADOS / "cobertura_resumen.csv"
    df_resumen.to_csv(ruta_resumen, index=False, encoding="utf-8-sig")

    total_mbpe = df_cob["BASELINE_1P_MBPE"].sum()
    pres_mbpe  = df_cob[df_cob["EN_PREDICCION"]]["BASELINE_1P_MBPE"].sum()
    print(f"\n{'='*55}")
    print(f"  COBERTURA DEL PORTAFOLIO  ({ruta_cob.name})")
    print(f"{'='*55}")
    print(f"  Total baseline:  {len(df_cob):3d} campos | {total_mbpe:.1f} MBPE")
    print(f"  En prediccion:   {df_cob['EN_PREDICCION'].sum():3d} campos | "
          f"{pres_mbpe:.1f} MBPE ({100*pres_mbpe/total_mbpe:.1f}%)")
    for mot, desc in [("sin_consolidado",    "sin datos fuente    "),
                      ("filial_migracion",   "migr. FILIAL recup. "),
                      ("consolidado_sin_sens", "Consol. sin sensib. ")]:
        sub_m = df_cob[df_cob["MOTIVO_AUSENCIA"] == mot]
        if not sub_m.empty:
            print(f"  Ausente {desc}: {len(sub_m):3d} campos | "
                  f"{sub_m['BASELINE_1P_MBPE'].sum():.1f} MBPE")
    print(f"{'='*55}\n")

    brent_ref = round((brent_obs_min + brent_obs_max) / 2)
    comp = generar_comparacion_vs_anterior(df_out, q_objetivo, brent_ref, ruta_snapshot)
    actualizar_changelog(q_objetivo, fecha_prediccion, len(campos), comp)

    print("\n=== 04_pbi_export.py — Completado ===")
