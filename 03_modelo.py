"""
03_modelo.py — Modelo 1: Delta Reservas 1P = f(Precio Neto) por campo

ARQUITECTURA DE 2 MODELOS (directriz 2026-06-11):
  Modelo 1 (este script): PRECIO_NETO -> DELTA_SENS_MBPE. La fisica de la sensibilidad
    de reservas vive en el precio realizado NETO (lo que el motor de FC certifica), no en
    el Brent. Es un problema 1D.
  Modelo 2 (03b_correlacion_brent.py): BRENT -> PRECIO_NETO por campo. Convierte el
    marcador Brent al precio neto que alimenta el Modelo 1. Permite Brent -> Delta por
    composicion en 04_pbi_export.py.

TARGET: DELTA_SENS_MBPE = ΔReservas = Sensibilidad − BASELINE_1P_MBPE (cierre del año
ANTERIOR, A−1). Desde 2026-07-09: el deck de Sensibilidad se construye sobre el último
libro certificado disponible (cierre A−1); restar el cierre del MISMO año (Checkpoint,
semántica previa) inyectaba el salto de recertificación A−1→A = confound de vigencia.

PREDICCION FINAL CON RE-ANCLAJE (decision 2026-06-12, MAESTRO §7.0/§10):
    VOLUMEN_1P_PRED(p) = max(BASELINE_LATEST + [f(p) − f(p_ref)], 0)
    p_ref = M2_campo(BRENT_REF);  BRENT_REF = Brent del ultimo quarter conocido.
    Piso duro: p < BK_ANCLA_PDP -> 0.
  El modelo aprende deltas contra el baseline de CADA vigencia; su promedio al precio
  actual no es 0 (niveles distintos por deplecion/revisiones). El re-anclaje impone la
  identidad financiera: a precio actual, reservas = ultimo 1P certificado. Requiere
  correlacion_brent.csv -> 03b corre ANTES que este script en run_pipeline.

MOTORES (elegidos por benchmark_modelo1.py, LOO-CV sobre 114 campos):
  - PRIMARIO  : Isotonica (MotorIsotonico) — menor MAE, respeta la escalera de anclas,
                100% monotona. Gano en mas campos por SKILL.
  - VALIDACION: Suave (MotorSuave, PCHIP sobre la espina isotonica) — segunda opinion
                suave/derivable que sigue de cerca a la primaria; su divergencia frente a
                la isotonica detecta problemas reales (no diferencias de forma de motor).
  - RETIRADO  : XGBoost. Era el primario por ser 3D (Brent+descuentos); en 1D pierde su
                ventaja y fue el PEOR del benchmark (skill mediano -0.074). Ademas su
                restriccion (1,-1,-1) sobre descuentos negativos invertia la monotonia
                (sintoma: a mayor precio neto, menor volumen en escenarios ALTOS).

Correcciones heredadas (vigentes):
  B1: LOO-CV calculado SOLO sobre puntos REALES. Los sinteticos van siempre al train
      (son el ancla fisica del modelo). mae_naive = referencia de skill.
  Sanity: monotonia, sub-breakeven≈0, asintota y divergencia verificadas en banda real.

Ver docs/MAESTRO.md §7 y §9 para justificacion del motor.
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from motores_modelo1 import (MotorIsotonico, MotorSuave, MotorSigmoide,
                             MotorLinealRobusto, MotorHuberIso, MotorFCPerfil,
                             MotorHibrido, CANDIDATOS_HIBRIDO,
                             piso_efectivo, volumen_anclado)
from track import sufijo_track, flag

_SUF = sufijo_track()   # '' Produccion; '_calidad' si PRED_TRACK=calidad
BASE_DIR    = Path(__file__).parent
STAGING     = BASE_DIR / "datos" / f"staging{_SUF}"
MODELOS_DIR = STAGING / "modelos"
PLOTS_DIR   = STAGING / "plots"
RESULTADOS  = BASE_DIR / f"resultados{_SUF}"
MODELOS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTADOS.mkdir(parents=True, exist_ok=True)

# Peso sintetico DINAMICO por nivel de escalera (anti-sesgo): masa total sintetica de cada
# nivel ≈ masa total de puntos reales. w_sint = n_real / n_nivel, con piso minimo para no
# anular el ancla fisica en campos con muchos sinteticos.
W_SINTETICO_MIN = 0.05

# Una sola feature: el Precio Neto (el Modelo 2 traduce Brent -> Neto aguas arriba)
FEATURE = "PRECIO_NETO_USD_BBL"
TARGET  = "DELTA_SENS_MBPE"


# Deck plano (directriz DIRECTRIZ_ESCALERA_DECK.md §4): un campo cuyo deck REAL no
# responde al precio (rango de deltas < DECK_PLANO_FRAC·baseline en ≥2 vigencias). Típico
# de campos de gas/GLP (Piedemonte/VPI: Pauto, Cupiagua Recetor) donde el ingreso fijo
# domina. Para estos la isotónica interpola una rampa espuria en la franja SIN datos
# entre el abandono sintético y la banda real (Pauto: −40% a Brent 60 sin ningún dato).
# El modelo honesto es "plano hasta el abandono": Vol = baseline sobre BK_PDP, 0 debajo.
DECK_PLANO_FRAC   = 0.02   # rango de deltas reales < 2%·baseline = insensible
DECK_PLANO_N_MIN  = 4      # mínimo de puntos reales para afirmar planitud

# Junta del modelo hibrido (investigacion 2026-07-15): donde termina el gobierno de
# la escalera de breakevens y arranca el motor superior. Mismo ancho de transicion
# que la escalera suavizada de 02_synthetic (s5).
HIBRIDO_TRANSICION_USD = 6.0

# Re-anclaje selectivo por confound (track Calidad, directriz §4bis.7.1). Discriminante
# SEGURO = sesgo material Y η² alto Y materialidad. El gate dorado tiene η² alto pero
# sesgo≈0 (su re-anclaje ya es sano) → queda fuera. La version GLOBAL de este centrado
# fue rechazada (experimento_vigencia, aplana sensibilidad real): por eso es selectivo.
REANCL_BASE_MIN  = 5.0     # MBPE — materialidad minima
REANCL_ETA2_MIN  = 0.4     # el AÑO explica ≥40% de la varianza del delta
REANCL_SESGO_MIN = 0.15    # |d_ref|/baseline > 15% = artefacto de recuperacion material


def eta2_vigencia(df_campo: pd.DataFrame) -> float:
    """η²: fraccion de la varianza del delta real explicada por el AÑO de vigencia.
    Misma formula que 04_pbi_export.calcular_confound_vigencia (reproduce sus valores).
    """
    real = df_campo[~df_campo["ES_SINTETICO"]]
    y = real[TARGET].dropna()
    if len(y) < 4:
        return np.nan
    grp = real.loc[y.index, "VIGENCIA"].astype(str).str.slice(0, 4)
    if grp.nunique() < 2:
        return np.nan
    gm = y.mean()
    ss_tot = float(((y - gm) ** 2).sum())
    if ss_tot <= 0:
        return np.nan
    ss_between = sum(len(sub) * (sub.mean() - gm) ** 2 for _, sub in y.groupby(grp))
    return float(ss_between / ss_tot)


def centrar_vigencias(df_campo: pd.DataFrame, fraccion: float = 1.0) -> pd.DataFrame:
    """Centra los deltas REALES por vigencia: quita el offset entre decks (recertificacion
    inter-cierre) y conserva la pendiente intra-vigencia (la respuesta real al precio).
    Sinteticos intactos. Re-anclaje selectivo por confound (directriz §4bis.7.1).

    fraccion (centrado PARCIAL, s9): cuanto del offset se remueve. 1.0 = centrado total
    (re-anclaje clasico). fraccion = η² = shrinkage tipo empirical-Bayes: se quita solo
    la fraccion de offset que la evidencia atribuye al AÑO — para casos frontera donde
    precio y año estan mezclados (CASABE η²=0.44) y el centrado total borra señal real."""
    out = df_campo.copy()
    real = out[~out["ES_SINTETICO"]]
    y = real[TARGET]
    grp = real["VIGENCIA"].astype(str).str.slice(0, 4)
    offset = grp.map(y.groupby(grp).mean()) - y.mean()
    out.loc[real.index, TARGET] = (y - fraccion * offset).values
    return out


def es_deck_plano(df_campo: pd.DataFrame, baseline: float) -> bool:
    """True si el deck real del campo es insensible al precio (directriz §4)."""
    if pd.isna(baseline) or baseline <= 0:
        return False
    reales = df_campo[~df_campo["ES_SINTETICO"]]
    deltas = reales[TARGET].dropna()
    if len(deltas) < DECK_PLANO_N_MIN:
        return False
    n_vig = reales.loc[deltas.index, "VIGENCIA"].astype(str).str.slice(0, 4).nunique()
    if n_vig < 2:
        return False
    rango = float(deltas.max() - deltas.min())
    return rango < DECK_PLANO_FRAC * float(baseline)


def peso_sintetico(n_real: int, n_sint: int) -> float:
    """Peso por sintetico para que su masa total ≈ la masa total de puntos reales."""
    if n_sint == 0:
        return 1.0
    return max(n_real / n_sint, W_SINTETICO_MIN)


def pesos_sinteticos_tramo(df_sint: pd.DataFrame, n_real: int) -> tuple:
    """
    Peso por fila sintetica, calculado POR NIVEL de la escalera por separado: cada nivel
    (vol=0, solo-PDP, PDP+clase, ...) balancea su masa total contra n_real de forma
    independiente. Sin esto, un escalon con muchos puntos diluye el ancla critica BAJO
    (vol=0, RANGO_USD=5 → solo 5 puntos) y el modelo deja de anclar vol≈0 sub-breakeven.

    Retorna (pesos alineados a df_sint, dict {nivel_redondeado: peso}).
    """
    niveles = df_sint["VOLUMEN_1P_SENSIBILIDAD_MBPE"].round(2).values
    w_nivel = {}
    for nv in np.unique(niveles):
        n_nv = int((niveles == nv).sum())
        w_nivel[float(nv)] = peso_sintetico(n_real, n_nv)
    pesos = np.array([w_nivel[float(nv)] for nv in niveles])
    return pesos, w_nivel


# ── Helpers ────────────────────────────────────────────────────────────────────

def preparar_datos_campo(df: pd.DataFrame, campo: str) -> pd.DataFrame:
    """
    Filtra datos de entrenamiento de un campo: filas con TARGET y Precio Neto disponibles.
    Incluye sinteticos (ancla fisica) y CONSOLIDADO con delta; excluye filas BASE
    (delta=NaN) y CONSOLIDADO sin target. Asi el modelo aprende la FORMA de la
    sensibilidad a precio, no la tendencia temporal de deplecion/revisiones.
    """
    sub = df[df["CAMPO"] == campo].copy()
    mask = sub[TARGET].notna() & sub[FEATURE].notna()
    return sub[mask].reset_index(drop=True)


def loo_cv_solo_reales(df_campo: pd.DataFrame) -> tuple:
    """
    B1: LOO-CV solo sobre puntos REALES. Sinteticos siempre van al train.

    Para cada punto real i: train = sinteticos + reales\{i}; test = real[i].
    Ambos motores (Isotonica primaria, Suave validacion) se evaluan en Precio Neto.

    Retorna (r2_iso, mae_iso, rmse_iso, me_iso, r2_su, mae_su, rmse_su, me_su, mae_naive).
    mae_naive = MAE del predictor ingenuo (media de deltas reales del fold): referencia de
    skill para N pequeño, donde R² en banda de precios estrecha castiga sin que el modelo
    sea malo.
    """
    df_real = df_campo[~df_campo["ES_SINTETICO"]].reset_index(drop=True)
    df_sint = df_campo[df_campo["ES_SINTETICO"]].reset_index(drop=True)

    n_real = len(df_real)
    if n_real < 2:
        return (np.nan,) * 9

    x_sint = df_sint[FEATURE].values if len(df_sint) > 0 else np.empty(0)
    y_sint = df_sint[TARGET].values  if len(df_sint) > 0 else np.empty(0)
    # Pesos: sinteticos anclan el piso por nivel; reales aprenden la curva (peso 1.0)
    w_sint = pesos_sinteticos_tramo(df_sint, n_real)[0] if len(df_sint) > 0 else np.empty(0)

    y_pred_iso   = np.zeros(n_real, dtype=float)
    y_pred_su    = np.zeros(n_real, dtype=float)
    y_pred_naive = np.zeros(n_real, dtype=float)

    for i in range(n_real):
        mask_tr = np.ones(n_real, dtype=bool)
        mask_tr[i] = False
        x_tr_real = df_real.loc[mask_tr, FEATURE].values
        y_tr_real = df_real.loc[mask_tr, TARGET].values

        x_train = np.concatenate([x_sint, x_tr_real]) if len(x_sint) > 0 else x_tr_real
        y_train = np.concatenate([y_sint, y_tr_real]) if len(y_sint) > 0 else y_tr_real
        w_train = np.concatenate([w_sint, np.ones(mask_tr.sum())]) if len(w_sint) > 0 \
            else np.ones(mask_tr.sum())

        x_test = [df_real.loc[i, FEATURE]]
        y_pred_iso[i] = float(MotorIsotonico().fit(x_train, y_train, sample_weight=w_train)
                              .predict(x_test)[0])
        y_pred_su[i]  = float(MotorSuave().fit(x_train, y_train, sample_weight=w_train)
                              .predict(x_test)[0])
        # Predictor ingenuo: media de deltas reales del fold (sin sinteticos)
        y_pred_naive[i] = float(np.mean(y_tr_real))

    y_true = df_real[TARGET].values

    def _metricas(y_t, y_p):
        r2   = r2_score(y_t, y_p) if len(y_t) > 1 else np.nan
        mae  = mean_absolute_error(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        me   = float(np.mean(y_p - y_t))   # bias: >0 sobreestima, <0 subestima
        return r2, mae, rmse, me

    r2i, maei, rmsei, mei = _metricas(y_true, y_pred_iso)
    r2s, maes, rmses, mes = _metricas(y_true, y_pred_su)
    mae_naive = mean_absolute_error(y_true, y_pred_naive)
    return r2i, maei, rmsei, mei, r2s, maes, rmses, mes, mae_naive


def loyo_cv_por_vigencia(df_campo: pd.DataFrame) -> tuple:
    """
    LOYO-CV (leave-one-YEAR-out, auditoria NORTE 2026-07-02): folds por AÑO de
    vigencia del Consolidado, no por punto.

    POR QUE: el LOO clasico deja 1 punto fuera y lo predicen los otros 3 del MISMO
    año (delta plano intra-vigencia) → subestima el error de generalizacion 4-9x en
    el gate dorado (CASTILLA: MAE_LOO=6.1 vs MAE_LOYO=45.8). El LOYO hace la pregunta
    del negocio: "¿puedes predecir el año que no viste?" — que es exactamente predecir
    el siguiente quarter. Es la metrica honesta del confound de vigencia (§3ter del
    informe): donde LOYO >> LOO, la curva aprendio el salto entre decks anuales, no
    una respuesta al precio.

    Train = sinteticos (pesos por nivel) + reales de los OTROS años; test = reales del
    año excluido. Ingenuo del fold: media de los deltas reales de train (mismo ingenuo
    que LOO, evaluado cross-año). SKILL_LOYO = 1 − MAE_LOYO/MAE_NAIVE_LOYO.

    Retorna (mae_loyo_iso, mae_naive_loyo, skill_loyo_iso, n_vigencias).
    NaN si el campo no tiene ≥2 años con puntos reales.
    """
    df_real = df_campo[~df_campo["ES_SINTETICO"]].reset_index(drop=True)
    df_sint = df_campo[df_campo["ES_SINTETICO"]].reset_index(drop=True)

    n_real = len(df_real)
    anios = df_real["VIGENCIA"].astype(str).str.slice(0, 4)
    anios_unicos = sorted(anios.unique())
    if n_real < 2 or len(anios_unicos) < 2:
        return (np.nan, np.nan, np.nan, len(anios_unicos))

    x_sint = df_sint[FEATURE].values if len(df_sint) > 0 else np.empty(0)
    y_sint = df_sint[TARGET].values  if len(df_sint) > 0 else np.empty(0)
    w_sint = pesos_sinteticos_tramo(df_sint, n_real)[0] if len(df_sint) > 0 else np.empty(0)

    abs_err_iso, abs_err_naive = [], []
    for anio in anios_unicos:
        mask_te = (anios == anio).values
        mask_tr = ~mask_te
        if mask_tr.sum() < 2:
            continue
        x_tr_real = df_real.loc[mask_tr, FEATURE].values
        y_tr_real = df_real.loc[mask_tr, TARGET].values

        x_train = np.concatenate([x_sint, x_tr_real]) if len(x_sint) > 0 else x_tr_real
        y_train = np.concatenate([y_sint, y_tr_real]) if len(y_sint) > 0 else y_tr_real
        w_train = np.concatenate([w_sint, np.ones(mask_tr.sum())]) if len(w_sint) > 0 \
            else np.ones(mask_tr.sum())

        x_test = df_real.loc[mask_te, FEATURE].values
        y_test = df_real.loc[mask_te, TARGET].values
        y_hat = MotorIsotonico().fit(x_train, y_train, sample_weight=w_train).predict(x_test)
        abs_err_iso.extend(np.abs(np.asarray(y_hat) - y_test).tolist())
        abs_err_naive.extend(np.abs(float(np.mean(y_tr_real)) - y_test).tolist())

    if not abs_err_iso:
        return (np.nan, np.nan, np.nan, len(anios_unicos))
    mae_loyo = float(np.mean(abs_err_iso))
    mae_naive_loyo = float(np.mean(abs_err_naive))
    skill_loyo = 1 - mae_loyo / mae_naive_loyo if mae_naive_loyo > 0.01 else np.nan
    return mae_loyo, mae_naive_loyo, skill_loyo, len(anios_unicos)


def residuales_loyo(df_campo: pd.DataFrame, factory=MotorIsotonico) -> np.ndarray:
    """Residuales FIRMADOS del LOYO (y_real − y_pred) para las bandas de incertidumbre
    (track Calidad, s9): cuanto se desvió el delta real del año no visto respecto a lo
    que la curva (entrenada sin ese año) predijo. Cuantiles P10/P90 de estos residuales
    = banda empírica honesta alrededor de la curva. Mismo protocolo que
    loyo_cv_por_vigencia (train = sintéticos + otros años). Vacío si <2 vigencias.

    factory: constructor del motor a evaluar. Default isotónica (champion); para
    campos con híbrido adoptado (2026-07-15) la banda se recalcula CON el híbrido —
    la incertidumbre publicada debe reflejar el modelo realmente servido."""
    df_real = df_campo[~df_campo["ES_SINTETICO"]].reset_index(drop=True)
    df_sint = df_campo[df_campo["ES_SINTETICO"]].reset_index(drop=True)
    n_real = len(df_real)
    anios = df_real["VIGENCIA"].astype(str).str.slice(0, 4)
    uni = sorted(anios.unique())
    if n_real < 2 or len(uni) < 2:
        return np.array([])
    x_s = df_sint[FEATURE].values if len(df_sint) else np.empty(0)
    y_s = df_sint[TARGET].values if len(df_sint) else np.empty(0)
    w_s = pesos_sinteticos_tramo(df_sint, n_real)[0] if len(df_sint) else np.empty(0)
    resid = []
    for a in uni:
        te = (anios == a).values
        tr = ~te
        if tr.sum() < 2:
            continue
        x_t = np.concatenate([x_s, df_real.loc[tr, FEATURE].values]) if len(x_s) else df_real.loc[tr, FEATURE].values
        y_t = np.concatenate([y_s, df_real.loc[tr, TARGET].values]) if len(y_s) else df_real.loc[tr, TARGET].values
        w_t = np.concatenate([w_s, np.ones(tr.sum())]) if len(w_s) else np.ones(tr.sum())
        y_h = factory().fit(x_t, y_t, sample_weight=w_t).predict(df_real.loc[te, FEATURE].values)
        resid.extend((df_real.loc[te, TARGET].values - np.asarray(y_h)).tolist())
    return np.array(resid)


def entrenar_final_campo(df_campo: pd.DataFrame) -> tuple:
    """Entrena los modelos finales con TODOS los datos (reales + sinteticos), en Precio
    Neto. Pesos sinteticos balanceados por nivel; los reales pesan 1.0.
    Retorna (modelo_primario_iso, modelo_validacion_suave)."""
    x   = df_campo[FEATURE].values
    y   = df_campo[TARGET].values
    es_sint = df_campo["ES_SINTETICO"].values
    n_real  = int((~es_sint).sum())
    pesos = np.ones(len(df_campo))
    if es_sint.any():
        w_sint, _ = pesos_sinteticos_tramo(df_campo[es_sint], n_real)
        pesos[es_sint] = w_sint

    m_iso   = MotorIsotonico().fit(x, y, sample_weight=pesos)
    m_suave = MotorSuave().fit(x, y, sample_weight=pesos)
    return m_iso, m_suave


def _perfil_de_campo(df_campo: pd.DataFrame) -> dict:
    """BK_P10..90 del campo desde el tablon (solo presente en track Calidad con perfil)."""
    out = {}
    for pct in (10, 25, 50, 75, 90):
        col = f"BK_P{pct}"
        if col in df_campo.columns:
            v = df_campo[col].dropna()
            if len(v):
                out[pct] = float(v.iloc[0])
    return out


def p_junta_campo(df_campo: pd.DataFrame) -> float:
    """Junta del hibrido POR CAMPO (no constante global — en campos con BK alto,
    ej. CAÑO SUR, la junta puede caer dentro del rango real y debe validarse):
    max(techo de la banda BK + transicion, ultimo precio sintetico) = donde termina
    la evidencia sintetica que gobierna la escalera."""
    bk_fin, _, _ = _anclas_campo(df_campo)
    sint = df_campo[df_campo["ES_SINTETICO"]]
    p_sint_max = float(sint[FEATURE].max()) if len(sint) else np.nan
    candidatos = [v for v in ((bk_fin + HIBRIDO_TRANSICION_USD
                               if bk_fin is not None else np.nan), p_sint_max)
                  if pd.notna(v)]
    return max(candidatos) if candidatos else np.nan


# Dispatch de motor por metodo (track Calidad, PRED_SELECCION_METODO). El default es la
# isotonica sobre la escalera vigente (perfil si PRED_PERFIL_SALIDA). reanclaje se maneja
# como transform (centrar) + isotonica; FCPerfil ignora los datos (curva del FC);
# hibrido:<motor> = escalera abajo + motor superior arriba (investigacion 2026-07-15).
def entrenar_con_metodo(df_campo, metodo, baseline, bk_sup, bk_inf,
                        metodo_validacion: str = None, p_junta: float = None):
    """Entrena el par (primario, validacion).
    Primario = metodo seleccionado (hibrido:* | Sigmoide | … | default isotonica).
    Validacion = METODO_VALIDACION del registro (2.o mejor LOYO, s12); si no hay,
    cae a Suave (comportamiento clasico Isotonica+Suave)."""
    x = df_campo[FEATURE].values
    y = df_campo[TARGET].values
    es_sint = df_campo["ES_SINTETICO"].values
    n_real = int((~es_sint).sum())
    pesos = np.ones(len(df_campo))
    if es_sint.any():
        pesos[es_sint] = pesos_sinteticos_tramo(df_campo[es_sint], n_real)[0]

    # Primario
    if metodo.startswith("hibrido:"):
        nombre_sup = metodo.split(":", 1)[1]
        pj = p_junta if pd.notna(p_junta) else p_junta_campo(df_campo)
        if nombre_sup in CANDIDATOS_HIBRIDO and pd.notna(pj):
            iso = MotorHibrido(CANDIDATOS_HIBRIDO[nombre_sup], pj) \
                .fit(x, y, sample_weight=pesos)
        else:
            iso = MotorIsotonico().fit(x, y, sample_weight=pesos)
    elif metodo == "FCPerfil":
        per = _perfil_de_campo(df_campo)
        if per and pd.notna(bk_sup) and pd.notna(bk_inf) and bk_sup > bk_inf:
            iso = MotorFCPerfil(per, baseline, bk_sup, bk_inf).fit(x, y)
        else:
            iso = MotorIsotonico().fit(x, y, sample_weight=pesos)
    elif metodo == "Sigmoide":
        iso = MotorSigmoide().fit(x, y, sample_weight=pesos)
    elif metodo == "LinealRobusto":
        iso = MotorLinealRobusto().fit(x, y, sample_weight=pesos)
    elif metodo == "HuberIso":
        iso = MotorHuberIso().fit(x, y, sample_weight=pesos)
    else:
        iso = MotorIsotonico().fit(x, y, sample_weight=pesos)

    # Validacion: 2.o mejor del ranking (s12) o Suave por defecto
    import seleccion_reglas as R
    if metodo_validacion and str(metodo_validacion).strip() not in ("", "nan", "None"):
        pj = p_junta if pd.notna(p_junta) else p_junta_campo(df_campo)
        factory_v = R.factory_desde_metodo(str(metodo_validacion), pj)
        suave = factory_v().fit(x, y, sample_weight=pesos)
    else:
        suave = MotorSuave().fit(x, y, sample_weight=pesos)
    return iso, suave


def cargar_registro_seleccion() -> dict:
    """Lee resultados_calidad/seleccion_metodos.csv → {campo: dict}.

    Gobernanza por track (2026-07-15): Produccion solo aplica ESTADO=ADOPTADO
    (metodos ratificados por finanzas); el track Calidad tambien aplica
    ADOPTADO_CALIDAD (hibridos/M2 con evidencia >5% s12 + checks fisicos,
    pendientes de ratificacion). Cada valor incluye METODO y opcionalmente
    METODO_VALIDACION (2.o mejor LOYO, s12)."""
    ruta = BASE_DIR / "resultados_calidad" / "seleccion_metodos.csv"
    if not ruta.exists():
        return {}
    reg = pd.read_csv(ruta)
    estados = {"ADOPTADO"} if not _SUF else {"ADOPTADO", "ADOPTADO_CALIDAD"}
    reg = reg[reg["ESTADO"].isin(estados)]
    out = {}
    for _, r in reg.iterrows():
        out[r["CAMPO"]] = {
            "METODO": r["METODO"],
            "METODO_VALIDACION": r.get("METODO_VALIDACION", None)
            if "METODO_VALIDACION" in reg.columns else None,
            "MEJORA_PCT": r.get("MEJORA_PCT", None),
        }
    return out


def _anclas_campo(df_campo: pd.DataFrame) -> tuple:
    """
    Anclas de breakeven del campo (convencion finanzas, post-swap en 01_etl):
      BK_ANCLA_FIN = piso SUPERIOR (delta=0; bajo el, PNP/PND no son viables)
      BK_ANCLA_PDP = piso INFERIOR (abandono; bajo el, vol 1P = 0)
    Son las MISMAS anclas que usa 02_synthetic. Seleccion identica a generar_sinteticos:
    vigencia MAS RECIENTE con anclas. Retorna (bk_fin, bk_pdp, vigencia).
    """
    cols = ["BK_ANCLA_FIN_USD_BBL", "BK_ANCLA_PDP_USD_BBL"]
    if not all(c in df_campo.columns for c in cols):
        return None, None, None
    anclas_sub = df_campo.dropna(subset=cols, how="all")
    if anclas_sub.empty:
        return None, None, None
    fila = anclas_sub.sort_values("VIGENCIA_BREAKEVEN", ascending=False).iloc[0]
    bk_fin = float(fila[cols[0]]) if pd.notna(fila[cols[0]]) else None
    bk_pdp = float(fila[cols[1]]) if pd.notna(fila[cols[1]]) else None
    if bk_fin is None:
        bk_fin = bk_pdp
    if bk_pdp is None:
        bk_pdp = bk_fin
    vbk = str(fila["VIGENCIA_BREAKEVEN"]) if pd.notna(fila.get("VIGENCIA_BREAKEVEN")) else None
    return bk_fin, bk_pdp, vbk


def generar_plot(campo: str, df_campo: pd.DataFrame,
                 iso_model, suave_model,
                 medianas_px: dict, baseline_latest: float,
                 baseline_pdp: float = 0.0,
                 p_ref: float = None,
                 delta_ref_iso: float = 0.0,
                 delta_ref_su: float = 0.0,
                 p_junta: float = None) -> None:
    """
    Plot de validacion. Eje X = PRECIO NETO (la variable nativa del Modelo 1). Eje X
    superior secundario = Brent equivalente usando las medianas de descuento del campo
    (solo referencia visual; la conversion formal Brent↔Neto la hace el Modelo 2).
    Panel izq = espacio delta (modelo crudo); panel der = volumen absoluto ANCLADO
    (baseline + delta − delta_ref, decision 2026-06-12) con marcador en (p_ref, baseline).
    """
    bk_fin, bk_pdp, _ = _anclas_campo(df_campo)

    med_cal = medianas_px.get("DESCUENTO_CALIDAD_USD_BBL", -7.0)
    med_tra = medianas_px.get("DESCUENTO_TRANSPORTE_USD_BBL", -3.3)

    # Grid en PRECIO NETO (eje del modelo)
    pneto_min = df_campo[FEATURE].min() - 5
    pneto_max = df_campo[FEATURE].max() + 10
    pneto_grid = np.linspace(pneto_min, pneto_max, 200)

    _bk_plot = bk_pdp if bk_fin is not None else None
    curva_iso_delta = iso_model.predict(pneto_grid)
    curva_su_delta  = suave_model.predict(pneto_grid)
    curva_iso_abs   = volumen_anclado(iso_model, pneto_grid, baseline_latest,
                                      delta_ref_iso, _bk_plot)
    curva_su_abs    = volumen_anclado(suave_model, pneto_grid, baseline_latest,
                                      delta_ref_su, _bk_plot)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    reales = df_campo[~df_campo["ES_SINTETICO"]]
    sint   = df_campo[df_campo["ES_SINTETICO"]]

    # Panel izquierdo: espacio delta vs Precio Neto
    ax1.scatter(reales[FEATURE], reales[TARGET],
                color="steelblue", s=70, zorder=5, label="Datos reales (delta)")
    ax1.scatter(sint[FEATURE], sint[TARGET],
                color="tomato", s=30, alpha=0.6, marker="x", zorder=4,
                label=f"Sinteticos (n={len(sint)})")
    ax1.plot(pneto_grid, curva_iso_delta, color="darkorange",
             linewidth=2.0, label="Isotonica Δ (primario)")
    ax1.plot(pneto_grid, curva_su_delta, color="purple",
             linewidth=1.5, linestyle="--", label="Suave Δ (validacion)")

    escalera_degenerada = (bk_fin is not None and bk_pdp is not None
                           and bk_fin <= bk_pdp + 1e-9)
    delta_escalon = -(baseline_latest - baseline_pdp)
    if bk_fin is not None:
        if escalera_degenerada:
            ax1.axvline(x=bk_fin, color="gray", linestyle=":", linewidth=1.5,
                        label=f"BK degenerado={bk_fin:.1f}")
            ax1.scatter([bk_fin], [0], color="gray", marker="v", s=90, zorder=6)
        else:
            ax1.axvline(x=bk_fin, color="gray", linestyle=":", linewidth=1.5,
                        label=f"BK_fin={bk_fin:.1f} (contabilizacion)")
            ax1.scatter([bk_fin], [0], color="gray", marker="v", s=90, zorder=6)
            ax1.axvline(x=bk_pdp, color="firebrick", linestyle=":", linewidth=1.5,
                        label=f"BK_pdp={bk_pdp:.1f} (abandono)")
            ax1.scatter([bk_pdp], [delta_escalon], color="firebrick",
                        marker="v", s=90, zorder=6)
    # Junta del hibrido (si aplica): frontera escalera <-> motor superior
    if p_junta is not None and pd.notna(p_junta):
        ax1.axvline(x=p_junta, color="purple", linestyle="-.", linewidth=1.3,
                    alpha=0.8, label=f"Junta hibrido={p_junta:.1f}")
    ax1.axhline(y=0, color="black", linewidth=0.5, alpha=0.4)
    ax1.set_xlabel("Precio Neto (USD/bbl)")
    ax1.set_ylabel("ΔReservas 1P (MBPE)")
    ax1.set_title(f"{campo} — Espacio delta")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.3)
    # Eje secundario superior: Brent equivalente (medianas del campo, referencia visual)
    sec1 = ax1.secondary_xaxis(
        "top", functions=(lambda p: p - med_cal - med_tra,
                          lambda b: b + med_cal + med_tra))
    sec1.set_xlabel("Brent equivalente (USD/bbl, medianas)", fontsize=8)

    # Panel derecho: volumen absoluto reconstruido
    ax2.scatter(reales[FEATURE],
                baseline_latest + reales[TARGET].fillna(0),
                color="steelblue", s=70, zorder=5, label="Reales (absoluto)")
    ax2.plot(pneto_grid, curva_iso_abs, color="darkorange",
             linewidth=2.0, label="Isotonica (absoluto)")
    ax2.plot(pneto_grid, curva_su_abs, color="purple",
             linewidth=1.5, linestyle="--", label="Suave (absoluto)")
    ax2.axhline(y=baseline_latest, color="purple", linewidth=1.0, linestyle=":",
                label=f"Baseline={baseline_latest:.0f}")
    if p_ref is not None and pd.notna(baseline_latest):
        ax2.scatter([p_ref], [baseline_latest], color="green", marker="*", s=180,
                    zorder=7, label=f"Ancla actual (p_ref={p_ref:.1f})")
    if bk_fin is not None:
        ax2.axvline(x=bk_fin, color="gray", linestyle=":", linewidth=1.5)
        ax2.scatter([bk_fin], [baseline_latest], color="gray", marker="v",
                    s=90, zorder=6)
        if not escalera_degenerada:
            ax2.axvline(x=bk_pdp, color="firebrick", linestyle=":", linewidth=1.5)
            ax2.scatter([bk_pdp], [baseline_pdp], color="firebrick", marker="v",
                        s=90, zorder=6)
            if baseline_pdp > 0:
                ax2.axhline(y=baseline_pdp, color="firebrick", linewidth=0.8,
                            linestyle="--", alpha=0.5,
                            label=f"Nivel PDP={baseline_pdp:.0f}")
    if p_junta is not None and pd.notna(p_junta):
        ax2.axvline(x=p_junta, color="purple", linestyle="-.", linewidth=1.3, alpha=0.8)
    ax2.set_xlabel("Precio Neto (USD/bbl)")
    ax2.set_ylabel("Volumen 1P (MBPE)")
    ax2.set_title(f"{campo} — Volumen absoluto (baseline + delta)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    sec2 = ax2.secondary_xaxis(
        "top", functions=(lambda p: p - med_cal - med_tra,
                          lambda b: b + med_cal + med_tra))
    sec2.set_xlabel("Brent equivalente (USD/bbl, medianas)", fontsize=8)

    plt.suptitle(campo, fontsize=13)
    plt.tight_layout()
    ruta = PLOTS_DIR / f"{campo.replace(' ', '_')}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot guardado: {ruta.name}")


def sanity_check(campo: str, iso_model, suave_model,
                 bk_fin: float, bk_pdp: float,
                 vol_max_delta: float, baseline_latest: float,
                 baseline_pdp: float = 0.0,
                 df_sint: pd.DataFrame = None,
                 pneto_reales: np.ndarray = None) -> dict:
    """
    Criterios funcionales del piloto, evaluados NATIVAMENTE en Precio Neto (ambos motores
    son 1D). Retorna dict con flags PASS/FAIL.
      C1  : bajo BK_pdp (abandono) → vol reconstruido ≈ 0.
      C1b : cada escalon de la escalera multi-clase → vol cerca del nivel del escalon.
      C2  : monotonia en banda [BK_fin+2, BK_fin+50].
      C3  : asintota superior (delta satura cerca del maximo historico) en BK_fin+60.
      C4  : divergencia primario (Isotonica) vs validacion (Suave) < 30% en banda real.
    """
    resultados = {}

    # C1: bajo el piso de abandono (BK_pdp) → delta ≈ -BASELINE (reservas nulas)
    px_sub  = bk_pdp - 5
    d_iso_s = float(iso_model.predict([px_sub])[0])
    d_su_s  = float(suave_model.predict([px_sub])[0])
    c1 = (baseline_latest + d_iso_s < 5) and (baseline_latest + d_su_s < 5)
    resultados["sub_breakeven_vol_cero"] = c1
    print(f"  [{'PASS' if c1 else 'FAIL'}] Sub-abandono (BK_pdp-5) vol reconstruido < 5 MBPE: "
          f"Iso={baseline_latest + d_iso_s:.1f}, Suave={baseline_latest + d_su_s:.1f}")

    # C1b: escalones de la escalera multi-clase
    escalones = []
    if df_sint is not None and len(df_sint) > 0:
        niveles_sint = df_sint["VOLUMEN_1P_SENSIBILIDAD_MBPE"].round(2)
        for nivel, grupo in df_sint[niveles_sint > 0].groupby(niveles_sint[niveles_sint > 0]):
            escalones.append((float(grupo["PRECIO_NETO_USD_BBL"].mean()), float(nivel)))
    elif bk_fin - bk_pdp > 1.0 and baseline_pdp > 0:
        escalones.append(((bk_pdp + bk_fin) / 2, baseline_pdp))

    if escalones:
        c1b_total = True
        for px_esc, nivel in sorted(escalones):
            d_iso_e = float(iso_model.predict([px_esc])[0])
            d_su_e  = float(suave_model.predict([px_esc])[0])
            tol = max(0.3 * nivel, 5.0)
            v_iso_e = baseline_latest + d_iso_e
            v_su_e  = baseline_latest + d_su_e
            ok = (abs(v_iso_e - nivel) <= tol) and (abs(v_su_e - nivel) <= tol)
            c1b_total = c1b_total and ok
            print(f"  [{'PASS' if ok else 'FAIL'}] Escalon ({px_esc:.1f}) vol ~= "
                  f"{nivel:.1f} (tol +-{tol:.1f}): Iso={v_iso_e:.1f}, Suave={v_su_e:.1f}")
        resultados["escalon_vol_pdp"] = c1b_total

    # C2: Monotonia en banda historica [BK_fin+2, BK_fin+50]
    px_band = np.linspace(bk_fin + 2, bk_fin + 50, 100)
    d_iso_b = iso_model.predict(px_band)
    d_su_b  = suave_model.predict(px_band)
    mono_iso = bool(np.all(np.diff(d_iso_b) >= -1e-9))
    mono_su  = bool(np.all(np.diff(d_su_b)  >= -1e-9))
    resultados["monotonia_iso"]   = mono_iso
    resultados["monotonia_suave"] = mono_su
    print(f"  [{'PASS' if mono_iso else 'FAIL'}] Monotonia Iso en banda BK+2 a BK+50")
    print(f"  [{'PASS' if mono_su else 'FAIL'}] Monotonia Suave en banda BK+2 a BK+50")

    # C3: Asintota superior (delta satura antes del maximo historico)
    px_high = bk_fin + 60
    d_iso_h = float(iso_model.predict([px_high])[0])
    d_su_h  = float(suave_model.predict([px_high])[0])
    c3_iso = d_iso_h <= vol_max_delta * 1.2 if vol_max_delta > 0 else True
    c3_su  = d_su_h  <= vol_max_delta * 1.2 if vol_max_delta > 0 else True
    resultados["asintota_iso"]   = c3_iso
    resultados["asintota_suave"] = c3_su
    print(f"  [{'PASS' if c3_iso else 'FAIL'}] Asintota Iso en Precio+60: "
          f"delta={d_iso_h:.1f} (max_hist={vol_max_delta:.1f})")
    print(f"  [{'PASS' if c3_su else 'FAIL'}] Asintota Suave en Precio+60: delta={d_su_h:.1f}")

    # C4: Divergencia Isotonica vs Suave < 30% en la banda REAL observada
    if pneto_reales is not None and len(pneto_reales) >= 2:
        px_lo, px_hi = float(np.min(pneto_reales)), float(np.max(pneto_reales))
        if px_hi - px_lo < 1.0:
            px_lo, px_hi = px_lo - 1.0, px_hi + 1.0
    else:
        px_lo, px_hi = 40.0, 80.0
    px_mid = np.linspace(px_lo, px_hi, 20)
    d_iso_m = iso_model.predict(px_mid)
    d_su_m  = suave_model.predict(px_mid)
    abs_iso = np.maximum(baseline_latest + d_iso_m, 0)
    abs_su  = np.maximum(baseline_latest + d_su_m,  0)
    denom   = np.maximum(abs_iso, 1.0)
    max_div = float((np.abs(abs_iso - abs_su) / denom).max())
    c4 = max_div < 0.30
    resultados["divergencia_ok"] = c4
    print(f"  [{'PASS' if c4 else 'FAIL'}] Divergencia max Iso/Suave en banda real "
          f"[{px_lo:.0f}-{px_hi:.0f}]: {max_div:.1%}")

    return resultados


def pneto_delta0(modelo) -> float:
    """
    Precio NETO donde el delta del modelo cruza 0 (equivalencia flat ~ deck). Responde
    "¿desde que precio neto el campo no pierde reservas vs baseline?". El Modelo 2 puede
    convertir este precio neto a Brent. NaN si la curva nunca alcanza 0 en [10, 130].
    """
    grid = np.arange(10.0, 130.0, 0.5)
    deltas = modelo.predict(grid)
    idx = np.where(deltas >= 0)[0]
    return float(grid[idx[0]]) if len(idx) > 0 else np.nan


if __name__ == "__main__":
    print("=== 03_modelo.py — Modelo 1: Delta = f(Precio Neto) [Isotonica + Suave] ===\n")

    ruta = STAGING / "tablon_unico.parquet"
    if not ruta.exists():
        raise FileNotFoundError("Ejecutar 01_etl.py y 02_synthetic.py primero.")

    df = pd.read_parquet(ruta)

    # Medianas de descuento por campo (solo para el eje Brent-equivalente de los plots)
    df_real_px = df[(~df["ES_SINTETICO"]) & df["BRENT_FLAT_USD_BBL"].notna()]
    medianas_campo = (df_real_px.groupby("CAMPO")[
        ["DESCUENTO_CALIDAD_USD_BBL", "DESCUENTO_TRANSPORTE_USD_BBL"]
    ].median().to_dict(orient="index"))

    # Baselines latest por campo (para reconstruccion de volumen absoluto)
    df_base = df[(df["ESCENARIO"] == "BASE") & df["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
    baselines = (df_base.sort_values("AÑO")
                 .groupby("CAMPO")["VOLUMEN_1P_OFICIAL_MBPE"]
                 .last().to_dict())

    # Baseline PDP por campo (nivel del escalon intermedio de la escalera)
    df_base_pdp = df[(df["ESCENARIO"] == "BASE") & df["VOLUMEN_PDP_MBPE"].notna()]
    baselines_pdp = (df_base_pdp.sort_values("AÑO")
                     .groupby("CAMPO")["VOLUMEN_PDP_MBPE"]
                     .last().to_dict())

    # ── Re-anclaje (2026-06-12; BRENT_REF re-definido 2026-07-09): p_ref = M2(BRENT_REF) ──
    # BRENT_REF = Brent OFICIAL de cierre del ultimo año certificado (68.64 = cierre 2025).
    # El ancla exige Vol(BRENT_REF) = baseline = 1P OFICIAL del ultimo cierre (VOLUMEN_1P_
    # OFICIAL, mismo año). Ese volumen se certifico bajo el deck SEC de ese cierre, cuyo
    # Brent es el oficial del año — NO el Brent del deck de Sensibilidad del quarter
    # siguiente (2026_Q1 = 68.01), que es un pronostico. Usar el Brent del quarter mezclaba
    # precio de vigencia 2026 con volumen de cierre 2025 (mismo tipo de mezcla de vigencias
    # que la normalizacion v2 elimino). round(·,2) ANTES de p_ref: 04 inserta
    # round(BRENT_REF,2) en la grilla — si p_ref usa mas precision la isotonica (step) puede
    # caer en otro escalon y romper la identidad Vol(BRENT_REF)=baseline por un escalon.
    ruta_m2 = STAGING / "correlacion_brent.csv"
    if not ruta_m2.exists():
        raise FileNotFoundError(
            "correlacion_brent.csv no existe: ejecutar 03b_correlacion_brent.py ANTES "
            "de 03_modelo.py (orden del pipeline desde 2026-06-12).")
    # Fila COMPLETA de correlacion_brent.csv por campo: neto_desde_brent respeta la
    # familia M2 adoptada (M2_PARAMS, investigacion 2026-07-15) y cae a ALPHA+BETA
    # lineal cuando no hay adopcion — misma conversion que usara 04_pbi_export.
    import importlib
    _m2 = importlib.import_module("03b_correlacion_brent")
    coef_m2 = {r["CAMPO"]: r.to_dict() for _, r in pd.read_csv(ruta_m2).iterrows()}
    df_cierre = df[df["ES_BASELINE"] & df["BRENT_FLAT_USD_BBL"].notna()
                   & df["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
    if df_cierre.empty:
        raise ValueError("Sin cierres oficiales con Brent en el tablon: no se puede fijar BRENT_REF.")
    anio_cierre = int(df_cierre["AÑO"].max())
    # Brent oficial es unico por año (igual para todos los campos): mediana como guard.
    BRENT_REF = round(float(df_cierre.loc[df_cierre["AÑO"] == anio_cierre,
                                          "BRENT_FLAT_USD_BBL"].median()), 2)
    VIGENCIA_REF = str(anio_cierre)
    p_ref_campo = {c: float(_m2.neto_desde_brent(v, [BRENT_REF])[0])
                   for c, v in coef_m2.items()}
    print(f"Re-anclaje: BRENT_REF={BRENT_REF:.2f} USD/bbl (cierre oficial {VIGENCIA_REF}); "
          f"p_ref por campo via Modelo 2.\n")

    usar_reanclaje = flag("PRED_REANCLAJE_CONFOUND")
    reanclados = []
    if usar_reanclaje:
        print("  [CALIDAD] Re-anclaje selectivo por confound ON "
              "(sesgo>15% & eta2>=0.4 & baseline>=5, directriz 4bis.7.1)\n")

    # Seleccion de metodo por campo (track Calidad): el registro seleccion_metodos.csv
    # gobierna el motor/transform por campo. Cuando esta ON, el registro sustituye la
    # deteccion automatica de confound (registro_sel puede incluir 'reanclaje').
    usar_seleccion = flag("PRED_SELECCION_METODO")
    usar_bandas = flag("PRED_BANDAS_LOYO")
    if usar_bandas:
        print("  [CALIDAD] Bandas de incertidumbre LOYO ON (residuales P10/P90 -> export)\n")
    registro_sel = cargar_registro_seleccion() if usar_seleccion else {}
    registro_mae = {}
    if usar_seleccion:
        _rp = BASE_DIR / "resultados_calidad" / "seleccion_metodos.csv"
        if not _rp.exists() and not _SUF:
            # Flag ratificado y ON por defecto en Produccion (track.py): el registro es
            # un insumo gobernado, no un artefacto derivado. Sin el, la seleccion cae
            # en {} en silencio y los conteos NORTE quedan mal sin aviso.
            raise FileNotFoundError(
                f"PRED_SELECCION_METODO esta ON en Produccion pero falta {_rp}. "
                "El registro de seleccion es un insumo gobernado (ver MAESTRO §10) — "
                "no se puede correr Produccion sin el.")
        if _rp.exists():
            _r = pd.read_csv(_rp)
            # Misma gobernanza por track que cargar_registro_seleccion()
            _estados = {"ADOPTADO"} if not _SUF else {"ADOPTADO", "ADOPTADO_CALIDAD"}
            _r = _r[_r["ESTADO"].isin(_estados)]
            registro_mae = dict(zip(_r["CAMPO"], _r["MAE_LOYO_METODO"]))
        print(f"  [CALIDAD] Seleccion de metodo por campo ON "
              f"({len(registro_sel)} overrides: "
              f"{', '.join(sorted({v['METODO'] for v in registro_sel.values()}))})\n")

    registros = []
    registros_sanity = []   # persistencia G1 (NORTE): CAMPO x CHECK x PASS
    campos = sorted(df["CAMPO"].unique())
    print(f"Campos: {len(campos)}\n")

    for campo in campos:
        print(f"{'=' * 50}\nCampo: {campo}\n{'=' * 50}")

        df_campo = preparar_datos_campo(df, campo)
        n_sint   = int(df_campo["ES_SINTETICO"].sum())
        n_real   = len(df_campo) - n_sint
        baseline_latest = baselines.get(campo, np.nan)

        print(f"  Datos: {len(df_campo)} filas ({n_real} reales delta + {n_sint} sinteticos)")
        print(f"  Baseline latest: {baseline_latest:.2f} MBPE")

        # Deck plano (directriz §4): campo insensible al precio segun su deck REAL.
        # 04_pbi_export usa este flag para servir curva plana (sin rampa isotonica
        # interpolada en la franja sin datos).
        deck_plano = es_deck_plano(df_campo, baseline_latest)
        if deck_plano:
            print(f"  [DECK_PLANO] deck insensible al precio -> curva plana en export")

        if n_real < 2 and n_sint < 2:
            print(f"  [SKIP] Sin datos suficientes (reales={n_real}, sinteticos={n_sint})")
            continue

        df_sint_campo = df_campo[df_campo["ES_SINTETICO"]]
        if len(df_sint_campo) > 0:
            _, w_niveles = pesos_sinteticos_tramo(df_sint_campo, n_real)
        else:
            w_niveles = {}
        w_bajo = w_niveles.get(0.0, np.nan)
        w_esc_vals = [w for nv, w in w_niveles.items() if nv > 0]
        w_esc  = float(np.mean(w_esc_vals)) if w_esc_vals else np.nan
        print(f"  Pesos sinteticos por nivel ({len(w_niveles)} niveles): "
              + ", ".join(f"{nv:.0f}->{w:.3f}" for nv, w in sorted(w_niveles.items()))
              + f"  (n_real={n_real}, n_sint={n_sint})")

        if n_real < 2:
            print(f"  [WARN] Sin datos reales (n={n_real} < 2). "
                  f"Entrenando solo en {n_sint} sinteticos. LOO-CV: N/A.")
            r2i = maei = rmsei = mei = r2s = maes = rmses = mes = mae_naive = np.nan
            skill_i = skill_s = np.nan
            mae_loyo = mae_naive_loyo = skill_loyo = np.nan
            n_vig_loyo = 0
        else:
            print(f"\n  [LOO-CV] Solo sobre {n_real} puntos reales (sinteticos en train)...")
            r2i, maei, rmsei, mei, r2s, maes, rmses, mes, mae_naive = \
                loo_cv_solo_reales(df_campo)
            skill_i = 1 - maei / mae_naive if pd.notna(maei) and mae_naive > 0.01 else np.nan
            skill_s = 1 - maes / mae_naive if pd.notna(maes) and mae_naive > 0.01 else np.nan
            print(f"  ISO (primario):  R2_LOO={r2i:.3f}  MAE={maei:.2f}  RMSE={rmsei:.2f}  "
                  f"SKILL={skill_i:.3f}  [REFERENCIAL]")
            print(f"  SUAVE (valid.):  R2_LOO={r2s:.3f}  MAE={maes:.2f}  RMSE={rmses:.2f}  "
                  f"SKILL={skill_s:.3f}  [REFERENCIAL]")
            # LOYO (leave-one-year-out): generalizacion cross-vigencia — la metrica
            # honesta del confound (§3ter). LOYO >> LOO = la curva aprendio el salto
            # entre años, no el precio.
            mae_loyo, mae_naive_loyo, skill_loyo, n_vig_loyo = \
                loyo_cv_por_vigencia(df_campo)
            if pd.notna(mae_loyo):
                infl = mae_loyo / maei if pd.notna(maei) and maei > 0.01 else np.nan
                print(f"  ISO LOYO ({n_vig_loyo} años): MAE={mae_loyo:.2f} "
                      f"(LOO x{infl:.1f})  SKILL_LOYO={skill_loyo:.3f}  [REFERENCIAL]")

        print(f"\n  Entrenando modelos finales (reales + sinteticos)...")
        # registro_sel[campo] es dict {METODO, METODO_VALIDACION} (s12) o ausente
        _sel = registro_sel.get(campo) if usar_seleccion else None
        metodo_campo = _sel["METODO"] if isinstance(_sel, dict) else (
            _sel if _sel else "default")
        metodo_valid = (_sel.get("METODO_VALIDACION") if isinstance(_sel, dict)
                        else None)
        if pd.isna(metodo_valid) if metodo_valid is not None else True:
            metodo_valid = None
        bk_fin_sel, bk_pdp_sel, _ = _anclas_campo(df_campo)
        if metodo_campo == "reanclaje":
            # transform: centrar por vigencia antes de ajustar (mismo mecanismo confound)
            df_campo = centrar_vigencias(df_campo)
            df_sint_campo = df_campo[df_campo["ES_SINTETICO"]]
        if metodo_campo != "default":
            pj_sel = p_junta_campo(df_campo) if str(metodo_campo).startswith("hibrido:") \
                else None
            iso_m, suave_m = entrenar_con_metodo(
                df_campo, metodo_campo, baseline_latest, bk_fin_sel, bk_pdp_sel,
                metodo_validacion=metodo_valid, p_junta=pj_sel)
            print(f"  [CALIDAD][SELECCION] primario={metodo_campo}"
                  + (f"  validacion={metodo_valid}" if metodo_valid else ""))
            # La metrica de record del metodo elegido es el A/B riguroso del benchmark
            if campo in registro_mae and pd.notna(registro_mae[campo]):
                mae_loyo = float(registro_mae[campo])
        else:
            iso_m, suave_m = entrenar_final_campo(df_campo)
            metodo_valid = "Suave"  # default historico

        slug = campo.replace(" ", "_")
        joblib.dump(iso_m,   MODELOS_DIR / f"{slug}_iso.joblib")
        joblib.dump(suave_m, MODELOS_DIR / f"{slug}_suave.joblib")

        # ── Re-anclaje: delta_ref = f(p_ref) por motor ────────────────────────
        p_ref = p_ref_campo.get(campo)
        if p_ref is None:
            print(f"  [WARN] {campo} sin coeficientes M2: re-anclaje con delta_ref=0")
            delta_ref_iso = delta_ref_su = 0.0
        else:
            delta_ref_iso = float(iso_m.predict([p_ref])[0])
            delta_ref_su  = float(suave_m.predict([p_ref])[0])
            print(f"  Re-anclaje: p_ref={p_ref:.2f}  delta_ref ISO={delta_ref_iso:.2f}  "
                  f"SUAVE={delta_ref_su:.2f}")

        # ── Re-anclaje selectivo por confound (track Calidad, directriz §4bis.7.1) ──
        # Cuando la SELECCION por campo esta ON, el registro gobierna (puede incluir
        # 'reanclaje'); se desactiva la deteccion automatica para no aplicarla dos veces.
        reancl = (metodo_campo == "reanclaje")
        sesgo0 = (abs(delta_ref_iso) / baseline_latest
                  if pd.notna(baseline_latest) and baseline_latest > 0 else np.nan)
        eta2_c = eta2_vigencia(df_campo)
        if (usar_reanclaje and not usar_seleccion
                and pd.notna(baseline_latest) and baseline_latest >= REANCL_BASE_MIN
                and pd.notna(eta2_c) and eta2_c >= REANCL_ETA2_MIN
                and pd.notna(sesgo0) and sesgo0 > REANCL_SESGO_MIN):
            # Centrar por vigencia y RE-ENTRENAR: la curva pierde el escalon de deck y
            # conserva solo la pendiente intra-vigencia (elasticidad real ≈ 0 aqui).
            df_campo = centrar_vigencias(df_campo)
            df_sint_campo = df_campo[df_campo["ES_SINTETICO"]]
            iso_m, suave_m = entrenar_final_campo(df_campo)
            joblib.dump(iso_m,   MODELOS_DIR / f"{slug}_iso.joblib")
            joblib.dump(suave_m, MODELOS_DIR / f"{slug}_suave.joblib")
            if n_real >= 2:
                r2i, maei, rmsei, mei, r2s, maes, rmses, mes, mae_naive = \
                    loo_cv_solo_reales(df_campo)
                skill_i = 1 - maei / mae_naive if pd.notna(maei) and mae_naive > 0.01 else np.nan
                skill_s = 1 - maes / mae_naive if pd.notna(maes) and mae_naive > 0.01 else np.nan
                mae_loyo, mae_naive_loyo, skill_loyo, n_vig_loyo = \
                    loyo_cv_por_vigencia(df_campo)
            if p_ref is not None:
                delta_ref_iso = float(iso_m.predict([p_ref])[0])
                delta_ref_su  = float(suave_m.predict([p_ref])[0])
            sesgo1 = (abs(delta_ref_iso) / baseline_latest if baseline_latest > 0 else np.nan)
            reancl = True
            reanclados.append({"CAMPO": campo, "ETA2": round(eta2_c, 3),
                               "SESGO_ANTES": round(sesgo0, 3),
                               "SESGO_DESPUES": round(sesgo1, 3),
                               "BASELINE": round(baseline_latest, 1),
                               "MAE_LOYO": round(mae_loyo, 2) if pd.notna(mae_loyo) else None})
            print(f"  [CALIDAD][REANCLADO_CONFOUND] eta2={eta2_c:.2f}  "
                  f"sesgo {sesgo0:.0%}->{sesgo1:.0%}  MAE_LOYO={mae_loyo:.2f}"
                  if pd.notna(mae_loyo) else
                  f"  [CALIDAD][REANCLADO_CONFOUND] eta2={eta2_c:.2f}  sesgo {sesgo0:.0%}->{sesgo1:.0%}")

        # ── Sanity checks ─────────────────────────────────────────────────────
        bk_fin, bk_pdp, vbk_campo = _anclas_campo(df_campo)
        baseline_pdp = float(baselines_pdp.get(campo, 0.0) or 0.0)
        medianas = dict(medianas_campo.get(campo, {
            "DESCUENTO_CALIDAD_USD_BBL": -7.0,
            "DESCUENTO_TRANSPORTE_USD_BBL": -3.3,
        }))
        if bk_fin is None:
            print(f"\n  [WARN] Sin breakeven disponible para {campo}: sanity check omitido.")
        else:
            vol_max_delta = float(df_campo.loc[~df_campo["ES_SINTETICO"], TARGET].max()) \
                if n_real > 0 else 0.0
            pneto_reales = df_campo.loc[~df_campo["ES_SINTETICO"], FEATURE].values
            print(f"\n  Sanity checks (BK_fin={bk_fin:.1f}, BK_pdp={bk_pdp:.1f}, "
                  f"baseline={baseline_latest:.1f}, PDP={baseline_pdp:.1f}):")
            checks = sanity_check(campo, iso_m, suave_m, bk_fin, bk_pdp,
                                  vol_max_delta, baseline_latest, baseline_pdp,
                                  df_sint=df_sint_campo, pneto_reales=pneto_reales)
            # Persistencia G1 (NORTE, deuda cerrada 2026-07-02): sin este artefacto
            # test_norte::test_g1 hacia skip y los gates fisicos no bloqueaban en CI
            for chk, ok in checks.items():
                registros_sanity.append({"CAMPO": campo, "CHECK": chk, "PASS": bool(ok)})

        # C5/C6 — invariantes del re-anclaje (sobre la reconstruccion anclada)
        # H1 (auditoria 2026-07-07): con piso_efectivo() el ancla se cumple SIEMPRE
        # (ya no existe la excepcion p_ref < bk_pdp: ese BK esta falsificado por la
        # certificacion y se capa). C6 se evalua bajo el piso EFECTIVO.
        if p_ref is not None and pd.notna(baseline_latest):
            _bk = bk_pdp if bk_fin is not None else None
            v_ref = float(volumen_anclado(iso_m, [p_ref], baseline_latest,
                                          delta_ref_iso, _bk, p_ref=p_ref)[0])
            c5 = abs(v_ref - max(baseline_latest, 0)) < 1e-6
            print(f"  C5 Vol(p_ref)=baseline: {'PASS' if c5 else 'FAIL'} "
                  f"(vol={v_ref:.2f} vs baseline={baseline_latest:.2f})")
            _bk_eff = piso_efectivo(_bk, p_ref, baseline_latest)
            if _bk_eff is not None:
                v0 = float(volumen_anclado(iso_m, [_bk_eff - 5.0], baseline_latest,
                                           delta_ref_iso, _bk, p_ref=p_ref)[0])
                print(f"  C6 Vol(piso_efectivo-5)=0: {'PASS' if v0 == 0.0 else 'FAIL'} "
                      f"(vol={v0:.2f})")

        # ── Plot ──────────────────────────────────────────────────────────────
        print()
        # Junta visible en el plot solo para campos con hibrido adoptado
        _pj_plot = (p_junta_campo(df_campo)
                    if metodo_campo.startswith("hibrido:") else None)
        generar_plot(campo, df_campo, iso_m, suave_m, medianas, baseline_latest,
                     baseline_pdp, p_ref=p_ref,
                     delta_ref_iso=delta_ref_iso, delta_ref_su=delta_ref_su,
                     p_junta=_pj_plot)

        # ── Predicciones en el tablon (en Precio Neto) ────────────────────────
        mask_c = df["CAMPO"] == campo
        df_pred = df[mask_c].copy()
        mask_feat = df_pred[FEATURE].notna()

        if mask_feat.any():
            x_in = df_pred.loc[mask_feat, FEATURE].values
            _bk_rec = bk_pdp if bk_fin is not None else None
            df.loc[mask_c & mask_feat, "PRED_ISOTONICA_MBPE"] = volumen_anclado(
                iso_m, x_in, baseline_latest, delta_ref_iso, _bk_rec, p_ref=p_ref)
            df.loc[mask_c & mask_feat, "PRED_SUAVE_MBPE"] = volumen_anclado(
                suave_m, x_in, baseline_latest, delta_ref_su, _bk_rec, p_ref=p_ref)

        mask_vol = mask_c & df["VOLUMEN_1P_OFICIAL_MBPE"].notna()
        df.loc[mask_vol, "DELTA_ISOTONICA_VS_OFICIAL"] = (
            df.loc[mask_vol, "PRED_ISOTONICA_MBPE"] - df.loc[mask_vol, "VOLUMEN_1P_OFICIAL_MBPE"])
        df.loc[mask_vol, "DELTA_SUAVE_VS_OFICIAL"] = (
            df.loc[mask_vol, "PRED_SUAVE_MBPE"] - df.loc[mask_vol, "VOLUMEN_1P_OFICIAL_MBPE"])

        # Precio Neto donde delta cruza 0 (equivalencia con el deck del baseline)
        pn0_i = pneto_delta0(iso_m)
        pn0_s = pneto_delta0(suave_m)

        registros.append({
            "CAMPO":              campo,
            "N_REAL_DELTA":       n_real,
            "N_SINTETICOS":       n_sint,
            "N_NIVELES_ESCALERA": len(w_niveles),
            "W_SINTETICO":           round(w_bajo, 4) if pd.notna(w_bajo) else None,
            "W_SINTETICO_ESCALERA":  round(w_esc, 4) if pd.notna(w_esc) else None,
            "BASELINE_LATEST":    round(baseline_latest, 2),
            # Re-anclaje (2026-06-12): punto actual donde Vol=baseline por construccion
            "BRENT_REF_USD_BBL":  round(BRENT_REF, 2),
            # 4 decimales: con 2, el shift del re-anclaje deja residuos de 0.01 MBPE
            # visibles en la verificacion Vol(BRENT_REF)=baseline del export
            "P_REF_USD_BBL":      round(p_ref, 4) if p_ref is not None else None,
            "DELTA_REF_ISO":      round(delta_ref_iso, 4),
            "DELTA_REF_SUAVE":    round(delta_ref_su, 4),
            # Motor PRIMARIO: Isotonica
            "R2_LOO_ISO":         round(r2i,  4) if pd.notna(r2i)  else None,
            "MAE_LOO_ISO":        round(maei, 2) if pd.notna(maei) else None,
            "RMSE_LOO_ISO":       round(rmsei,2) if pd.notna(rmsei) else None,
            "ME_ISO":             round(mei,  2) if pd.notna(mei)  else None,
            "MAE_REL_ISO":        round(maei / baseline_latest, 4)
                                  if pd.notna(maei) and baseline_latest > 0 else None,
            "SKILL_ISO":          round(skill_i, 4) if pd.notna(skill_i) else None,
            # Motor VALIDACION: Suave
            "R2_LOO_SUAVE":       round(r2s,  4) if pd.notna(r2s)  else None,
            "MAE_LOO_SUAVE":      round(maes, 2) if pd.notna(maes) else None,
            "RMSE_LOO_SUAVE":     round(rmses,2) if pd.notna(rmses) else None,
            "ME_SUAVE":           round(mes,  2) if pd.notna(mes)  else None,
            "MAE_REL_SUAVE":      round(maes / baseline_latest, 4)
                                  if pd.notna(maes) and baseline_latest > 0 else None,
            "SKILL_SUAVE":        round(skill_s, 4) if pd.notna(skill_s) else None,
            "MAE_NAIVE":          round(mae_naive, 2) if pd.notna(mae_naive) else None,
            # LOYO (leave-one-year-out, auditoria NORTE 2026-07-02): error de
            # generalizacion CROSS-VIGENCIA — la pregunta real del negocio (predecir
            # el año no visto). LOYO >> LOO delata el confound de vigencia (§3ter).
            "MAE_LOYO_ISO":       round(mae_loyo, 2) if pd.notna(mae_loyo) else None,
            "MAE_NAIVE_LOYO":     round(mae_naive_loyo, 2) if pd.notna(mae_naive_loyo) else None,
            "SKILL_LOYO_ISO":     round(skill_loyo, 4) if pd.notna(skill_loyo) else None,
            "N_VIGENCIAS_LOYO":   n_vig_loyo,
            # Precio Neto donde delta=0 (equivale al deck del baseline)
            "PNETO_DELTA0_ISO":   round(pn0_i, 1) if pd.notna(pn0_i) else None,
            "PNETO_DELTA0_SUAVE": round(pn0_s, 1) if pd.notna(pn0_s) else None,
            # Ratio RMSE/MAE del primario: >2 indica fold LOO dominado por outlier
            "OUTLIER_RATIO_ISO":      round(rmsei / maei, 3)
                                      if pd.notna(maei) and maei > 0 else None,
            "ALERTA_LOO_OUTLIER_ISO": bool(rmsei / maei > 2.0)
                                      if pd.notna(maei) and maei > 0 else False,
            # Deck plano (directriz §4): 04_pbi_export sirve curva plana para estos campos
            "DECK_PLANO":         bool(deck_plano),
            # Re-anclaje selectivo por confound (track Calidad, directriz §4bis.7.1)
            "REANCLADO_CONFOUND": bool(reancl),
            # Metodo seleccionado por campo (track Calidad, PRED_SELECCION_METODO)
            "METODO_SELECCIONADO": metodo_campo,
            # s12: primario + validacion dinamica (2.o mejor LOYO); default = Suave
            "METODO_PRIMARIO":     metodo_campo if metodo_campo != "default" else "Isotonica",
            "METODO_VALIDACION":   metodo_valid if metodo_valid else "Suave",
        })

        # Divergencia primario vs validacion en banda real (s12; sanity C4 generalizado)
        _pnet_r = df_campo.loc[~df_campo["ES_SINTETICO"], FEATURE].values
        if len(_pnet_r) >= 2 and pd.notna(baseline_latest):
            import seleccion_reglas as _R
            registros[-1]["DIVERGENCIA_PRIMARIO_VALID_PCT"] = round(
                100.0 * _R.divergencia_volumen(iso_m, suave_m, baseline_latest, _pnet_r),
                2)

        # Bandas de incertidumbre LOYO (track Calidad, s9): cuantiles empiricos de los
        # residuales firmados sobre el df_campo VIGENTE (post-centrado si aplico) —
        # la banda refleja el tratamiento realmente usado por la curva final.
        # Con hibrido adoptado (2026-07-15) la banda se recalcula CON el hibrido.
        if usar_bandas:
            if metodo_campo.startswith("hibrido:"):
                _nombre_sup = metodo_campo.split(":", 1)[1]
                _pj = p_junta_campo(df_campo)
                if _nombre_sup in CANDIDATOS_HIBRIDO and pd.notna(_pj):
                    _factory_bandas = (lambda f=CANDIDATOS_HIBRIDO[_nombre_sup],
                                       pj=_pj: MotorHibrido(f, pj))
                else:
                    _factory_bandas = MotorIsotonico
            else:
                _factory_bandas = MotorIsotonico
            _res = residuales_loyo(df_campo, factory=_factory_bandas)
            if len(_res) >= 2:
                # P25/P75 (s13): con N~3 vigencias los cuantiles 10/90 degeneran en
                # min/max de los residuales — cuartiles interiores son defendibles.
                registros[-1]["LOYO_RESID_P25"] = round(float(np.quantile(_res, 0.25)), 4)
                registros[-1]["LOYO_RESID_P75"] = round(float(np.quantile(_res, 0.75)), 4)
        print()

    # ── Metricas ──────────────────────────────────────────────────────────────
    df_met = pd.DataFrame(registros)
    df_met.to_csv(STAGING / "metricas.csv", index=False, encoding="utf-8-sig")
    df_met.to_csv(RESULTADOS / "metricas.csv", index=False, encoding="utf-8-sig")
    print(f"\n  Matriz de metricas: {RESULTADOS / 'metricas.csv'}")

    if usar_reanclaje:
        if reanclados:
            df_re = pd.DataFrame(reanclados)
            df_re.to_csv(RESULTADOS / "reanclaje_confound.csv", index=False, encoding="utf-8-sig")
            print(f"\n  [CALIDAD] Re-anclados por confound ({len(reanclados)}): "
                  f"{RESULTADOS / 'reanclaje_confound.csv'}")
            print(df_re.to_string(index=False))
        else:
            print("\n  [CALIDAD] Ningun campo cumplio el criterio de re-anclaje por confound.")

    # ── Sanity checks persistidos (G1 NORTE) ──────────────────────────────────
    df_sanity = pd.DataFrame(registros_sanity)
    df_sanity.to_csv(STAGING / "sanity_checks.csv", index=False, encoding="utf-8-sig")
    n_fail = int((~df_sanity["PASS"]).sum()) if len(df_sanity) else 0
    print(f"  Sanity checks G1: {STAGING / 'sanity_checks.csv'} "
          f"({len(df_sanity)} checks, {n_fail} FAIL)")
    print(f"\n{'=' * 50}")
    print("Metricas LOO-CV sobre puntos REALES (REFERENCIALES — N pequeño):")
    print(df_met.to_string(index=False))

    # ── Resumen pareto: top-10 por reservas 1P ────────────────────────────────
    from homologacion import Homologador
    _h = Homologador()
    _insens = (df.groupby("CAMPO")["BRENT_INSENSITIVE"]
                 .apply(lambda s: bool(s.dropna().astype(bool).any())))
    _es_filial = df_met["CAMPO"].apply(
        lambda c: str(_h.atributos(c).get("NEW GERENCIA", "")).strip().upper() == "FILIAL")
    df_pareto = df_met[
        (~_es_filial)
        & ~df_met["CAMPO"].map(_insens).fillna(False).astype(bool)
        & (df_met["N_SINTETICOS"] > 0)
    ].nlargest(10, "BASELINE_LATEST").reset_index(drop=True)
    cols_pareto = ["CAMPO", "BASELINE_LATEST", "N_NIVELES_ESCALERA",
                   "R2_LOO_ISO", "MAE_REL_ISO", "MAE_REL_SUAVE",
                   "SKILL_ISO", "SKILL_SUAVE",
                   "PNETO_DELTA0_ISO", "PNETO_DELTA0_SUAVE"]
    df_pareto[cols_pareto].to_csv(STAGING / "pareto_top10.csv",
                                  index=False, encoding="utf-8-sig")
    print(f"\n{'=' * 50}")
    print("PARETO top-10 por reservas 1P (sin filiales / brent-insensitive / sin BK):")
    print(df_pareto[cols_pareto].to_string(index=False))

    # ── Tramo de vigilancia (pareto 10-20) — gobierno de calidad, SIN gate duro ───
    # Artifact 8a8f2cbc Nivel 3 (2026-07-10): la materialidad media (10-30 MBPE) es
    # donde el modelo es mas debil y el gate-9 lo oculta. Se listan aparte con la
    # metrica de sesgo de recuperacion (|d_ref|/baseline) y los errores LOO/LOYO para
    # auditoria. NO se convierten en gate duro: NORTE (test_norte.py) no los evalua,
    # este CSV es solo reporte de vigilancia (candidatos: NARE, LISAMA, RECETOR...).
    df_rank = df_met[
        (~_es_filial)
        & ~df_met["CAMPO"].map(_insens).fillna(False).astype(bool)
        & (df_met["N_SINTETICOS"] > 0)
    ].nlargest(20, "BASELINE_LATEST").reset_index(drop=True)
    df_rank["RANK_PARETO"] = df_rank.index + 1
    df_rank["SESGO_RECUP"] = (df_rank["DELTA_REF_ISO"].abs() /
                              df_rank["BASELINE_LATEST"].replace(0, np.nan)).round(4)
    df_vigilancia = df_rank[df_rank["RANK_PARETO"] > 10].copy()
    cols_vig = ["RANK_PARETO", "CAMPO", "BASELINE_LATEST", "SESGO_RECUP",
                "MAE_REL_ISO", "SKILL_ISO", "MAE_LOYO_ISO", "SKILL_LOYO_ISO",
                "PNETO_DELTA0_ISO"]
    df_vigilancia[cols_vig].to_csv(STAGING / "tramo_vigilancia.csv",
                                   index=False, encoding="utf-8-sig")
    print(f"\n{'=' * 50}")
    print("TRAMO DE VIGILANCIA (pareto 10-20) — reporte de calidad, SIN gate duro:")
    print(df_vigilancia[cols_vig].to_string(index=False))

    # ── Guardar tablon con predicciones ──────────────────────────────────────
    df.to_parquet(STAGING / "tablon_unico.parquet", index=False)
    df.to_csv(STAGING / "tablon_unico.csv", index=False, encoding="utf-8-sig")
    print("\nTablon actualizado con predicciones.")
    print("\n=== 03_modelo.py — Completado ===")
