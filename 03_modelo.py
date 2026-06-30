"""
03_modelo.py — Modelo 1: Delta Reservas 1P = f(Precio Neto) por campo

ARQUITECTURA DE 2 MODELOS (directriz 2026-06-11):
  Modelo 1 (este script): PRECIO_NETO -> DELTA_SENS_MBPE. La fisica de la sensibilidad
    de reservas vive en el precio realizado NETO (lo que el motor de FC certifica), no en
    el Brent. Es un problema 1D.
  Modelo 2 (03b_correlacion_brent.py): BRENT -> PRECIO_NETO por campo. Convierte el
    marcador Brent al precio neto que alimenta el Modelo 1. Permite Brent -> Delta por
    composicion en 04_pbi_export.py.

TARGET: DELTA_SENS_MBPE = ΔReservas = Sensibilidad − Baseline_vigencia.

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

from motores_modelo1 import MotorIsotonico, MotorSuave, volumen_anclado

BASE_DIR    = Path(__file__).parent
STAGING     = BASE_DIR / "datos" / "staging"
MODELOS_DIR = STAGING / "modelos"
PLOTS_DIR   = STAGING / "plots"
RESULTADOS  = BASE_DIR / "resultados"
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
                 delta_ref_su: float = 0.0) -> None:
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

    # ── Re-anclaje (2026-06-12): p_ref por campo = M2(BRENT_REF) ─────────────
    # BRENT_REF = Brent del ultimo quarter conocido del Consolidado (punto actual).
    ruta_m2 = STAGING / "correlacion_brent.csv"
    if not ruta_m2.exists():
        raise FileNotFoundError(
            "correlacion_brent.csv no existe: ejecutar 03b_correlacion_brent.py ANTES "
            "de 03_modelo.py (orden del pipeline desde 2026-06-12).")
    coef_m2 = pd.read_csv(ruta_m2).set_index("CAMPO")[["ALPHA", "BETA"]].to_dict("index")
    df_q = df[(~df["ES_SINTETICO"]) & (~df["ES_BASELINE"])
              & df["BRENT_FLAT_USD_BBL"].notna() & df["VIGENCIA"].notna()]
    if df_q.empty:
        raise ValueError("Sin quarters Consolidado en el tablon: no se puede fijar BRENT_REF.")
    # VIGENCIA "YYYY_Qn" ordena lexicograficamente; el ultimo quarter es el punto actual
    BRENT_REF = float(df_q.sort_values("VIGENCIA")["BRENT_FLAT_USD_BBL"].iloc[-1])
    VIGENCIA_REF = str(df_q["VIGENCIA"].max())
    p_ref_campo = {c: v["ALPHA"] + v["BETA"] * BRENT_REF for c, v in coef_m2.items()}
    print(f"Re-anclaje: BRENT_REF={BRENT_REF:.2f} USD/bbl (quarter {VIGENCIA_REF}); "
          f"p_ref por campo via Modelo 2.\n")

    registros = []
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

        print(f"\n  Entrenando modelos finales (reales + sinteticos)...")
        iso_m, suave_m = entrenar_final_campo(df_campo)

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
            sanity_check(campo, iso_m, suave_m, bk_fin, bk_pdp,
                         vol_max_delta, baseline_latest, baseline_pdp,
                         df_sint=df_sint_campo, pneto_reales=pneto_reales)

        # C5/C6 — invariantes del re-anclaje (sobre la reconstruccion anclada)
        if p_ref is not None and pd.notna(baseline_latest):
            _bk = bk_pdp if bk_fin is not None else None
            v_ref = float(volumen_anclado(iso_m, [p_ref], baseline_latest,
                                          delta_ref_iso, _bk)[0])
            c5 = abs(v_ref - max(baseline_latest, 0)) < 1e-6 or \
                 (_bk is not None and p_ref < _bk)
            print(f"  C5 Vol(p_ref)=baseline: {'PASS' if c5 else 'FAIL'} "
                  f"(vol={v_ref:.2f} vs baseline={baseline_latest:.2f})")
            if _bk is not None:
                v0 = float(volumen_anclado(iso_m, [_bk - 5.0], baseline_latest,
                                           delta_ref_iso, _bk)[0])
                print(f"  C6 Vol(bk_pdp-5)=0: {'PASS' if v0 == 0.0 else 'FAIL'} "
                      f"(vol={v0:.2f})")

        # ── Plot ──────────────────────────────────────────────────────────────
        print()
        generar_plot(campo, df_campo, iso_m, suave_m, medianas, baseline_latest,
                     baseline_pdp, p_ref=p_ref,
                     delta_ref_iso=delta_ref_iso, delta_ref_su=delta_ref_su)

        # ── Predicciones en el tablon (en Precio Neto) ────────────────────────
        mask_c = df["CAMPO"] == campo
        df_pred = df[mask_c].copy()
        mask_feat = df_pred[FEATURE].notna()

        if mask_feat.any():
            x_in = df_pred.loc[mask_feat, FEATURE].values
            _bk_rec = bk_pdp if bk_fin is not None else None
            df.loc[mask_c & mask_feat, "PRED_ISOTONICA_MBPE"] = volumen_anclado(
                iso_m, x_in, baseline_latest, delta_ref_iso, _bk_rec)
            df.loc[mask_c & mask_feat, "PRED_SUAVE_MBPE"] = volumen_anclado(
                suave_m, x_in, baseline_latest, delta_ref_su, _bk_rec)

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
            # Precio Neto donde delta=0 (equivale al deck del baseline)
            "PNETO_DELTA0_ISO":   round(pn0_i, 1) if pd.notna(pn0_i) else None,
            "PNETO_DELTA0_SUAVE": round(pn0_s, 1) if pd.notna(pn0_s) else None,
            # Ratio RMSE/MAE del primario: >2 indica fold LOO dominado por outlier
            "OUTLIER_RATIO_ISO":      round(rmsei / maei, 3)
                                      if pd.notna(maei) and maei > 0 else None,
            "ALERTA_LOO_OUTLIER_ISO": bool(rmsei / maei > 2.0)
                                      if pd.notna(maei) and maei > 0 else False,
        })
        print()

    # ── Metricas ──────────────────────────────────────────────────────────────
    df_met = pd.DataFrame(registros)
    df_met.to_csv(STAGING / "metricas.csv", index=False, encoding="utf-8-sig")
    df_met.to_csv(RESULTADOS / "metricas.csv", index=False, encoding="utf-8-sig")
    print(f"\n  Matriz de metricas: {RESULTADOS / 'metricas.csv'}")
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

    # ── Guardar tablon con predicciones ──────────────────────────────────────
    df.to_parquet(STAGING / "tablon_unico.parquet", index=False)
    df.to_csv(STAGING / "tablon_unico.csv", index=False, encoding="utf-8-sig")
    print("\nTablon actualizado con predicciones.")
    print("\n=== 03_modelo.py — Completado ===")
