"""
03_modelo.py — Entrenamiento XGBoost (primario) + Isotonica (validacion) por campo

TARGET: DELTA_SENS_MBPE = ΔReservas = Sensibilidad − Baseline_vigencia.
Predicion final: VOLUMEN_1P_PRED = BASELINE_LATEST + DELTA_PRED.

Los modelos aprenden la FORMA de la sensibilidad a precio (pendiente), no el nivel absoluto.
Esto aisla el efecto precio de los saltos de nivel inter-vigencia (deplecion, regalias 2026).

Correcciones 2026-06-04:
  B1: LOO-CV calculado SOLO sobre puntos REALES. Los sinteticos van siempre al train
      (son el ancla fisica del modelo). Metricas separadas: real vs sintetico.
  Sanity: monotonia verificada en la banda COMPLETA de precios observados (no una slice).
  Asintota: criterio 3 implementado (vol_max ahora se usa).
  Divergencia: verificada en toda la banda historica ($40-$80), no en un punto.

Ver docs/MAESTRO.md §7 y §9 para justificacion del motor.
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

BASE_DIR    = Path(__file__).parent
STAGING     = BASE_DIR / "datos" / "staging"
MODELOS_DIR = STAGING / "modelos"
PLOTS_DIR   = STAGING / "plots"
RESULTADOS  = BASE_DIR / "resultados"
MODELOS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTADOS.mkdir(parents=True, exist_ok=True)

# ── Hiperparametros XGBoost ────────────────────────────────────────────────────
# Con N~8-30 puntos por campo, el modelo anterior (300 árboles, profundidad 3)
# sobreajustaba. Se reduce capacidad y se añade regularización.
XGB_PARAMS = dict(
    n_estimators=100,           # era 300; reducido para N pequeño, suficiente para anclar el piso
    max_depth=2,                # era 3; profundidad 2 = 3 hojas/árbol, menos step-functions
    min_child_weight=3,         # era 5; algo menos estricto con N reducido
    learning_rate=0.05,
    reg_lambda=2.0,             # L2: penaliza pesos extremos, suaviza curva
    gamma=0.1,                  # poda: split solo si reduce loss ≥ 0.1
    monotone_constraints=(1, -1, -1),   # BRENT↑→Delta↑, Calidad/Transporte↑→Delta↓
    objective="reg:squarederror",
    random_state=42,
    verbosity=0,
)

# Peso sintetico DINAMICO por campo (anti-sesgo): masa total sintetica ≈ masa
# total real, w_sint = n_real / max(n_sint, 1), con piso minimo para no anular
# el ancla fisica en campos con muchos sinteticos. Antes era 1.0 fijo, lo que
# dejaba a los sinteticos (hasta ~9:1 sobre reales) dominar el entrenamiento y
# sesgar la prediccion hacia abajo (ME<0 generalizado).
W_SINTETICO_MIN = 0.05


def peso_sintetico(n_real: int, n_sint: int) -> float:
    """Peso por sintetico para que su masa total ≈ la masa total de puntos reales."""
    if n_sint == 0:
        return 1.0
    return max(n_real / n_sint, W_SINTETICO_MIN)


def pesos_sinteticos_tramo(df_sint: pd.DataFrame, n_real: int) -> np.ndarray:
    """
    Peso por fila sintetica, calculado POR TRAMO (BAJO vs ESCALERA) por separado:
    cada tramo balancea su masa total contra n_real de forma independiente.

    Sin esto, un campo con ESCALERA grande (muchos puntos vol=BASELINE_PDP) diluye
    el ancla critica BAJO (vol=0, RANGO_USD=5 → solo 5 puntos) cuando ambos
    comparten un unico peso global — el modelo deja de anclar vol≈0 sub-breakeven
    (test_sub_breakeven_vol_cero fallaba en CASTILLA/CASTILLA NORTE/RUBIALES).
    """
    es_bajo = (df_sint["VOLUMEN_1P_SENSIBILIDAD_MBPE"].values == 0)
    n_bajo  = int(es_bajo.sum())
    n_esc   = len(df_sint) - n_bajo
    w_bajo  = peso_sintetico(n_real, n_bajo)
    w_esc   = peso_sintetico(n_real, n_esc)
    return np.where(es_bajo, w_bajo, w_esc), w_bajo, w_esc

FEATURES_XGB = ["BRENT_FLAT_USD_BBL",
                "DESCUENTO_CALIDAD_USD_BBL",
                "DESCUENTO_TRANSPORTE_USD_BBL"]
FEATURE_ISO  = "PRECIO_NETO_USD_BBL"
TARGET       = "DELTA_SENS_MBPE"   # espacio delta — target del modelo


# ── Helpers ────────────────────────────────────────────────────────────────────

def preparar_datos_campo(df: pd.DataFrame, campo: str) -> pd.DataFrame:
    """
    Filtra datos de entrenamiento para un campo.

    Incluye:
      - Puntos sinteticos (ES_SINTETICO=True): DELTA = -BASELINE (ancla fisica)
      - Puntos CONSOLIDADO con DELTA disponible (precio fijo, vigencia fija)

    Excluye:
      - Filas BASE (DELTA=NaN por diseño: no hay punto de sensibilidad)
      - Filas CONSOLIDADO sin target (TARGET_NULO: precios sin reservas en esa Q)

    Esta seleccion garantiza que el modelo aprende la FORMA de la sensibilidad a precio,
    no la tendencia temporal de deplecion/revisiones/saltos de nivel (regalias 2026).
    """
    sub = df[df["CAMPO"] == campo].copy()

    # Solo filas con TARGET disponible y features completas
    mask = (
        sub[TARGET].notna() &
        sub[FEATURE_ISO].notna() &
        sub[FEATURES_XGB].notna().all(axis=1)
    )
    return sub[mask].reset_index(drop=True)


def loo_cv_solo_reales(df_campo: pd.DataFrame) -> tuple:
    """
    B1: LOO-CV solo sobre puntos REALES. Sinteticos siempre van al train.

    Para cada punto real i:
      train = todos los sinteticos + todos los reales excepto i
      test  = real[i]

    Retorna (r2_xgb, mae_xgb, rmse_xgb, r2_iso, mae_iso, rmse_iso) sobre puntos reales.
    Garantiza que la metrica refleja el ajuste a datos observados, no el trivial ajuste
    a los ceros sinteticos (predecir 0 en vol=0 no aporta informacion de la curva real).

    Pesos sinteticos: ver pesos_sinteticos_tramo (BAJO/ESCALERA balanceados por separado
    contra n_real), para que ningun tramo domine sobre los puntos reales del fold.
    """
    df_real = df_campo[~df_campo["ES_SINTETICO"]].reset_index(drop=True)
    df_sint = df_campo[df_campo["ES_SINTETICO"]].reset_index(drop=True)

    n_real = len(df_real)
    if n_real < 2:
        nan6 = (np.nan,) * 6
        return nan6

    X_sint  = df_sint[FEATURES_XGB].values  if len(df_sint) > 0 else np.empty((0, 3))
    y_sint  = df_sint[TARGET].values         if len(df_sint) > 0 else np.empty(0)
    Xi_sint = df_sint[FEATURE_ISO].values    if len(df_sint) > 0 else np.empty(0)
    # Pesos: sintéticos anclan el piso por tramo (BAJO/ESCALERA), reales aprenden la curva (1.0)
    w_sint  = pesos_sinteticos_tramo(df_sint, n_real)[0] if len(df_sint) > 0 else np.empty(0)

    y_pred_xgb = np.zeros(n_real, dtype=float)
    y_pred_iso = np.zeros(n_real, dtype=float)

    for i in range(n_real):
        mask_tr = np.ones(n_real, dtype=bool)
        mask_tr[i] = False

        X_tr_real  = df_real.loc[mask_tr, FEATURES_XGB].values
        y_tr_real  = df_real.loc[mask_tr, TARGET].values
        Xi_tr_real = df_real.loc[mask_tr, FEATURE_ISO].values

        X_train  = np.vstack([X_sint, X_tr_real])   if len(X_sint) > 0 else X_tr_real
        y_train  = np.concatenate([y_sint, y_tr_real]) if len(y_sint) > 0 else y_tr_real
        Xi_train = np.concatenate([Xi_sint, Xi_tr_real]) if len(Xi_sint) > 0 else Xi_tr_real
        w_train  = np.concatenate([w_sint, np.ones(mask_tr.sum())]) if len(w_sint) > 0 else np.ones(mask_tr.sum())

        # XGBoost
        m_xgb = xgb.XGBRegressor(**XGB_PARAMS)
        m_xgb.fit(X_train, y_train, sample_weight=w_train)
        y_pred_xgb[i] = float(m_xgb.predict(
            df_real.loc[[i], FEATURES_XGB].values)[0])

        # IsotonicRegression 1D
        m_iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        m_iso.fit(Xi_train, y_train, sample_weight=w_train)
        y_pred_iso[i] = float(m_iso.predict(
            [df_real.loc[i, FEATURE_ISO]])[0])

    y_true = df_real[TARGET].values

    def _metricas(y_t, y_p):
        r2   = r2_score(y_t, y_p) if len(y_t) > 1 else np.nan
        mae  = mean_absolute_error(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        # Bias (Mean Error): >0 = modelo sobreestima, <0 = subestima
        me   = float(np.mean(y_p - y_t))
        return r2, mae, rmse, me

    r2x, maex, rmsex, mex = _metricas(y_true, y_pred_xgb)
    r2i, maei, rmsei, mei = _metricas(y_true, y_pred_iso)
    return r2x, maex, rmsex, mex, r2i, maei, rmsei, mei


def entrenar_final_campo(df_campo: pd.DataFrame, w_bajo: float, w_esc: float) -> tuple:
    """Entrena modelos finales con TODOS los datos (reales + sinteticos)."""
    X_xgb = df_campo[FEATURES_XGB].values
    X_iso = df_campo[FEATURE_ISO].values
    y     = df_campo[TARGET].values
    # Pesos alineados al orden del DataFrame: BAJO/ESCALERA balanceados por separado contra reales
    es_sint = df_campo["ES_SINTETICO"].values
    es_bajo = df_campo["VOLUMEN_1P_SENSIBILIDAD_MBPE"].values == 0
    pesos = np.ones(len(df_campo))
    pesos[es_sint & es_bajo]  = w_bajo
    pesos[es_sint & ~es_bajo] = w_esc

    m_xgb = xgb.XGBRegressor(**XGB_PARAMS)
    m_xgb.fit(X_xgb, y, sample_weight=pesos)

    m_iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    m_iso.fit(X_iso, y, sample_weight=pesos)
    return m_xgb, m_iso


def generar_plot(campo: str, df_campo: pd.DataFrame,
                 xgb_model, iso_model,
                 medianas_px: dict, baseline_latest: float) -> None:
    """
    Plot de validacion visual. Eje Y = espacio delta (ΔReservas vs precio).
    Eje Y derecho = volumen absoluto reconstruido (baseline + delta).
    """
    bk_arr = df_campo["BREAKEVEN_FINANCIERO_USD_BBL"].dropna().values
    bk_fin = float(bk_arr[0]) if len(bk_arr) > 0 else None
    px_min = df_campo[FEATURE_ISO].min() - 5
    px_max = df_campo[FEATURE_ISO].max() + 10
    px_grid = np.linspace(px_min, px_max, 200)

    med_cal = medianas_px.get("DESCUENTO_CALIDAD_USD_BBL", -7.0)
    med_tra = medianas_px.get("DESCUENTO_TRANSPORTE_USD_BBL", -3.3)
    brent_grid = px_grid - med_cal - med_tra

    X_grid_xgb = np.column_stack([
        brent_grid,
        np.full_like(px_grid, med_cal),
        np.full_like(px_grid, med_tra),
    ])
    curva_xgb_delta = xgb_model.predict(X_grid_xgb)
    curva_iso_delta = iso_model.predict(px_grid)

    # Reconstruccion absoluta: baseline + delta
    curva_xgb_abs = np.maximum(baseline_latest + curva_xgb_delta, 0)
    curva_iso_abs = np.maximum(baseline_latest + curva_iso_delta, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel izquierdo: espacio delta
    reales = df_campo[~df_campo["ES_SINTETICO"]]
    sint   = df_campo[df_campo["ES_SINTETICO"]]

    ax1.scatter(reales[FEATURE_ISO], reales[TARGET],
                color="steelblue", s=70, zorder=5, label="Datos reales (delta)")
    ax1.scatter(sint[FEATURE_ISO], sint[TARGET],
                color="tomato", s=30, alpha=0.6, marker="x", zorder=4,
                label=f"Sinteticos (n={len(sint)})")
    ax1.plot(px_grid, curva_xgb_delta, color="green",
             linewidth=2.0, label="XGBoost Δ")
    ax1.plot(px_grid, curva_iso_delta, color="darkorange",
             linewidth=1.5, linestyle="--", label="Isotonica Δ")
    if bk_fin is not None:
        ax1.axvline(x=bk_fin, color="gray", linestyle=":", linewidth=1.5,
                    label=f"BK_fin={bk_fin:.1f}")
    ax1.axhline(y=0, color="black", linewidth=0.5, alpha=0.4)
    ax1.set_xlabel("Precio Neto (USD/bbl)")
    ax1.set_ylabel("ΔReservas 1P (MBPE)")
    ax1.set_title(f"{campo} — Espacio delta")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel derecho: volumen absoluto reconstruido
    ax2.scatter(reales[FEATURE_ISO],
                baseline_latest + reales[TARGET].fillna(0),
                color="steelblue", s=70, zorder=5, label="Reales (absoluto)")
    ax2.plot(px_grid, curva_xgb_abs, color="green",
             linewidth=2.0, label="XGBoost (absoluto)")
    ax2.plot(px_grid, curva_iso_abs, color="darkorange",
             linewidth=1.5, linestyle="--", label="Isotonica (absoluto)")
    ax2.axhline(y=baseline_latest, color="purple", linewidth=1.0, linestyle=":",
                label=f"Baseline={baseline_latest:.0f}")
    if bk_fin is not None:
        ax2.axvline(x=bk_fin, color="gray", linestyle=":", linewidth=1.5)
    ax2.set_xlabel("Precio Neto (USD/bbl)")
    ax2.set_ylabel("Volumen 1P (MBPE)")
    ax2.set_title(f"{campo} — Volumen absoluto (baseline + delta)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(campo, fontsize=13)
    plt.tight_layout()

    ruta = PLOTS_DIR / f"{campo.replace(' ', '_')}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot guardado: {ruta.name}")


def sanity_check(campo: str, xgb_model, iso_model,
                 bk_fin: float, medianas: dict,
                 vol_max_delta: float, baseline_latest: float) -> dict:
    """
    Verifica los 4 criterios funcionales del piloto.
    Ahora bloqueante en el runner (retorna dict con flags PASS/FAIL).
    C3 (asintota): usa vol_max_delta (corrige bug M3 donde vol_max era ignorado).
    Monotonia: verificada sobre la banda historica completa (40 a 90 USD/bbl), no solo 50pts.
    """
    med_cal = medianas.get("DESCUENTO_CALIDAD_USD_BBL", -7.0)
    med_tra = medianas.get("DESCUENTO_TRANSPORTE_USD_BBL", -3.3)
    resultados = {}

    # C1: sub-breakeven → delta ≈ -BASELINE (reservas nulas)
    px_sub  = bk_fin - 5
    b_sub   = px_sub - med_cal - med_tra
    d_xgb_s = float(xgb_model.predict(np.array([[b_sub, med_cal, med_tra]]))[0])
    d_iso_s = float(iso_model.predict(np.array([px_sub]))[0])
    # En espacio delta, sub-breakeven debe ser ≤ -BASELINE o muy negativo
    c1 = (baseline_latest + d_xgb_s < 5) and (baseline_latest + d_iso_s < 5)
    resultados["sub_breakeven_vol_cero"] = c1
    print(f"  [{'PASS' if c1 else 'FAIL'}] Sub-breakeven vol reconstruido < 5 MBPE: "
          f"XGB={baseline_latest + d_xgb_s:.1f}, Iso={baseline_latest + d_iso_s:.1f}")

    # C2: Monotonia en banda historica completa [BK+2, BK+50]
    px_band = np.linspace(bk_fin + 2, bk_fin + 50, 100)
    b_band  = px_band - med_cal - med_tra
    X_band  = np.column_stack([b_band, np.full(100, med_cal), np.full(100, med_tra)])
    d_xgb_b = xgb_model.predict(X_band)
    d_iso_b = iso_model.predict(px_band)
    mono_xgb = bool(np.all(np.diff(d_xgb_b) >= 0.0))   # tol 0 (no -0.1)
    mono_iso = bool(np.all(np.diff(d_iso_b) >= 0.0))
    resultados["monotonia_xgb"] = mono_xgb
    resultados["monotonia_iso"] = mono_iso
    print(f"  [{'PASS' if mono_xgb else 'FAIL'}] Monotonia XGB en banda BK+2 a BK+50")
    print(f"  [{'PASS' if mono_iso else 'FAIL'}] Monotonia Iso en banda BK+2 a BK+50")

    # C3: Asintota superior visible (delta satura antes del maximo historico)
    px_high = bk_fin + 60
    b_high  = px_high - med_cal - med_tra
    d_xgb_h = float(xgb_model.predict(np.array([[b_high, med_cal, med_tra]]))[0])
    d_iso_h = float(iso_model.predict(np.array([px_high]))[0])
    # Saturation: max delta achieved debe ser <= vol_max_delta * 1.2 (holgura 20%)
    c3_xgb = d_xgb_h <= vol_max_delta * 1.2 if vol_max_delta > 0 else True
    c3_iso = d_iso_h <= vol_max_delta * 1.2 if vol_max_delta > 0 else True
    resultados["asintota_xgb"] = c3_xgb
    resultados["asintota_iso"] = c3_iso
    print(f"  [{'PASS' if c3_xgb else 'FAIL'}] Asintota XGB en Precio+60: "
          f"delta={d_xgb_h:.1f} (max_hist={vol_max_delta:.1f})")
    print(f"  [{'PASS' if c3_iso else 'FAIL'}] Asintota Iso en Precio+60: "
          f"delta={d_iso_h:.1f}")

    # C4: Divergencia XGB vs Iso < 30% en banda $40-$80
    px_mid = np.linspace(max(40 - baseline_latest * 0, 40), 80, 20)
    b_mid  = px_mid - med_cal - med_tra
    X_mid  = np.column_stack([b_mid, np.full(20, med_cal), np.full(20, med_tra)])
    d_xgb_m = xgb_model.predict(X_mid)
    d_iso_m  = iso_model.predict(px_mid)
    # Comparar en espacio absoluto donde los modelos deben concordar
    abs_xgb = np.maximum(baseline_latest + d_xgb_m, 0)
    abs_iso  = np.maximum(baseline_latest + d_iso_m,  0)
    denom    = np.maximum(abs_xgb, 1.0)
    divs     = np.abs(abs_xgb - abs_iso) / denom
    max_div  = float(divs.max())
    c4 = max_div < 0.30
    resultados["divergencia_ok"] = c4
    print(f"  [{'PASS' if c4 else 'FAIL'}] Divergencia max XGB/Iso en $40-$80: {max_div:.1%}")

    return resultados


if __name__ == "__main__":
    print("=== 03_modelo.py — XGBoost + Isotonica (espacio delta) ===\n")

    ruta = STAGING / "tablon_unico.parquet"
    if not ruta.exists():
        raise FileNotFoundError("Ejecutar 01_etl.py y 02_synthetic.py primero.")

    df = pd.read_parquet(ruta)

    # Medianas de descuento por campo (filas reales con precio)
    df_real_px = df[(~df["ES_SINTETICO"]) & df["BRENT_FLAT_USD_BBL"].notna()]
    medianas_campo = (df_real_px.groupby("CAMPO")[
        ["DESCUENTO_CALIDAD_USD_BBL", "DESCUENTO_TRANSPORTE_USD_BBL"]
    ].median().to_dict(orient="index"))

    # Baselines latest por campo (para reconstruccion de volumen absoluto)
    df_base = df[(df["ESCENARIO"] == "BASE") & df["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
    baselines = (df_base.sort_values("AÑO")
                 .groupby("CAMPO")["VOLUMEN_1P_OFICIAL_MBPE"]
                 .last().to_dict())

    registros = []
    campos = sorted(df["CAMPO"].unique())
    print(f"Campos: {campos}\n")

    for campo in campos:
        print(f"{'=' * 50}\nCampo: {campo}\n{'=' * 50}")

        df_campo = preparar_datos_campo(df, campo)
        n_sint   = int(df_campo["ES_SINTETICO"].sum())
        n_real   = len(df_campo) - n_sint
        baseline_latest = baselines.get(campo, np.nan)

        print(f"  Datos: {len(df_campo)} filas ({n_real} reales delta + {n_sint} sinteticos)")
        print(f"  Baseline latest: {baseline_latest:.2f} MBPE")

        if n_real < 2 and n_sint < 2:
            # Sin ningún dato utilizable: skip total
            print(f"  [SKIP] Sin datos suficientes (reales={n_real}, sinteticos={n_sint})")
            continue

        # Peso sintetico dinamico POR TRAMO: BAJO (vol=0) y ESCALERA (vol=PDP)
        # se balancean por separado contra n_real (ver pesos_sinteticos_tramo)
        df_sint_campo = df_campo[df_campo["ES_SINTETICO"]]
        _, w_bajo, w_esc = pesos_sinteticos_tramo(df_sint_campo, n_real)
        print(f"  Peso sintetico BAJO/ESCALERA: {w_bajo:.3f} / {w_esc:.3f} "
              f"(n_real={n_real}, n_sint={n_sint})")

        if n_real < 2:
            # Solo sinteticos disponibles: entrenamiento válido para la zona sub-breakeven,
            # pero el LOO-CV sobre reales no es calculable. El modelo aprenderá la forma
            # del anclaje físico; las métricas quedan NaN hasta tener datos de Consolidado.
            print(f"  [WARN] Sin datos reales (n={n_real} < 2). "
                  f"Entrenando solo en {n_sint} sinteticos. LOO-CV: N/A.")
            r2x = maex = rmsex = mex = r2i = maei = rmsei = mei = np.nan
        else:
            # ── LOO-CV (B1: solo sobre puntos reales) ────────────────────────────
            print(f"\n  [LOO-CV] Solo sobre {n_real} puntos reales (sinteticos siempre en train)...")
            r2x, maex, rmsex, mex, r2i, maei, rmsei, mei = loo_cv_solo_reales(df_campo)
            print(f"  XGB:  R2_LOO={r2x:.3f}  MAE={maex:.2f}  RMSE={rmsex:.2f}  [REFERENCIAL]")
            print(f"  ISO:  R2_LOO={r2i:.3f}  MAE={maei:.2f}  RMSE={rmsei:.2f}  [REFERENCIAL]")

        # ── Modelos finales (todos los datos) ────────────────────────────────
        print(f"\n  Entrenando modelos finales (reales + sinteticos)...")
        xgb_m, iso_m = entrenar_final_campo(df_campo, w_bajo, w_esc)

        joblib.dump(xgb_m, MODELOS_DIR / f"{campo.replace(' ', '_')}_xgb.joblib")
        joblib.dump(iso_m,  MODELOS_DIR / f"{campo.replace(' ', '_')}_iso.joblib")

        # ── Sanity checks ─────────────────────────────────────────────────────
        bk_arr = df_campo["BREAKEVEN_FINANCIERO_USD_BBL"].dropna().values
        if len(bk_arr) == 0:
            print(f"\n  [WARN] Sin breakeven disponible para {campo}: sanity check omitido.")
        else:
            bk_fin = float(bk_arr[0])
            medianas = medianas_campo.get(campo, {
                "DESCUENTO_CALIDAD_USD_BBL": -7.0,
                "DESCUENTO_TRANSPORTE_USD_BBL": -3.3,
            })
            vol_max_delta = float(df_campo.loc[~df_campo["ES_SINTETICO"], TARGET].max()) \
                if n_real > 0 else 0.0

            print(f"\n  Sanity checks (BK_fin={bk_fin:.1f}, baseline={baseline_latest:.1f}):")
            sanity_check(campo, xgb_m, iso_m, bk_fin, medianas, vol_max_delta, baseline_latest)

        # ── Plot ──────────────────────────────────────────────────────────────
        print()
        generar_plot(campo, df_campo, xgb_m, iso_m, medianas, baseline_latest)

        # ── Predicciones en el tablon ─────────────────────────────────────────
        mask_c = df["CAMPO"] == campo
        df_pred = df[mask_c].copy()

        mask_feat_xgb = df_pred[FEATURES_XGB].notna().all(axis=1)
        mask_feat_iso = df_pred[FEATURE_ISO].notna()

        if mask_feat_xgb.any():
            d_xgb = xgb_m.predict(df_pred.loc[mask_feat_xgb, FEATURES_XGB].values)
            df.loc[mask_c & mask_feat_xgb, "PRED_XGBOOST_MBPE"] = np.maximum(
                baseline_latest + d_xgb, 0)

        if mask_feat_iso.any():
            d_iso = iso_m.predict(df_pred.loc[mask_feat_iso, FEATURE_ISO].values)
            df.loc[mask_c & mask_feat_iso, "PRED_ISOTONICA_MBPE"] = np.maximum(
                baseline_latest + d_iso, 0)

        # Delta vs OFICIAL solo donde hay OFICIAL disponible
        mask_vol = mask_c & df["VOLUMEN_1P_OFICIAL_MBPE"].notna()
        df.loc[mask_vol, "DELTA_XGBOOST_VS_OFICIAL"] = (
            df.loc[mask_vol, "PRED_XGBOOST_MBPE"] - df.loc[mask_vol, "VOLUMEN_1P_OFICIAL_MBPE"])
        df.loc[mask_vol, "DELTA_ISOTONICA_VS_OFICIAL"] = (
            df.loc[mask_vol, "PRED_ISOTONICA_MBPE"] - df.loc[mask_vol, "VOLUMEN_1P_OFICIAL_MBPE"])

        registros.append({
            "CAMPO":              campo,
            "N_REAL_DELTA":       n_real,
            "N_SINTETICOS":       n_sint,
            "W_SINTETICO":           round(w_bajo, 4),
            "W_SINTETICO_ESCALERA":  round(w_esc, 4),
            "BASELINE_LATEST":    round(baseline_latest, 2),
            "R2_LOO_XGB":         round(r2x,  4) if pd.notna(r2x)  else None,
            "MAE_LOO_XGB":        round(maex, 2) if pd.notna(maex) else None,
            "RMSE_LOO_XGB":       round(rmsex,2) if pd.notna(rmsex) else None,
            "ME_XGB":             round(mex,  2) if pd.notna(mex)  else None,
            "MAE_REL_XGB":        round(maex / baseline_latest, 4)
                                  if pd.notna(maex) and baseline_latest > 0 else None,
            "R2_LOO_ISO":         round(r2i,  4) if pd.notna(r2i)  else None,
            "MAE_LOO_ISO":        round(maei, 2) if pd.notna(maei) else None,
            "RMSE_LOO_ISO":       round(rmsei,2) if pd.notna(rmsei) else None,
            "ME_ISO":             round(mei,  2) if pd.notna(mei)  else None,
            "MAE_REL_ISO":        round(maei / baseline_latest, 4)
                                  if pd.notna(maei) and baseline_latest > 0 else None,
            # Ratio RMSE/MAE: >2 indica un fold LOO dominado por un outlier extremo
            "OUTLIER_RATIO_XGB":       round(rmsex / maex, 3)
                                       if pd.notna(maex) and maex > 0 else None,
            "ALERTA_LOO_OUTLIER_XGB":  bool(rmsex / maex > 2.0)
                                       if pd.notna(maex) and maex > 0 else False,
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

    # ── Guardar tablon con predicciones ──────────────────────────────────────
    df.to_parquet(STAGING / "tablon_unico.parquet", index=False)
    df.to_csv(STAGING / "tablon_unico.csv", index=False, encoding="utf-8-sig")
    print("\nTablon actualizado con predicciones.")
    print("\n=== 03_modelo.py — Completado ===")
