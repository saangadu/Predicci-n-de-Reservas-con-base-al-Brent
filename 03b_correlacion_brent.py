"""
03b_correlacion_brent.py — Modelo 2: Precio Aceite = g(Brent) por campo

Segundo modelo de la arquitectura de 2 (directriz 2026-06-11; reformado 2026-06-12).
Traduce el marcador Brent al PRECIO ACEITE realizado de cada campo (columna
PRECIO_NETO_USD_BBL del tablon), que es la entrada del Modelo 1 (03_modelo.py).
Componiendo ambos se obtiene Brent -> Precio Aceite -> Delta Reservas (lo hace 04).

POR QUE UNA REGRESION DIRECTA Aceite = α + β·Brent (y no descuentos separados):
  La literatura (World Bank/ESMAP 2005) muestra que el diferencial de calidad se AMPLIA
  con el nivel de precio (terminos de interaccion descuento×Brent). Pero al ajustar eso
  POR CAMPO con datos anuales la señal es demasiado ruidosa: en 154 campos con n>=6, la
  pendiente del descuento de calidad vs Brent es significativa (p<0.10) solo en el 16%.
  La regresion directa absorbe los descuentos implicitamente en β (<1 tipicamente).
  Los descuentos quedan AISLADOS del flujo de prediccion (directriz 2026-06-12).

SOLO PUNTOS HIST REALES (directriz 2026-06-12):
  La recta se ajusta unicamente con cierres anuales certificados (ES_BASELINE, precios
  realizados, Brent $43-98). Los 9 quarters del Consolidado se EXCLUYEN: son pronosticos
  independientes apilados en banda Brent $68-82 — sin dispersion en X no aportan
  pendiente, solo varianza vertical. Evidencia (160 campos comparables): R² mediano
  0.909 -> 0.967; campos R²>0.9 de 85 -> 141 al excluirlos (LOS ACEITES 0.31 -> 0.97).

METODO:
  - Theil-Sen (regresion robusta a outliers; mediana de pendientes) por campo.
  - Monotonia: se exige β>0 (Brent↑ -> Aceite↑), invariante para que la composicion con
    el Modelo 1 preserve Brent↑ -> Delta↑. Si Theil-Sen da β<=0, se degrada a proporcional.
  - Sin escenarios BAJO/ALTO (retirados 2026-06-12): una sola recta BASE.

DEGRADACION (n insuficiente):
  n>=5  -> Theil-Sen.
  2<=n<5 -> proporcional Aceite = k·Brent (k = mediana de Aceite/Brent; k>0 garantizado).
  n<2   -> fallback de portafolio: Aceite = k_portafolio·Brent, con k_portafolio =
           mediana de β de los campos THEILSEN validos (METODO=FALLBACK_BETA_PORTAFOLIO,
           ES_FALLBACK=True, ALERTA=SIN_HIST_FALLBACK_BETA). Reemplaza el fallback de
           medianas de descuentos (retirado: los descuentos salen del flujo).

METRICAS (por campo, en correlacion_brent.csv):
  R2 / RMSE        : bondad de ajuste in-sample (optimista con N pequeño).
  R2_LOO / MAE_LOO : validacion out-of-sample (LOO-CV de la recta) — generalizacion real.
  ES_FALLBACK      : True si el campo no tiene relacion propia (tag visible hasta el export).

Salidas:
  datos/staging/correlacion_brent.csv   (coeficientes + metricas + descuento implicito)
  resultados/correlacion_brent.csv      (espejo para Power BI)
  datos/staging/plots_correlacion/<campo>.png  (gate dorado + top materiales)
  resultados/plots_analisis/correlacion_brent_resumen.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR    = Path(__file__).parent
STAGING     = BASE_DIR / "datos" / "staging"
RESULTADOS  = BASE_DIR / "resultados"
PLOTS_DIR   = STAGING / "plots_correlacion"
PLOTS_ANAL  = RESULTADOS / "plots_analisis"
STAGING.mkdir(parents=True, exist_ok=True)
RESULTADOS.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_ANAL.mkdir(parents=True, exist_ok=True)

N_MIN_THEILSEN = 5    # minimo de puntos para Theil-Sen robusto
N_MIN_PROP     = 2    # minimo para proporcional Aceite=k*Brent
GATE_DORADO    = ["CASTILLA", "CASTILLA NORTE", "CASTILLA ESTE", "RUBIALES"]


def datos_reales_campo(df: pd.DataFrame) -> pd.DataFrame:
    """Puntos HIST reales (cierres anuales BASE con precio REALIZADO). Se excluyen
    sinteticos y quarters Consolidado: los quarters son pronosticos apilados en banda
    Brent $68-82 que contaminan la pendiente (decision 2026-06-12, MAESTRO §10)."""
    return df[(~df["ES_SINTETICO"]) &
              df["ES_BASELINE"] &
              df["BRENT_FLAT_USD_BBL"].notna() &
              df["PRECIO_NETO_USD_BBL"].notna()].copy()


def _loo_recta(b: np.ndarray, y: np.ndarray, metodo: str) -> tuple[float, float]:
    """
    LOO-CV de la recta Neto=g(Brent): deja un punto fuera, reajusta con el MISMO metodo
    sobre el resto, predice el punto excluido y acumula el error out-of-sample.

    Devuelve (MAE_LOO, R2_LOO). R2_LOO compara el SS de los residuales LOO contra la
    varianza total (1 = perfecto; <=0 = no supera predecir la media). Mide capacidad
    GENERALIZADORA de la recta, a diferencia de R²/RMSE in-sample (optimistas con N~8).
    """
    n = len(b)
    if n < 4 or np.ptp(b) <= 1e-6:
        return np.nan, np.nan  # muy pocos puntos para un hold-out informativo
    errores = []
    for i in range(n):
        b_tr = np.delete(b, i); y_tr = np.delete(y, i)
        if np.ptp(b_tr) <= 1e-6:
            continue
        if metodo == "THEILSEN":
            s, a, _, _ = stats.theilslopes(y_tr, b_tr)
            if s <= 0:  # mismo guard que el ajuste global: degradar a proporcional
                s = float(np.median(y_tr / np.where(b_tr == 0, np.nan, b_tr))); a = 0.0
        else:  # PROPORCIONAL
            s = float(np.median(y_tr / np.where(b_tr == 0, np.nan, b_tr))); a = 0.0
        errores.append(y[i] - (a + s * b[i]))
    if not errores:
        return np.nan, np.nan
    errores = np.asarray(errores, dtype=float)
    mae_loo = float(np.mean(np.abs(errores)))
    ss_res  = float(np.sum(errores ** 2))
    ss_tot  = float(np.sum((y - np.mean(y)) ** 2))
    r2_loo  = (1 - ss_res / ss_tot) if ss_tot > 1e-9 else np.nan
    return mae_loo, r2_loo


def ajustar_campo(g: pd.DataFrame) -> dict:
    """
    Ajusta Aceite = g(Brent) para un campo con sus puntos HIST. Retorna coeficientes,
    metricas y metodo. Garantiza pendiente positiva (degrada si no). Con n<2 marca
    PENDIENTE_FALLBACK: la segunda pasada del __main__ le asigna el k de portafolio
    (no se puede resolver aqui porque requiere los β de todos los campos).
    """
    b = g["BRENT_FLAT_USD_BBL"].values.astype(float)
    y = g["PRECIO_NETO_USD_BBL"].values.astype(float)
    n = len(g)
    alerta = ""

    if n >= N_MIN_THEILSEN and np.ptp(b) > 1e-6:
        # Theil-Sen: pendiente = mediana de pendientes pareadas (robusto a outliers)
        slope, intercept, _, _ = stats.theilslopes(y, b)
        metodo = "THEILSEN"
        if slope <= 0:
            # Pendiente no positiva: viola Brent↑->Aceite↑. Degradar a proporcional.
            slope = float(np.median(y / np.where(b == 0, np.nan, b)))
            intercept = 0.0
            metodo = "PROPORCIONAL"
            alerta = "BETA_NO_POSITIVA"
    elif n >= N_MIN_PROP:
        slope = float(np.median(y / np.where(b == 0, np.nan, b)))
        intercept = 0.0
        metodo = "PROPORCIONAL"
        alerta = "POCOS_PUNTOS"
    else:
        # Sin historia de precio: se resuelve en la segunda pasada (k de portafolio)
        slope = np.nan
        intercept = 0.0
        metodo = "PENDIENTE_FALLBACK"
        alerta = "SIN_HIST_FALLBACK_BETA"

    # Residuales y metricas de ajuste
    if metodo != "PENDIENTE_FALLBACK":
        y_hat = intercept + slope * b
        resid = y - y_hat
        rmse = float(np.sqrt(np.mean(resid ** 2))) if n > 0 else np.nan
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if n > 1 else 0.0
        r2 = (1 - ss_res / ss_tot) if ss_tot > 1e-9 else np.nan
        mae_loo, r2_loo = _loo_recta(b, y, metodo)
    else:
        rmse = r2 = mae_loo = r2_loo = np.nan

    return {
        "N_PUNTOS":       n,
        "METODO":         metodo,
        "ALPHA":          round(intercept, 4),
        "BETA":           round(slope, 4) if pd.notna(slope) else None,
        "R2":             round(r2, 4) if pd.notna(r2) else None,
        "RMSE":           round(rmse, 3) if pd.notna(rmse) else None,
        "R2_LOO":         round(r2_loo, 4) if pd.notna(r2_loo) else None,
        "MAE_LOO":        round(mae_loo, 3) if pd.notna(mae_loo) else None,
        "BRENT_MIN_OBS":  round(float(b.min()), 2) if n > 0 else None,
        "BRENT_MAX_OBS":  round(float(b.max()), 2) if n > 0 else None,
        "ES_FALLBACK":    metodo == "PENDIENTE_FALLBACK",
        "ALERTA":         alerta,
    }


def neto_desde_brent(coef: dict, brent) -> np.ndarray:
    """
    Predictor del Modelo 2: convierte Brent -> Precio Aceite para un campo.
    Importable desde 03_modelo.py (p_ref del re-anclaje) y 04_pbi_export.py
    (composicion Brent -> Aceite -> Delta). Sin escenarios desde 2026-06-12.
    """
    brent = np.asarray(brent, dtype=float)
    return coef["ALPHA"] + coef["BETA"] * brent


def plot_campo(campo: str, g: pd.DataFrame, coef: dict) -> None:
    """Scatter Brent vs Aceite + recta Theil-Sen (sin bandas desde 2026-06-12)."""
    b = g["BRENT_FLAT_USD_BBL"].values.astype(float)
    y = g["PRECIO_NETO_USD_BBL"].values.astype(float)
    grid = np.linspace(b.min() - 5, b.max() + 10, 100) if len(b) else np.linspace(40, 100, 100)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(b, y, color="steelblue", s=60, zorder=5,
               label=f"Reales HIST (n={coef['N_PUNTOS']})")
    ax.plot(grid, neto_desde_brent(coef, grid), color="darkorange",
            linewidth=2, label=f"Aceite={coef['ALPHA']:.1f}+{coef['BETA']:.2f}·Brent")
    # Linea Aceite=Brent (descuento cero) como referencia
    ax.plot(grid, grid, color="gray", linewidth=0.8, alpha=0.5, label="Aceite=Brent (desc=0)")
    ax.set_xlabel("Brent Flat (USD/bbl)")
    ax.set_ylabel("Precio Aceite (USD/bbl)")
    ax.set_title(f"{campo} — Modelo 2: Aceite = g(Brent)  "
                 f"[{coef['METODO']}, R2={coef['R2']}]")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    ruta = PLOTS_DIR / f"{campo.replace(' ', '_')}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot: {ruta.name}")


def plot_resumen(df_coef: pd.DataFrame) -> None:
    """Distribucion de β y R² del Modelo 2 sobre el portafolio."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    betas = df_coef["BETA"].dropna()
    ax1.hist(betas, bins=30, color="steelblue", alpha=0.8, edgecolor="white")
    ax1.axvline(1.0, color="gray", linestyle="--", label="β=1 (Neto sigue 1:1 al Brent)")
    ax1.axvline(float(betas.median()), color="firebrick", linestyle="--",
                label=f"Mediana={betas.median():.2f}")
    ax1.set_xlabel("β (sensibilidad Neto/Brent)")
    ax1.set_ylabel("N° campos"); ax1.set_title("Pendiente Neto vs Brent")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    r2 = df_coef["R2"].dropna()
    ax2.hist(r2, bins=30, color="seagreen", alpha=0.8, edgecolor="white")
    ax2.axvline(float(r2.median()), color="firebrick", linestyle="--",
                label=f"Mediana={r2.median():.2f}")
    ax2.set_xlabel("R²"); ax2.set_ylabel("N° campos")
    ax2.set_title("Bondad de ajuste del Modelo 2"); ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.suptitle("Modelo 2 (Neto = g(Brent)) — resumen del portafolio", fontsize=13)
    plt.tight_layout()
    ruta = PLOTS_ANAL / "correlacion_brent_resumen.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Resumen: {ruta}")


if __name__ == "__main__":
    print("=== 03b_correlacion_brent.py — Modelo 2: Aceite = g(Brent) ===\n")
    ruta = STAGING / "tablon_unico.parquet"
    if not ruta.exists():
        raise FileNotFoundError("Ejecutar 01_etl.py primero.")
    df = pd.read_parquet(ruta)

    df_real = datos_reales_campo(df)

    # Brent de referencia para reportar el descuento implicito (mediana del Consolidado real)
    cons = df[(~df["ES_SINTETICO"]) & (~df["ES_BASELINE"]) & df["BRENT_FLAT_USD_BBL"].notna()]
    brent_ref = float(cons["BRENT_FLAT_USD_BBL"].median()) if not cons.empty else 75.0

    registros = []
    campos = sorted(df["CAMPO"].unique())
    for campo in campos:
        g = df_real[df_real["CAMPO"] == campo]
        registros.append({"CAMPO": campo, **ajustar_campo(g)})

    df_coef = pd.DataFrame(registros)

    # Segunda pasada: fallback de portafolio para campos sin historia de precio.
    # k = mediana del ratio Aceite/Brent implicito de las rectas Theil-Sen evaluadas
    # en el Brent de referencia ((α+β·ref)/ref). La β sola sobreestima (mediana ~1.13
    # porque α<0 tipico la compensa); el ratio en el punto de operacion es la
    # relacion tipica realizada del portafolio. Sin descuentos no hay otra fuente.
    ts = df_coef[df_coef["METODO"] == "THEILSEN"]
    if not ts.empty:
        ratios = (ts["ALPHA"] + ts["BETA"] * brent_ref) / brent_ref
        k_portafolio = round(float(ratios.median()), 4)
    else:
        k_portafolio = 1.0
    mask_fb = df_coef["METODO"] == "PENDIENTE_FALLBACK"
    df_coef.loc[mask_fb, ["ALPHA", "BETA", "METODO"]] = [0.0, k_portafolio,
                                                         "FALLBACK_BETA_PORTAFOLIO"]
    print(f"  k_portafolio (mediana ratio Aceite/Brent @ref de THEILSEN) = {k_portafolio}  "
          f"-> aplicado a {int(mask_fb.sum())} campos sin historia (ES_FALLBACK=True)")

    # Descuento implicito @Brent ref para auditoria (Brent − Aceite, positivo)
    df_coef["BRENT_REF"] = round(brent_ref, 2)
    df_coef["NETO_REF_BASE"] = (df_coef["ALPHA"] + df_coef["BETA"] * brent_ref).round(2)
    df_coef["DESCUENTO_IMPLICITO_REF"] = (brent_ref - df_coef["NETO_REF_BASE"]).round(2)

    df_coef.to_csv(STAGING / "correlacion_brent.csv", index=False, encoding="utf-8-sig")
    df_coef.to_csv(RESULTADOS / "correlacion_brent.csv", index=False, encoding="utf-8-sig")
    print(f"  Coeficientes: {RESULTADOS / 'correlacion_brent.csv'}  ({len(df_coef)} campos)")

    # Resumen por metodo
    print(f"\n{'-'*60}\n  Resumen Modelo 2\n{'-'*60}")
    print("  Metodo de ajuste por campo:")
    print(df_coef["METODO"].value_counts().to_string())
    val = df_coef[df_coef["METODO"] == "THEILSEN"]
    if not val.empty:
        print(f"\n  Theil-Sen (n={len(val)}): beta mediana={val['BETA'].median():.3f}  "
              f"R2 mediana={val['R2'].median():.3f}  "
              f"R2>0.9: {(val['R2'] > 0.9).mean():.0%}")
        r2loo = val["R2_LOO"].dropna()
        if not r2loo.empty:
            print(f"  LOO-CV: R2_LOO mediana={r2loo.median():.3f}  "
                  f"MAE_LOO mediana={val['MAE_LOO'].median():.2f} USD/bbl")
    print(f"\n  Gate dorado:")
    for campo in GATE_DORADO:
        r = df_coef[df_coef["CAMPO"] == campo]
        if not r.empty:
            r = r.iloc[0]
            print(f"    {campo:<16} Aceite = {r['ALPHA']:.2f} + {r['BETA']:.3f}*Brent  "
                  f"[{r['METODO']}, R2={r['R2']}, R2_LOO={r['R2_LOO']}, "
                  f"desc_impl@{brent_ref:.0f}={r['DESCUENTO_IMPLICITO_REF']:.1f}]")

    # Plots: gate dorado + top-6 materiales por baseline
    print("\n  Generando plots...")
    plot_campos = list(GATE_DORADO)
    baselines = (df[(df["ESCENARIO"] == "BASE") & df["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
                 .groupby("CAMPO")["VOLUMEN_1P_OFICIAL_MBPE"].max()
                 .sort_values(ascending=False))
    for c in baselines.index[:6]:
        if c not in plot_campos:
            plot_campos.append(c)
    for campo in plot_campos:
        g = df_real[df_real["CAMPO"] == campo]
        r = df_coef[df_coef["CAMPO"] == campo]
        if not r.empty and len(g) > 0:
            plot_campo(campo, g, r.iloc[0].to_dict())

    plot_resumen(df_coef)
    print("\n=== 03b_correlacion_brent.py — Completado ===")
