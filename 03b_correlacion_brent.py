"""
03b_correlacion_brent.py — Modelo 2: Precio Neto = g(Brent) por campo

Segundo modelo de la arquitectura de 2 (directriz 2026-06-11). Traduce el marcador Brent
al PRECIO NETO realizado de cada campo, que es la entrada del Modelo 1 (03_modelo.py).
Componiendo ambos se obtiene Brent -> Precio Neto -> Delta Reservas (lo hace 04).

POR QUE UNA REGRESION DIRECTA Neto = α + β·Brent (y no descuentos separados):
  La literatura (World Bank/ESMAP 2005) muestra que el diferencial de calidad se AMPLIA
  con el nivel de precio (terminos de interaccion descuento×Brent). Pero al ajustar eso
  POR CAMPO con datos anuales la señal es demasiado ruidosa: en 154 campos con n>=6, la
  pendiente del descuento de calidad vs Brent es significativa (p<0.10) solo en el 16%.
  En cambio, la regresion directa Neto = α + β·Brent logra R² mediano 0.916 (82% de campos
  > 0.8) y supera a "descuentos separados" y "proporcional" en los 154 campos. El termino
  β (<1 tipicamente) ABSORBE implicitamente la ampliacion del descuento con el precio.

METODO:
  - Theil-Sen (regresion robusta a outliers; mediana de pendientes) por campo, con puntos
    reales BASE (cierres anuales) + Consolidado (quarters). Robustez necesaria con N~8-18
    y algun trimestre atipico.
  - Monotonia: se exige β>0 (Brent↑ -> Neto↑), invariante para que la composicion con el
    Modelo 1 preserve Brent↑ -> Delta↑. Si Theil-Sen da β<=0, se degrada a proporcional.

ESCENARIOS (reemplazan los P10/P90 de descuentos del diseño anterior):
  Banda = recta ± cuantiles de los RESIDUALES (Neto_obs − Neto_hat):
    BASE = α + β·Brent ; BAJO = BASE + P10_resid ; ALTO = BASE + P90_resid
  Asi el descuento implicito (Brent − Neto) VARIA con el Brent en cada escenario, en vez
  de ser una constante historica desligada del precio.

DEGRADACION (n insuficiente):
  n>=5  -> Theil-Sen.
  2<=n<5 -> proporcional Neto = k·Brent (k = mediana de Neto/Brent; k>0 garantizado).
  n<2   -> medianas historicas: Neto = Brent + med_cal + med_tra (ALERTA=FALLBACK_MEDIANAS).

Salidas:
  datos/staging/correlacion_brent.csv   (coeficientes + bandas + descuento implicito @Brent ref)
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
N_MIN_PROP     = 2    # minimo para proporcional Neto=k*Brent
GATE_DORADO    = ["CASTILLA", "CASTILLA NORTE", "CASTILLA ESTE", "RUBIALES"]

ESCENARIOS = ["BAJO", "BASE", "ALTO"]


def datos_reales_campo(df: pd.DataFrame) -> pd.DataFrame:
    """Puntos reales con Brent y Precio Neto disponibles (BASE anual + Consolidado).
    Se excluyen sinteticos (no aportan a la formacion de precio)."""
    return df[(~df["ES_SINTETICO"]) &
              df["BRENT_FLAT_USD_BBL"].notna() &
              df["PRECIO_NETO_USD_BBL"].notna()].copy()


def ajustar_campo(g: pd.DataFrame, med_cal: float, med_tra: float) -> dict:
    """
    Ajusta Neto = g(Brent) para un campo. Retorna coeficientes, bandas de residuales,
    metricas y metodo. Garantiza pendiente positiva (degrada si no).
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
            # Pendiente no positiva: viola Brent↑->Neto↑. Degradar a proporcional.
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
        # Sin datos para regresion: usar medianas de descuento historicas
        slope = 1.0
        intercept = float(med_cal + med_tra)   # Neto = Brent + cal + tra (cal,tra<0)
        metodo = "FALLBACK_MEDIANAS"
        alerta = "FALLBACK_MEDIANAS"

    # Residuales y metricas de ajuste
    y_hat = intercept + slope * b
    resid = y - y_hat
    rmse = float(np.sqrt(np.mean(resid ** 2))) if n > 0 else np.nan
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if n > 1 else 0.0
    r2 = (1 - ss_res / ss_tot) if ss_tot > 1e-9 else np.nan

    # Cuantiles de residuales para las bandas BAJO/ALTO. Con n pequeño, P10/P90 colapsan
    # a min/max (mismo orden semantico, sin inventar colas).
    if n >= 5:
        q10, q50, q90 = (float(np.percentile(resid, 10)),
                         float(np.percentile(resid, 50)),
                         float(np.percentile(resid, 90)))
    elif n >= 2:
        q10, q50, q90 = float(np.min(resid)), 0.0, float(np.max(resid))
    else:
        q10 = q50 = q90 = 0.0

    return {
        "N_PUNTOS":       n,
        "METODO":         metodo,
        "ALPHA":          round(intercept, 4),
        "BETA":           round(slope, 4),
        "R2":             round(r2, 4) if pd.notna(r2) else None,
        "RMSE":           round(rmse, 3) if pd.notna(rmse) else None,
        "RESID_P10":      round(q10, 3),
        "RESID_P50":      round(q50, 3),
        "RESID_P90":      round(q90, 3),
        "BRENT_MIN_OBS":  round(float(b.min()), 2) if n > 0 else None,
        "BRENT_MAX_OBS":  round(float(b.max()), 2) if n > 0 else None,
        "ALERTA":         alerta,
    }


def neto_desde_brent(coef: dict, brent, escenario: str = "BASE") -> np.ndarray:
    """
    Predictor del Modelo 2: convierte Brent -> Precio Neto para un campo y escenario.
    Importable desde 04_pbi_export.py para componer Brent -> Neto -> Delta.

      BASE = α + β·Brent (recta Theil-Sen, estimacion central)
      BAJO = BASE + (RESID_P10 − RESID_P50)   (offset <= 0)
      ALTO = BASE + (RESID_P90 − RESID_P50)   (offset >= 0)

    Las bandas se centran en la MEDIANA de residuales (no en 0): Theil-Sen minimiza la
    mediana de |residuales|, no su media, por lo que P10/P90 crudos pueden no bracketar
    el 0 y romper el orden BAJO<=BASE<=ALTO (caso CARIBE). Centrar en P50 garantiza el
    orden por construccion (P10<=P50<=P90) preservando el ancho del spread observado.
    """
    brent = np.asarray(brent, dtype=float)
    base = coef["ALPHA"] + coef["BETA"] * brent
    if escenario == "BAJO":
        return base + (coef["RESID_P10"] - coef["RESID_P50"])
    if escenario == "ALTO":
        return base + (coef["RESID_P90"] - coef["RESID_P50"])
    return base


def plot_campo(campo: str, g: pd.DataFrame, coef: dict) -> None:
    """Scatter Brent vs Neto + recta Theil-Sen + banda de escenarios."""
    b = g["BRENT_FLAT_USD_BBL"].values.astype(float)
    y = g["PRECIO_NETO_USD_BBL"].values.astype(float)
    grid = np.linspace(b.min() - 5, b.max() + 10, 100) if len(b) else np.linspace(40, 100, 100)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(b, y, color="steelblue", s=60, zorder=5, label=f"Reales (n={coef['N_PUNTOS']})")
    ax.plot(grid, neto_desde_brent(coef, grid, "BASE"), color="darkorange",
            linewidth=2, label=f"BASE: Neto={coef['ALPHA']:.1f}+{coef['BETA']:.2f}·Brent")
    ax.plot(grid, neto_desde_brent(coef, grid, "ALTO"), color="green",
            linewidth=1, linestyle="--", label="ALTO (P90 resid)")
    ax.plot(grid, neto_desde_brent(coef, grid, "BAJO"), color="firebrick",
            linewidth=1, linestyle="--", label="BAJO (P10 resid)")
    ax.fill_between(grid, neto_desde_brent(coef, grid, "BAJO"),
                    neto_desde_brent(coef, grid, "ALTO"), color="orange", alpha=0.12)
    # Linea Neto=Brent (descuento cero) como referencia
    ax.plot(grid, grid, color="gray", linewidth=0.8, alpha=0.5, label="Neto=Brent (desc=0)")
    ax.set_xlabel("Brent Flat (USD/bbl)")
    ax.set_ylabel("Precio Neto (USD/bbl)")
    ax.set_title(f"{campo} — Modelo 2: Neto = g(Brent)  "
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
    print("=== 03b_correlacion_brent.py — Modelo 2: Neto = g(Brent) ===\n")
    ruta = STAGING / "tablon_unico.parquet"
    if not ruta.exists():
        raise FileNotFoundError("Ejecutar 01_etl.py primero.")
    df = pd.read_parquet(ruta)

    df_real = datos_reales_campo(df)

    # Medianas de descuento por campo (fallback cuando no hay puntos para regresion)
    medianas = (df_real.groupby("CAMPO")[["DESCUENTO_CALIDAD_USD_BBL",
                                          "DESCUENTO_TRANSPORTE_USD_BBL"]]
                .median().to_dict(orient="index"))

    # Brent de referencia para reportar el descuento implicito (mediana del Consolidado real)
    cons = df[(~df["ES_SINTETICO"]) & (~df["ES_BASELINE"]) & df["BRENT_FLAT_USD_BBL"].notna()]
    brent_ref = float(cons["BRENT_FLAT_USD_BBL"].median()) if not cons.empty else 75.0

    registros = []
    campos = sorted(df["CAMPO"].unique())
    for campo in campos:
        g = df_real[df_real["CAMPO"] == campo]
        med = medianas.get(campo, {})
        med_cal = float(med.get("DESCUENTO_CALIDAD_USD_BBL", -7.0))
        med_tra = float(med.get("DESCUENTO_TRANSPORTE_USD_BBL", -3.3))
        coef = ajustar_campo(g, med_cal, med_tra)

        # Descuento implicito @Brent ref para auditoria (Brent − Neto, positivo)
        neto_ref = float(neto_desde_brent(coef, np.array([brent_ref]), "BASE")[0])
        desc_impl = brent_ref - neto_ref

        registros.append({
            "CAMPO":             campo,
            **coef,
            "BRENT_REF":         round(brent_ref, 2),
            "NETO_REF_BASE":     round(neto_ref, 2),
            "DESCUENTO_IMPLICITO_REF": round(desc_impl, 2),
        })

    df_coef = pd.DataFrame(registros)
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
              f"R2>0.8: {(val['R2'] > 0.8).mean():.0%}")
    print(f"\n  Gate dorado:")
    for campo in GATE_DORADO:
        r = df_coef[df_coef["CAMPO"] == campo]
        if not r.empty:
            r = r.iloc[0]
            print(f"    {campo:<16} Neto = {r['ALPHA']:.2f} + {r['BETA']:.3f}*Brent  "
                  f"[{r['METODO']}, R2={r['R2']}, desc_impl@{brent_ref:.0f}={r['DESCUENTO_IMPLICITO_REF']:.1f}]")

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
