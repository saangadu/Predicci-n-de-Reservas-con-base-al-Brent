"""
diag_plot_pareto.py — Curva Vol(Brent) con el acantilado isotonico marcado (analisis offline)

NO forma parte de run_pipeline.py ni altera el modelo. Complementa diag_escalon.py
(informe de gerencia 2026-07-01, WS1): visualiza para los campos Pareto la curva completa
de la matriz de prediccion, sombrea la banda Brent observada en el Consolidado ($68-82,
9 quarters apilados — ver MAESTRO memoria [[consolidado-y-rango-precio]]) y marca la linea
vertical del "Brent oficial de comparacion" que usa 04_pbi_export.py::generar_comparacion_vs_anterior:

    brent_ref = round((brent_obs_min + brent_obs_max) / 2)   # linea 728 de 04_pbi_export.py

Ese punto (≈$75) NO es un deck de precio real: es el punto medio de 9 pronosticos
trimestrales independientes apilados en la misma banda estrecha. El hallazgo de esta
sesion es que ese punto medio cae, para varios campos grandes (CASTILLA, CHICHIMENE SW),
justo despues del escalon de la isotonica — el numero "oficial" de comparacion trimestral
descansa sobre el punto de maxima fragilidad de la curva, no sobre un pronostico de mercado.

Salida: resultados/plots_analisis/cliff_<CAMPO>.png (uno por campo).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR   = Path(__file__).parent
STAGING    = BASE_DIR / "datos" / "staging"
RESULTADOS = BASE_DIR / "resultados"
PLOTS_DIR  = RESULTADOS / "plots_analisis"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CAMPOS_PARETO = ["RUBIALES", "CASTILLA", "CASTILLA NORTE", "CAÑO SUR ESTE", "CHICHIMENE SW"]


def brent_ref_comparacion(df_tablon: pd.DataFrame) -> tuple[float, float, float]:
    """Reproduce el calculo de 04_pbi_export.py L728: punto medio de la banda
    Brent observada en el Consolidado (9 quarters reales, sin sinteticos/baseline)."""
    sub = df_tablon[(~df_tablon["ES_SINTETICO"]) & (~df_tablon["ES_BASELINE"])
                    & df_tablon["BRENT_FLAT_USD_BBL"].notna()]
    obs_min = float(sub["BRENT_FLAT_USD_BBL"].min())
    obs_max = float(sub["BRENT_FLAT_USD_BBL"].max())
    return round((obs_min + obs_max) / 2), obs_min, obs_max


def plot_campo(campo: str, df_pred: pd.DataFrame, brent_deck: float,
               banda_min: float, banda_max: float) -> None:
    sub = df_pred[(df_pred["CAMPO"] == campo) & (df_pred["MOTOR"] == "Isotonica")]
    sub = sub.sort_values("BRENT_USD_BBL")
    if sub.empty:
        print(f"  [WARN] {campo}: sin filas en output_matriz_prediccion.csv, omitiendo")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(sub["BRENT_USD_BBL"], sub["VOLUMEN_1P_PREDICHO_MBPE"], color="steelblue",
            linewidth=2, label="Isotonica (predicho)")
    ax.axvspan(banda_min, banda_max, color="gold", alpha=0.15,
               label=f"Banda Consolidado observada [${banda_min:.0f}, ${banda_max:.0f}]")
    ax.axvline(brent_deck, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Punto oficial de comparacion (midpoint banda) = ${brent_deck:.0f}")

    baseline = sub["VOLUMEN_1P_BASELINE_MBPE"].dropna()
    if not baseline.empty:
        ax.axhline(float(baseline.iloc[0]), color="grey", linestyle=":", linewidth=1,
                   label=f"Baseline 2025 = {float(baseline.iloc[0]):.1f} MBPE")

    ax.set_xlabel("Brent USD/bbl")
    ax.set_ylabel("Volumen 1P predicho (MBPE)")
    ax.set_title(f"{campo} — curva Vol(Brent) y acantilado isotonico")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ruta = PLOTS_DIR / f"cliff_{campo.replace(' ', '_')}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {campo:<20} -> {ruta}")


if __name__ == "__main__":
    print("=== diag_plot_pareto.py — Curvas Vol(Brent) con acantilado marcado ===\n")

    ruta_tablon = STAGING / "tablon_unico.parquet"
    ruta_pred = RESULTADOS / "output_matriz_prediccion.csv"
    if not ruta_tablon.exists():
        raise FileNotFoundError("Ejecutar 01_etl.py primero (tablon_unico.parquet no existe).")
    if not ruta_pred.exists():
        raise FileNotFoundError("Ejecutar 04_pbi_export.py primero (output_matriz_prediccion.csv no existe).")

    df_tablon = pd.read_parquet(ruta_tablon)
    df_pred = pd.read_csv(ruta_pred)

    brent_deck, banda_min, banda_max = brent_ref_comparacion(df_tablon)
    print(f"Banda Consolidado observada: [${banda_min:.1f}, ${banda_max:.1f}]")
    print(f"Punto oficial de comparacion (midpoint): ${brent_deck:.0f}\n")

    for campo in CAMPOS_PARETO:
        plot_campo(campo, df_pred, brent_deck, banda_min, banda_max)

    print(f"\nGuardado en: {PLOTS_DIR}")
