"""
prediccion_puntual.py — prediccion puntual a un precio Brent arbitrario.

Offline, solo lectura: NO reentrena ni corre el pipeline. Interpola linealmente
entre los dos puntos de la grilla publicada (paso $1 USD/bbl) que rodean al
Brent solicitado, sobre el output ya generado en resultados/. No cambia
ningun modelo, motor ni decision — es una lectura auxiliar del export vigente.

Motivo de usar el export ya existente (2026-07-17, s14/s16) y no una corrida
fresca: 2026_Q2 aun no trae reservas reales (columna Reservas 1P placeholder
= copia de 2026_Q1, ver docs_public/DIFFICULTIES.md); correr el pipeline hoy
mezclaria precios reales de Q2 con volumen inventado. Este export congelado
es el estado "antes del Q2 real" — el que se compara mas adelante contra el
reentrenamiento cuando llegue el dato real (tarea diferida, ver MAESTRO §9).

Uso: python prediccion_puntual.py --brent 82.9
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
RESULTADOS = BASE_DIR / "resultados"
ENTRADA = RESULTADOS / "output_matriz_prediccion.csv"

COLS_INTERPOLAR = [
    "PRECIO_NETO_EFECTIVO_USD_BBL", "DELTA_PRED_MBPE",
    "VOLUMEN_1P_PREDICHO_MBPE", "DELTA_VS_BASE_MBPE", "DELTA_VS_BASE_PCT",
    "VOL_P25_MBPE", "VOL_P75_MBPE", "ANCHO_BANDA_LOYO_MBPE",
]
COLS_PASAR = [
    "CAMPO", "MOTOR", "VOLUMEN_1P_BASELINE_MBPE", "TIPO_MODELO", "TIPO_DATO",
    "NIVEL_CONFIANZA", "MOTIVO_CONFIANZA", "Q_OBJETIVO", "VIGENCIA_BASE",
    "FECHA_PREDICCION",
]


def _interpolar(brent: float) -> pd.DataFrame:
    df = pd.read_csv(ENTRADA, encoding="utf-8-sig")
    grid = sorted(df["BRENT_USD_BBL"].unique())
    if brent < grid[0] or brent > grid[-1]:
        raise ValueError(f"Brent {brent} fuera de la grilla publicada "
                          f"[{grid[0]}, {grid[-1]}]")
    # punto exacto en grilla (incluye BRENT_REF si redondea a el)
    exactos = [g for g in grid if math.isclose(g, brent, abs_tol=1e-6)]
    if exactos:
        sub = df[np.isclose(df["BRENT_USD_BBL"], exactos[0])].copy()
        sub["BRENT_USD_BBL"] = brent
        return sub[COLS_PASAR + ["BRENT_USD_BBL"] + COLS_INTERPOLAR]

    lo = max(g for g in grid if g < brent)
    hi = min(g for g in grid if g > brent)
    w = (brent - lo) / (hi - lo)

    df_lo = df[np.isclose(df["BRENT_USD_BBL"], lo)].set_index(["CAMPO", "MOTOR"])
    df_hi = df[np.isclose(df["BRENT_USD_BBL"], hi)].set_index(["CAMPO", "MOTOR"])
    idx_comun = df_lo.index.intersection(df_hi.index)
    df_lo, df_hi = df_lo.loc[idx_comun], df_hi.loc[idx_comun]

    salida = df_lo[COLS_PASAR[2:]].copy()  # cols estaticas (sin CAMPO/MOTOR, ya en index)
    salida["BRENT_USD_BBL"] = brent
    for c in COLS_INTERPOLAR:
        salida[c] = df_lo[c] * (1 - w) + df_hi[c] * w
    return salida.reset_index()[COLS_PASAR + ["BRENT_USD_BBL"] + COLS_INTERPOLAR]


def _resumen_portafolio(sub: pd.DataFrame) -> dict:
    iso = sub[sub["MOTOR"] == "Isotonica"]
    vol_total = iso["VOLUMEN_1P_PREDICHO_MBPE"].sum()
    baseline_total = iso["VOLUMEN_1P_BASELINE_MBPE"].sum()
    # cuadratura de semianchos (regla s14, ver docs_public/METHODOLOGY.md):
    # sumar sobreestima al asumir errores perfectamente correlacionados
    p25 = iso["VOL_P25_MBPE"].dropna()
    p75 = iso["VOL_P75_MBPE"].dropna()
    vol_p25 = (p25 - iso.loc[p25.index, "VOLUMEN_1P_PREDICHO_MBPE"]).abs()
    vol_p75 = (p75 - iso.loc[p75.index, "VOLUMEN_1P_PREDICHO_MBPE"]).abs()
    banda_cuadratura = math.sqrt((vol_p25**2).sum() + (vol_p75**2).sum()) \
        if len(vol_p25) else float("nan")
    return {
        "n_campos": iso["CAMPO"].nunique(),
        "volumen_1p_total_mbpe": vol_total,
        "baseline_total_mbpe": baseline_total,
        "delta_vs_baseline_mbpe": vol_total - baseline_total,
        "delta_vs_baseline_pct": 100 * (vol_total - baseline_total) / baseline_total
        if baseline_total else float("nan"),
        "banda_cuadratura_mbpe": banda_cuadratura,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brent", type=float, required=True,
                     help="Precio Brent USD/bbl a predecir (ej. 82.9)")
    args = ap.parse_args()

    sub = _interpolar(args.brent)
    RESULTADOS.mkdir(exist_ok=True)
    fecha = pd.Timestamp.now().strftime("%Y-%m-%d")
    ruta_out = RESULTADOS / f"prediccion_puntual_brent_{args.brent}_{fecha}.csv"
    sub.to_csv(ruta_out, index=False, encoding="utf-8-sig")

    resumen = _resumen_portafolio(sub)
    print(f"\n{'='*60}")
    print(f"Prediccion puntual — Brent = ${args.brent} USD/bbl")
    print(f"Fuente: {ENTRADA.name} (corrida 2026-07-17, s14/s16 — pre-Q2 real)")
    print(f"{'='*60}")
    print(f"Campos: {resumen['n_campos']}")
    print(f"Volumen 1P predicho (Isotonica, portafolio): "
          f"{resumen['volumen_1p_total_mbpe']:.1f} MBPE")
    print(f"Baseline (cierre previo, portafolio): "
          f"{resumen['baseline_total_mbpe']:.1f} MBPE")
    print(f"Delta vs baseline: {resumen['delta_vs_baseline_mbpe']:+.1f} MBPE "
          f"({resumen['delta_vs_baseline_pct']:+.2f}%)")
    print(f"Banda P25/P75 portafolio (cuadratura): "
          f"±{resumen['banda_cuadratura_mbpe']:.1f} MBPE")
    print(f"\nExport detallado: {ruta_out}")


if __name__ == "__main__":
    main()
