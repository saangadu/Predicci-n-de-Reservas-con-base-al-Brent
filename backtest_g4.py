"""
backtest_g4.py — Backtesting rodante pred-vs-real (NORTE G4, harness offline)

NO forma parte de run_pipeline.py. Automatiza el paso 3 del proceso G4 de docs/NORTE.md:
al llegar el dato REAL de un quarter que fue predicho, comparar prediccion vs real por
campo. Esta es la metrica de utilidad REAL del piloto — LOO/LOYO son solo proxies.

Como funciona:
  1. Lee los snapshots de resultados/historico_predicciones/prediccion_<Q>_generada_*.csv
     (motor Isotonica, fila con BRENT mas cercano al Brent real del quarter).
  2. Busca en el tablon el dato real del quarter objetivo: filas CONSOLIDADO_<Q> con
     VOLUMEN_1P_SENSIBILIDAD_MBPE (cuando el usuario anexa el Q real al Consolidado).
  3. Si el dato existe: error por campo (pred - real, absoluto y relativo) →
     resultados/backtest_g4.csv. Si no existe: sale limpio con aviso (no bloquea).

Regla G4.4 (NORTE): el reentrenamiento posterior usa SOLO datos reales — este script
es de evaluacion, jamas inyecta predicciones como datos.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR   = Path(__file__).parent
STAGING    = BASE_DIR / "datos" / "staging"
RESULTADOS = BASE_DIR / "resultados"
HISTORICO  = RESULTADOS / "historico_predicciones"


def quarters_predichos() -> dict:
    """Snapshots disponibles: {quarter: ruta del snapshot mas reciente}."""
    out = {}
    for p in sorted(HISTORICO.glob("prediccion_*_generada_*.csv")):
        m = re.match(r"prediccion_(\d{4}_Q\d)_generada_", p.name)
        if m:
            out[m.group(1)] = p   # sorted() → se queda el mas reciente
    return out


def real_del_quarter(df: pd.DataFrame, q: str) -> pd.DataFrame:
    """Dato real del quarter: filas CONSOLIDADO_<q> con volumen de sensibilidad.
    Retorna DataFrame CAMPO / REAL_MBPE / BRENT_REAL (vacio si el Q no ha llegado)."""
    sub = df[(df["ESCENARIO"] == f"CONSOLIDADO_{q}")
             & df["VOLUMEN_1P_SENSIBILIDAD_MBPE"].notna()]
    if sub.empty:
        return pd.DataFrame(columns=["CAMPO", "REAL_MBPE", "BRENT_REAL"])
    return (sub.groupby("CAMPO")
            .agg(REAL_MBPE=("VOLUMEN_1P_SENSIBILIDAD_MBPE", "first"),
                 BRENT_REAL=("BRENT_FLAT_USD_BBL", "first"))
            .reset_index())


def prediccion_en_brent(snapshot: Path, reales: pd.DataFrame) -> pd.DataFrame:
    """Del snapshot, la prediccion Isotonica en la fila de Brent mas cercana al real
    DE CADA CAMPO (fix 2026-08-05: antes se usaba un unico Brent — el primero no-nulo
    del quarter — para todo el portafolio, lo cual es incorrecto si el Brent real
    difiere por campo, p.ej. por fecha de cierre de vigencia distinta). `reales`
    trae columnas CAMPO/BRENT_REAL."""
    pred = pd.read_csv(snapshot)
    pred = pred[pred["MOTOR"] == "Isotonica"].copy()
    pred = pred.merge(reales[["CAMPO", "BRENT_REAL"]], on="CAMPO", how="inner")
    pred["_dist"] = (pred["BRENT_USD_BBL"] - pred["BRENT_REAL"]).abs()
    idx = pred.groupby("CAMPO")["_dist"].idxmin()
    cols = ["CAMPO", "BRENT_USD_BBL", "VOLUMEN_1P_PREDICHO_MBPE", "NIVEL_CONFIANZA"]
    cols = [c for c in cols if c in pred.columns]
    return pred.loc[idx, cols].rename(columns={
        "BRENT_USD_BBL": "BRENT_PRED", "VOLUMEN_1P_PREDICHO_MBPE": "PRED_MBPE"})


if __name__ == "__main__":
    print("=== backtest_g4.py — Backtesting pred-vs-real (NORTE G4) ===\n")

    ruta = STAGING / "tablon_unico.parquet"
    if not ruta.exists():
        raise FileNotFoundError("Ejecutar 01_etl.py primero (tablon_unico.parquet no existe).")
    df = pd.read_parquet(ruta)

    snapshots = quarters_predichos()
    if not snapshots:
        print("Sin snapshots en resultados/historico_predicciones/ — nada que backtestear.")
        raise SystemExit(0)

    filas = []
    for q, snap in snapshots.items():
        reales = real_del_quarter(df, q)
        if reales.empty:
            print(f"  {q}: dato real aun no disponible en el Consolidado — pendiente.")
            continue
        sin_brent = reales["BRENT_REAL"].isna().sum()
        if sin_brent:
            print(f"  [WARN] {q}: {sin_brent} campos sin Brent real en el tablon, se omiten.")
        reales_con_brent = reales[reales["BRENT_REAL"].notna()]
        if reales_con_brent.empty:
            print(f"  [WARN] {q}: ningun campo con Brent real, omitiendo quarter.")
            continue
        # Brent real POR CAMPO (fix: antes se aplicaba un unico Brent a todo el
        # portafolio — ver docstring de prediccion_en_brent).
        pred = prediccion_en_brent(snap, reales_con_brent)
        m = pred.merge(reales_con_brent[["CAMPO", "REAL_MBPE"]], on="CAMPO", how="inner")
        m["Q_OBJETIVO"] = q
        m["ERROR_MBPE"] = (m["PRED_MBPE"] - m["REAL_MBPE"]).round(2)
        m["ERROR_REL"] = (m["ERROR_MBPE"] / m["REAL_MBPE"].replace(0, np.nan)).round(4)
        filas.append(m)
        brent_real = reales_con_brent["BRENT_REAL"].mean()
        print(f"  {q}: {len(m)} campos comparados (Brent real promedio ~${brent_real:.1f}, "
              f"snapshot {snap.name})")
        print(f"     MAE portafolio: {m['ERROR_MBPE'].abs().mean():.2f} MBPE | "
              f"mediana error rel: {m['ERROR_REL'].abs().median():.1%}")

    if not filas:
        print("\nNingun quarter predicho tiene dato real todavia. G4 queda pendiente "
              "hasta anexar el proximo Q al Consolidado (proceso trimestral).")
        raise SystemExit(0)

    df_bt = pd.concat(filas, ignore_index=True)
    ruta_out = RESULTADOS / "backtest_g4.csv"
    df_bt.to_csv(ruta_out, index=False, encoding="utf-8-sig")
    print(f"\nGuardado: {ruta_out}")
