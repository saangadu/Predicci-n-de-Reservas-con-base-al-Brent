"""
experimento_vigencia.py — Experimento offline: centrado por año de certificación (Modelo 1)

Pregunta (2026-06-11): campos como AKACIAS (skill≈0) y CUPIAGUA (skill<0) no superan al
predictor ingenuo porque su Delta salta entre AÑOS de certificación (2024→2025) SIN
relación con el precio. La isotónica de producción aplana esa nube y empata (o pierde)
contra la media.

Hipótesis a probar: descomponer  Delta = efecto_año + f(Precio Neto)
  - efecto_año  = media de Delta dentro de cada AÑO (nivel de la certificación).
  - f(PrecioNeto) = isotónica sobre los Delta CENTRADOS (Delta − media_año).
  - Predicción re-anclada al AÑO más reciente: delta_hat = media_año_reciente + f(precio).

Se compara, por campo y vía LOO-CV sobre puntos reales, el SKILL del modelo de producción
(M0: isotónica directa) contra el experimento (M1: centrado por año).

CRITERIO DE ACEPTACIÓN (para integrar en 03_modelo.py):
  Subir el SKILL de AKACIAS/CUPIAGUA SIN degradar el gate dorado. Además, M1 solo aporta
  valor real si su mejora NO depende de conocer la etiqueta de año del punto de prueba
  (que en producción NO se conoce: siempre se re-ancla al año reciente). Por eso medimos
  también el RANGO de la curva de producción (delta máx − mín en la banda de precios):
  si M1 sigue siendo plano, la "ganancia" de SKILL es fuga de la etiqueta de año y no se
  traduce en sensibilidad al precio → NO integrar.

NO altera el pipeline. Salida: resultados/experimento_vigencia.csv
"""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from motores_modelo1 import MotorIsotonico

BASE_DIR   = Path(__file__).parent
STAGING    = BASE_DIR / "datos" / "staging"
RESULTADOS = BASE_DIR / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

modelo = importlib.import_module("03_modelo")

FEATURE = "PRECIO_NETO_USD_BBL"
TARGET  = "DELTA_SENS_MBPE"
# Gate Dorado = pareto-10 (directriz 2026-07-09; ver docs/NORTE.md)
GATE_DORADO = ["RUBIALES", "CASTILLA", "CAÑO SUR ESTE", "CASTILLA NORTE", "AKACIAS",
               "CHICHIMENE", "CHICHIMENE SW", "LA CIRA", "CUPIAGUA", "YARIGUI-CANTAGALLO"]
FOCO        = ["AKACIAS", "CUPIAGUA"]   # campos diana del experimento


def _skill(y_true, y_pred, y_naive):
    """SKILL = 1 - MAE_modelo / MAE_naive. >0 => supera a la media ingenua del fold."""
    mae   = mean_absolute_error(y_true, y_pred)
    mae_n = mean_absolute_error(y_true, y_naive)
    return (1 - mae / mae_n) if mae_n > 0.01 else np.nan


def loo_baseline(df_campo: pd.DataFrame) -> tuple[float, float]:
    """M0: isotónica de producción (reales + sintéticos pesados). Devuelve (skill, rango_curva)."""
    df_real = df_campo[~df_campo["ES_SINTETICO"]].reset_index(drop=True)
    df_sint = df_campo[df_campo["ES_SINTETICO"]].reset_index(drop=True)
    n = len(df_real)
    if n < 2:
        return np.nan, np.nan
    xs = df_sint[FEATURE].values if len(df_sint) else np.empty(0)
    ys = df_sint[TARGET].values if len(df_sint) else np.empty(0)
    ws = modelo.pesos_sinteticos_tramo(df_sint, n)[0] if len(df_sint) else np.empty(0)

    y_pred = np.zeros(n); y_naive = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        xr = df_real.loc[mask, FEATURE].values; yr = df_real.loc[mask, TARGET].values
        x_tr = np.concatenate([xs, xr]) if len(xs) else xr
        y_tr = np.concatenate([ys, yr]) if len(ys) else yr
        w_tr = np.concatenate([ws, np.ones(mask.sum())]) if len(ws) else np.ones(mask.sum())
        mod = MotorIsotonico().fit(x_tr, y_tr, sample_weight=w_tr)
        y_pred[i]  = float(mod.predict([df_real.loc[i, FEATURE]])[0])
        y_naive[i] = float(np.mean(yr))
    skill = _skill(df_real[TARGET].values, y_pred, y_naive)

    # Rango de la curva de producción (sensibilidad al precio en la banda observada)
    x_all = df_campo[FEATURE].values; y_all = df_campo[TARGET].values
    es_s  = df_campo["ES_SINTETICO"].values; w_all = np.ones(len(df_campo))
    if es_s.any():
        w_all[es_s] = modelo.pesos_sinteticos_tramo(df_campo[es_s], n)[0]
    mod_full = MotorIsotonico().fit(x_all, y_all, sample_weight=w_all)
    banda = df_real[FEATURE]
    grid = np.linspace(float(banda.min()), float(banda.max()), 50)
    curva = mod_full.predict(grid)
    rango = float(curva.max() - curva.min())
    return skill, rango


def loo_centrado_anio(df_campo: pd.DataFrame) -> tuple[float, float]:
    """
    M1: Delta = media_año + f(Precio Neto centrado por año). f via isotónica sobre reales
    centrados. La predicción de cada fold USA la etiqueta de año del punto de prueba (info
    no disponible en producción) → su SKILL es un techo optimista, útil para el diagnóstico.
    """
    df_real = df_campo[~df_campo["ES_SINTETICO"]].reset_index(drop=True)
    n = len(df_real)
    if n < 2 or "AÑO" not in df_real.columns:
        return np.nan, np.nan
    anios = df_real["AÑO"].values

    y_pred = np.zeros(n); y_naive = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        xr = df_real.loc[mask, FEATURE].values; yr = df_real.loc[mask, TARGET].values
        ar = anios[mask]
        # Medias por año en el train; fallback a media global si el año del test no está
        medias = {a: float(np.mean(yr[ar == a])) for a in np.unique(ar)}
        media_global = float(np.mean(yr))
        yr_centrado = yr - np.array([medias[a] for a in ar])
        f = MotorIsotonico().fit(xr, yr_centrado)
        nivel_i = medias.get(anios[i], media_global)
        y_pred[i]  = nivel_i + float(f.predict([df_real.loc[i, FEATURE]])[0])
        y_naive[i] = media_global
    skill = _skill(df_real[TARGET].values, y_pred, y_naive)

    # Curva de producción re-anclada al año más reciente: media_reciente + f(precio)
    anio_reciente = int(np.max(anios))
    medias_all = {a: float(np.mean(df_real.loc[anios == a, TARGET]))
                  for a in np.unique(anios)}
    yr_centrado_all = (df_real[TARGET].values
                       - np.array([medias_all[a] for a in anios]))
    f_all = MotorIsotonico().fit(df_real[FEATURE].values, yr_centrado_all)
    grid = np.linspace(float(df_real[FEATURE].min()), float(df_real[FEATURE].max()), 50)
    curva = medias_all[anio_reciente] + f_all.predict(grid)
    rango = float(curva.max() - curva.min())
    return skill, rango


if __name__ == "__main__":
    print("=== experimento_vigencia.py — centrado por año vs isotónica directa ===\n")
    ruta = STAGING / "tablon_unico.parquet"
    if not ruta.exists():
        raise FileNotFoundError("Ejecutar 01_etl.py y 02_synthetic.py primero.")
    df = pd.read_parquet(ruta)

    # Pareto = gate dorado + foco + campos materiales con >=6 reales (candidatos a ALTA)
    baselines = (df[(df["ESCENARIO"] == "BASE") & df["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
                 .groupby("CAMPO")["VOLUMEN_1P_OFICIAL_MBPE"].max()
                 .sort_values(ascending=False))
    materiales = list(baselines.index[:20])
    campos = list(dict.fromkeys(GATE_DORADO + FOCO + materiales))

    registros = []
    for campo in campos:
        dfc = modelo.preparar_datos_campo(df, campo)
        n_real = int((~dfc["ES_SINTETICO"]).sum())
        if n_real < 2:
            continue
        skill0, rango0 = loo_baseline(dfc)
        skill1, rango1 = loo_centrado_anio(dfc)
        registros.append({
            "CAMPO": campo, "N_REAL": n_real,
            "SKILL_M0_DIRECTA": round(skill0, 3) if pd.notna(skill0) else None,
            "SKILL_M1_CENTRADO": round(skill1, 3) if pd.notna(skill1) else None,
            "DELTA_SKILL": (round(skill1 - skill0, 3)
                            if pd.notna(skill0) and pd.notna(skill1) else None),
            "RANGO_CURVA_M0_MBPE": round(rango0, 2) if pd.notna(rango0) else None,
            "RANGO_CURVA_M1_MBPE": round(rango1, 2) if pd.notna(rango1) else None,
        })

    res = pd.DataFrame(registros)
    res.to_csv(RESULTADOS / "experimento_vigencia.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    print(f"\n  Guardado: {RESULTADOS / 'experimento_vigencia.csv'}")

    # Lectura del criterio de aceptación
    print(f"\n{'-'*64}\n  Lectura del criterio de aceptación\n{'-'*64}")
    foco = res[res["CAMPO"].isin(FOCO)]
    gate = res[res["CAMPO"].isin(GATE_DORADO)]
    print("  FOCO (AKACIAS/CUPIAGUA):")
    print(foco[["CAMPO", "SKILL_M0_DIRECTA", "SKILL_M1_CENTRADO",
                "RANGO_CURVA_M0_MBPE", "RANGO_CURVA_M1_MBPE"]].to_string(index=False))
    print("\n  GATE DORADO (no debe degradarse):")
    print(gate[["CAMPO", "SKILL_M0_DIRECTA", "SKILL_M1_CENTRADO"]].to_string(index=False))
    print("\n  NOTA: si RANGO_CURVA_M1 sigue ~0, la mejora de SKILL es fuga de la etiqueta")
    print("  de anio (no sensibilidad al precio) y NO debe integrarse a produccion.")
    print("\n=== experimento_vigencia.py — Completado ===")
