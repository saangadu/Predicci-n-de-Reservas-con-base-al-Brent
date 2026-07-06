"""
experimento_normalizacion.py — Experimento offline: ¿normalizar el target reduce el
confound de vigencia? (Modelo 1)

Pregunta del usuario (2026-07-06): el confound de vigencia mezcla el efecto precio con el
salto de deck anual. ¿Normalizar por porcentaje (delta / baseline de su vigencia) reduce la
diferencia entre vigencias y por tanto el confound?

Se comparan TRES targets para la isotónica de M1, sobre los mismos puntos reales:
  M0  DELTA_SENS_MBPE                      (producción actual: delta vs baseline de vigencia)
  Mpct DELTA_SENS / BASELINE_1P_VIGENCIA   (normalización por % que propone el usuario)
  Mvol VOLUMEN_1P_SENSIBILIDAD_MBPE        (volumen absoluto = baseline_vig + delta; equivale
                                            a re-expresar todo contra un baseline común)

Diagnóstico por campo:
  - ETA2 del AÑO sobre cada target (cuánto del target lo explica la vigencia = magnitud del
    confound). Un target mejor tiene ETA2 más bajo O un ETA2 alineado con el precio.
  - SOLAPA de precio entre años ($): invariante a cualquier normalización. Si es ~0 las bandas
    no se cruzan → la sensibilidad intra-año NO es identificable con NINGÚN método.
  - SKILL_LOO medido SIEMPRE en volumen absoluto (comparable entre variantes): se predice el
    volumen de cada punto real dejándolo fuera y se compara contra el predictor ingenuo.
  - RANGO de la curva de volumen en la banda (MBPE): sensibilidad real al precio.

CRITERIO (igual que experimento_vigencia): una variante solo es mejor si sube SKILL/RANGO SIN
fuga de etiqueta de año y SIN degradar el gate dorado.

NO altera el pipeline. Salida: resultados/experimento_normalizacion.csv
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

FEATURE   = "PRECIO_NETO_USD_BBL"
DELTA     = "DELTA_SENS_MBPE"
VOL       = "VOLUMEN_1P_SENSIBILIDAD_MBPE"
BASE_VIG  = "BASELINE_1P_VIGENCIA_MBPE"
GATE_DORADO = ["CASTILLA", "CASTILLA NORTE", "CASTILLA ESTE", "RUBIALES"]
FOCO        = ["AKACIAS", "CUPIAGUA", "RUBIALES"]


def eta2(grupos: np.ndarray, y: np.ndarray) -> float:
    """Fracción de la varianza de y explicada por el grupo (año). 1 = el año lo explica todo."""
    grand = float(np.mean(y))
    ss_tot = float(np.sum((y - grand) ** 2))
    if ss_tot < 1e-9:
        return np.nan
    ss_bet = sum(len(y[grupos == g]) * (float(np.mean(y[grupos == g])) - grand) ** 2
                 for g in np.unique(grupos))
    return ss_bet / ss_tot


def solapa_precio(precio: np.ndarray, anios: np.ndarray) -> float:
    """Mayor solape de banda de precio entre pares de años ($). 0 = bandas separadas."""
    us = np.unique(anios)
    if len(us) < 2:
        return np.nan
    bandas = {a: (float(precio[anios == a].min()), float(precio[anios == a].max())) for a in us}
    mejor = 0.0
    for i, a in enumerate(us):
        for b in us[i + 1:]:
            lo = max(bandas[a][0], bandas[b][0]); hi = min(bandas[a][1], bandas[b][1])
            mejor = max(mejor, hi - lo)
    return mejor


def _skill_vol(vol_true, vol_pred, vol_naive):
    mae = mean_absolute_error(vol_true, vol_pred)
    mae_n = mean_absolute_error(vol_true, vol_naive)
    return (1 - mae / mae_n) if mae_n > 0.01 else np.nan


def loo_vol(reales: pd.DataFrame, target: str, base_latest: float) -> tuple[float, float]:
    """LOO sobre puntos reales; predice el VOLUMEN ABSOLUTO por variante y lo puntúa
    contra el volumen real (comparable entre variantes). Devuelve (skill, rango_curva_vol).

    - target=DELTA: vol_pred = baseline_vigencia_i + f_delta(precio_i)
    - target=PCT:   vol_pred = baseline_vigencia_i * (1 + f_pct(precio_i))
    - target=VOL:   vol_pred = f_vol(precio_i)
    """
    r = reales.reset_index(drop=True)
    n = len(r)
    if n < 2:
        return np.nan, np.nan
    x = r[FEATURE].values
    bvig = r[BASE_VIG].values
    vol_true = r[VOL].values
    if target == "DELTA":
        y = r[DELTA].values
    elif target == "PCT":
        y = (r[DELTA].values / bvig)
    else:
        y = r[VOL].values

    vol_pred = np.zeros(n); vol_naive = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool); m[i] = False
        f = MotorIsotonico().fit(x[m], y[m])
        yi = float(f.predict([x[i]])[0])
        if target == "DELTA":
            vol_pred[i] = bvig[i] + yi
        elif target == "PCT":
            vol_pred[i] = bvig[i] * (1.0 + yi)
        else:
            vol_pred[i] = yi
        vol_naive[i] = float(np.mean(vol_true[m]))
    skill = _skill_vol(vol_true, vol_pred, vol_naive)

    # Curva de volumen sobre la banda observada (sensibilidad real al precio)
    f_all = MotorIsotonico().fit(x, y)
    grid = np.linspace(float(x.min()), float(x.max()), 50)
    yg = f_all.predict(grid)
    if target == "DELTA":
        vol_g = base_latest + yg          # re-anclaje aditivo (como producción)
    elif target == "PCT":
        vol_g = base_latest * (1.0 + yg)  # re-anclaje multiplicativo
    else:
        vol_g = yg
    return skill, float(vol_g.max() - vol_g.min())


if __name__ == "__main__":
    print("=== experimento_normalizacion.py — delta vs delta% vs volumen absoluto ===\n")
    ruta = STAGING / "tablon_unico.parquet"
    if not ruta.exists():
        raise FileNotFoundError("Ejecutar 01_etl.py y 02_synthetic.py primero.")
    df = pd.read_parquet(ruta)

    baselines = (df[(df["ESCENARIO"] == "BASE") & df["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
                 .groupby("CAMPO")["VOLUMEN_1P_OFICIAL_MBPE"].max())
    materiales = list(baselines.sort_values(ascending=False).index[:20])
    campos = list(dict.fromkeys(GATE_DORADO + FOCO + materiales))

    registros = []
    for campo in campos:
        sub = df[(df["CAMPO"] == campo) & (~df["ES_SINTETICO"]) & (~df["ES_BASELINE"])
                 & df[DELTA].notna() & df[FEATURE].notna() & df[BASE_VIG].notna()
                 & df[VOL].notna()].copy()
        n = len(sub)
        n_anios = sub["AÑO"].nunique()
        if n < 4 or n_anios < 2:
            continue
        anios = sub["AÑO"].values
        base_latest = float(baselines.get(campo, np.nan))
        if not np.isfinite(base_latest) or base_latest <= 0:
            continue

        sk_d, rg_d = loo_vol(sub, "DELTA", base_latest)
        sk_p, rg_p = loo_vol(sub, "PCT",   base_latest)
        sk_v, rg_v = loo_vol(sub, "VOL",   base_latest)
        registros.append({
            "CAMPO": campo, "N_REAL": n, "N_ANIOS": n_anios,
            "BASELINE_MBPE": round(base_latest, 1),
            "SOLAPA_PRECIO_USD": round(solapa_precio(sub[FEATURE].values, anios), 2),
            "ETA2_ANIO_DELTA": round(eta2(anios, sub[DELTA].values), 3),
            "ETA2_ANIO_PCT":   round(eta2(anios, (sub[DELTA] / sub[BASE_VIG]).values), 3),
            "ETA2_ANIO_VOL":   round(eta2(anios, sub[VOL].values), 3),
            "SKILL_DELTA": round(sk_d, 3) if pd.notna(sk_d) else None,
            "SKILL_PCT":   round(sk_p, 3) if pd.notna(sk_p) else None,
            "SKILL_VOL":   round(sk_v, 3) if pd.notna(sk_v) else None,
            "RANGO_VOL_DELTA": round(rg_d, 1) if pd.notna(rg_d) else None,
            "RANGO_VOL_PCT":   round(rg_p, 1) if pd.notna(rg_p) else None,
            "RANGO_VOL_VOL":   round(rg_v, 1) if pd.notna(rg_v) else None,
        })

    res = pd.DataFrame(registros)
    res.to_csv(RESULTADOS / "experimento_normalizacion.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print(res.to_string(index=False))
    print(f"\n  Guardado: {RESULTADOS / 'experimento_normalizacion.csv'}")

    print(f"\n{'-'*72}\n  Lectura del criterio\n{'-'*72}")
    val = res[res["SKILL_DELTA"].notna() & res["SKILL_VOL"].notna()]
    if not val.empty:
        mej_pct = (val["SKILL_PCT"] > val["SKILL_DELTA"] + 0.02).sum()
        mej_vol = (val["SKILL_VOL"] > val["SKILL_DELTA"] + 0.02).sum()
        print(f"  Campos donde %-normalización mejora skill (>+0.02): {mej_pct}/{len(val)}")
        print(f"  Campos donde volumen-absoluto mejora skill (>+0.02): {mej_vol}/{len(val)}")
        sep = val[val["SOLAPA_PRECIO_USD"] < 2.0]
        print(f"  Campos con bandas de precio separadas (<$2 solape): {len(sep)}/{len(val)} "
              f"-> sensibilidad intra-ano NO identificable en ninguna variante")
    print("\n  NOTA: SOLAPA~0 es el limite de identificacion; ninguna normalizacion crea")
    print("  variacion de precio donde los anos no se solapan. La %-normalizacion solo")
    print("  reescala; el volumen absoluto re-expresa contra baseline comun (quita el salto).")
    print("\n=== experimento_normalizacion.py — Completado ===")
