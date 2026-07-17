"""
experimento_recencia.py — Experimento offline: ¿pesar más la vigencia más reciente?

Pregunta (2026-07-10): hoy cada punto real del deck pesa 1.0 en el entrenamiento de M1
(la última vigencia 2026_Q1 es 1 de 9 puntos, sin preferencia de recencia). ¿Sube la
calidad de generalización cross-vigencia (MAE_LOYO) si la última vigencia pesa 2× o 3×?

Diseño: LOYO-CV (leave-one-YEAR-out) — la métrica honesta del negocio (predecir el año
no visto = el siguiente quarter). En cada fold se entrena con sintéticos + reales de los
OTROS años; los reales del AÑO más reciente presente en el train reciben el peso de
recencia W. Se compara el MAE_LOYO del gate dorado entre W ∈ {1, 2, 3}.

CRITERIO DE ACEPTACIÓN (para integrar en 03_modelo.py):
  El peso de recencia SOLO se adopta si baja el MAE_LOYO mediano del gate dorado SIN
  degradar campos individuales. N=9 es diminuto: subir el peso de una vigencia amplifica
  su ruido; si no gana claramente, se documenta el resultado y se mantiene peso 1.0.

NO altera el pipeline. Salida: resultados/experimento_recencia.csv
"""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from motores_modelo1 import MotorIsotonico

BASE_DIR   = Path(__file__).parent
STAGING    = BASE_DIR / "datos" / "staging"
RESULTADOS = BASE_DIR / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

modelo = importlib.import_module("03_modelo")

FEATURE = "PRECIO_NETO_USD_BBL"
TARGET  = "DELTA_SENS_MBPE"
# Gate dorado (pareto-9 vigente, MAESTRO / CLAUDE.md)
GATE_DORADO = ["RUBIALES", "CASTILLA", "CAÑO SUR ESTE", "CASTILLA NORTE", "AKACIAS",
               "CHICHIMENE", "LA CIRA", "CUPIAGUA", "YARIGUI-CANTAGALLO"]
PESOS_RECENCIA = [1.0, 2.0, 3.0]


def loyo_con_recencia(df_campo: pd.DataFrame, w_recencia: float) -> float:
    """
    MAE_LOYO isotónico aplicando peso `w_recencia` a los reales del año más reciente
    presente en el train de cada fold. Réplica de 03_modelo.loyo_cv_por_vigencia con
    el peso de recencia añadido. NaN si el campo no tiene ≥2 años reales.
    """
    df_real = df_campo[~df_campo["ES_SINTETICO"]].reset_index(drop=True)
    df_sint = df_campo[df_campo["ES_SINTETICO"]].reset_index(drop=True)
    n_real = len(df_real)
    anios = df_real["VIGENCIA"].astype(str).str.slice(0, 4)
    anios_unicos = sorted(anios.unique())
    if n_real < 2 or len(anios_unicos) < 2:
        return np.nan

    x_sint = df_sint[FEATURE].values if len(df_sint) > 0 else np.empty(0)
    y_sint = df_sint[TARGET].values  if len(df_sint) > 0 else np.empty(0)
    w_sint = modelo.pesos_sinteticos_tramo(df_sint, n_real)[0] if len(df_sint) > 0 else np.empty(0)

    abs_err = []
    for anio in anios_unicos:
        mask_te = (anios == anio).values
        mask_tr = ~mask_te
        if mask_tr.sum() < 2:
            continue
        x_tr = df_real.loc[mask_tr, FEATURE].values
        y_tr = df_real.loc[mask_tr, TARGET].values
        anios_tr = anios[mask_tr]
        # Peso de recencia: reales del año más reciente presente en el train
        anio_reciente_tr = max(anios_tr.unique())
        w_tr_real = np.where(anios_tr.values == anio_reciente_tr, w_recencia, 1.0)

        x_train = np.concatenate([x_sint, x_tr]) if len(x_sint) > 0 else x_tr
        y_train = np.concatenate([y_sint, y_tr]) if len(y_sint) > 0 else y_tr
        w_train = np.concatenate([w_sint, w_tr_real]) if len(w_sint) > 0 else w_tr_real

        x_test = df_real.loc[mask_te, FEATURE].values
        y_test = df_real.loc[mask_te, TARGET].values
        y_hat = MotorIsotonico().fit(x_train, y_train, sample_weight=w_train).predict(x_test)
        abs_err.extend(np.abs(np.asarray(y_hat) - y_test).tolist())

    return float(np.mean(abs_err)) if abs_err else np.nan


if __name__ == "__main__":
    print("=== experimento_recencia.py — peso de recencia en M1 (LOYO) ===\n")
    df = pd.read_parquet(STAGING / "tablon_unico.parquet")

    filas = []
    for campo in sorted(df["CAMPO"].unique()):
        df_campo = modelo.preparar_datos_campo(df, campo)
        if int((~df_campo["ES_SINTETICO"]).sum()) < 2:
            continue
        fila = {"CAMPO": campo, "EN_GATE": campo in GATE_DORADO}
        for w in PESOS_RECENCIA:
            fila[f"MAE_LOYO_W{w:.0f}"] = loyo_con_recencia(df_campo, w)
        filas.append(fila)

    res = pd.DataFrame(filas)
    res.to_csv(RESULTADOS / "experimento_recencia.csv", index=False, encoding="utf-8-sig")

    gate = res[res["EN_GATE"] & res["MAE_LOYO_W1"].notna()]
    print(f"Gate dorado ({len(gate)} campos con LOYO):\n")
    cols = [f"MAE_LOYO_W{w:.0f}" for w in PESOS_RECENCIA]
    print(gate[["CAMPO"] + cols].to_string(index=False))
    print("\nMediana MAE_LOYO gate dorado por peso de recencia:")
    for w in PESOS_RECENCIA:
        print(f"  W={w:.0f}x : {gate[f'MAE_LOYO_W{w:.0f}'].median():.3f} MBPE")

    base = gate["MAE_LOYO_W1"].median()
    print("\nVeredicto:")
    for w in PESOS_RECENCIA[1:]:
        med = gate[f"MAE_LOYO_W{w:.0f}"].median()
        signo = "MEJORA" if med < base - 0.01 else ("EMPATE" if abs(med - base) <= 0.01 else "EMPEORA")
        print(f"  W={w:.0f}x -> {signo} (mediana {med:.3f} vs {base:.3f} base)")
    print("\nCriterio: adoptar solo si MEJORA claro sin degradar campos. "
          "Ver docs/MAESTRO.md §12.")
    print(f"\nSalida: {RESULTADOS / 'experimento_recencia.csv'}")
