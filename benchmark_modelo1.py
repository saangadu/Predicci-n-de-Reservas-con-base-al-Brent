"""
benchmark_modelo1.py — Comparativa de motores 1D para el Modelo 1 (analisis offline)

Responde la pregunta del usuario (2026-06-11): al pasar de 3D (Brent+descuentos) a 1D
(solo Precio Neto), ¿que motor matematico es el mejor para Delta Reservas = f(Precio Neto)?

Compara por campo, con LOO-CV sobre PUNTOS REALES (sinteticos siempre en train, mismos
pesos por nivel de escalera que 03_modelo.py), los 4 candidatos de motores_modelo1.py:
Isotonica, XGBoost1D, Suave (PCHIP), Sigmoide.

Metrica primaria: SKILL = 1 - MAE_modelo / MAE_naive (predictor ingenuo = media de deltas
reales del fold). SKILL>0 => el motor aporta sobre la media. Tambien MAE, RMSE y monotonia.

Salidas:
  resultados/benchmark_modelo1.csv            (una fila por campo×motor)
  resultados/benchmark_modelo1_resumen.csv    (agregado por motor)
  resultados/plots_analisis/benchmark_m1_<campo>.png   (gate dorado)

NO altera el pipeline; es soporte para decidir el motor primario y el de validacion.
"""

import importlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from motores_modelo1 import CANDIDATOS

BASE_DIR   = Path(__file__).parent
STAGING    = BASE_DIR / "datos" / "staging"
RESULTADOS = BASE_DIR / "resultados"
PLOTS_DIR  = RESULTADOS / "plots_analisis"
RESULTADOS.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Reutiliza la preparacion de datos y el esquema de pesos del modelo de produccion
modelo = importlib.import_module("03_modelo")

FEATURE = "PRECIO_NETO_USD_BBL"
TARGET  = "DELTA_SENS_MBPE"
GATE_DORADO = ["CASTILLA", "CASTILLA NORTE", "CASTILLA ESTE", "RUBIALES"]


def loo_cv_motor(df_campo: pd.DataFrame, factory) -> dict:
    """
    LOO-CV de un motor 1D sobre los puntos reales del campo (sinteticos siempre en train).
    Replica el esquema de 03_modelo.loo_cv_solo_reales pero para un motor generico 1D y
    una sola feature (Precio Neto). Retorna metricas sobre los reales.
    """
    df_real = df_campo[~df_campo["ES_SINTETICO"]].reset_index(drop=True)
    df_sint = df_campo[df_campo["ES_SINTETICO"]].reset_index(drop=True)
    n_real = len(df_real)
    if n_real < 2:
        return dict(skill=np.nan, mae=np.nan, rmse=np.nan, r2=np.nan, me=np.nan, mono=np.nan)

    xs = df_sint[FEATURE].values if len(df_sint) else np.empty(0)
    ys = df_sint[TARGET].values if len(df_sint) else np.empty(0)
    ws = modelo.pesos_sinteticos_tramo(df_sint, n_real)[0] if len(df_sint) else np.empty(0)

    y_pred  = np.zeros(n_real)
    y_naive = np.zeros(n_real)
    for i in range(n_real):
        m = np.ones(n_real, dtype=bool); m[i] = False
        xr = df_real.loc[m, FEATURE].values
        yr = df_real.loc[m, TARGET].values
        x_tr = np.concatenate([xs, xr]) if len(xs) else xr
        y_tr = np.concatenate([ys, yr]) if len(ys) else yr
        w_tr = np.concatenate([ws, np.ones(m.sum())]) if len(ws) else np.ones(m.sum())
        mod = factory().fit(x_tr, y_tr, sample_weight=w_tr)
        y_pred[i]  = float(mod.predict([df_real.loc[i, FEATURE]])[0])
        y_naive[i] = float(np.mean(yr))

    y_true = df_real[TARGET].values
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2    = r2_score(y_true, y_pred) if n_real > 1 else np.nan
    me    = float(np.mean(y_pred - y_true))
    mae_n = mean_absolute_error(y_true, y_naive)
    skill = 1 - mae / mae_n if mae_n > 0.01 else np.nan

    # Monotonia: entrenar con todo y verificar la curva final en banda observada
    x_all = df_campo[FEATURE].values
    y_all = df_campo[TARGET].values
    es_s  = df_campo["ES_SINTETICO"].values
    w_all = np.ones(len(df_campo))
    if es_s.any():
        w_all[es_s] = modelo.pesos_sinteticos_tramo(df_campo[es_s], n_real)[0]
    mod_full = factory().fit(x_all, y_all, sample_weight=w_all)
    grid = np.linspace(float(x_all.min()), float(x_all.max()), 100)
    curva = mod_full.predict(grid)
    mono = float(np.all(np.diff(curva) >= -1e-6))
    return dict(skill=skill, mae=mae, rmse=rmse, r2=r2, me=me, mono=mono)


def plot_gate(campo: str, df_campo: pd.DataFrame) -> None:
    """Curvas de los 4 motores sobre los datos del campo (espacio delta vs Precio Neto)."""
    reales = df_campo[~df_campo["ES_SINTETICO"]]
    sint   = df_campo[df_campo["ES_SINTETICO"]]
    n_real = len(reales)
    x_all = df_campo[FEATURE].values
    y_all = df_campo[TARGET].values
    es_s  = df_campo["ES_SINTETICO"].values
    w_all = np.ones(len(df_campo))
    if es_s.any():
        w_all[es_s] = modelo.pesos_sinteticos_tramo(df_campo[es_s], n_real)[0]

    grid = np.linspace(float(x_all.min()) - 2, float(x_all.max()) + 5, 300)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(reales[FEATURE], reales[TARGET], color="steelblue", s=60,
               zorder=5, label="Reales")
    ax.scatter(sint[FEATURE], sint[TARGET], color="tomato", s=20, marker="x",
               alpha=0.5, zorder=4, label=f"Sinteticos (n={len(sint)})")
    colores = {"Isotonica": "darkorange", "XGBoost1D": "green",
               "Suave": "purple", "Sigmoide": "crimson"}
    for nombre, factory in CANDIDATOS.items():
        try:
            mod = factory().fit(x_all, y_all, sample_weight=w_all)
            ax.plot(grid, mod.predict(grid), label=nombre, linewidth=1.6,
                    color=colores.get(nombre))
        except Exception as e:
            print(f"    [WARN] {campo}/{nombre}: {e}")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("Precio Neto (USD/bbl)")
    ax.set_ylabel("Delta Reservas 1P (MBPE)")
    ax.set_title(f"{campo} — Motores 1D candidatos (Modelo 1)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    ruta = PLOTS_DIR / f"benchmark_m1_{campo.replace(' ', '_')}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot: {ruta.name}")


if __name__ == "__main__":
    print("=== benchmark_modelo1.py — Motores 1D (Precio Neto -> Delta) ===\n")
    ruta = STAGING / "tablon_unico.parquet"
    if not ruta.exists():
        raise FileNotFoundError("Ejecutar 01_etl.py y 02_synthetic.py primero.")
    df = pd.read_parquet(ruta)

    campos = sorted(df["CAMPO"].unique())
    registros = []
    for campo in campos:
        df_campo = modelo.preparar_datos_campo(df, campo)
        n_real = int((~df_campo["ES_SINTETICO"]).sum())
        if n_real < 2:
            continue   # sin reales no hay LOO-CV informativo
        for nombre, factory in CANDIDATOS.items():
            try:
                met = loo_cv_motor(df_campo, factory)
            except Exception as e:
                print(f"  [WARN] {campo}/{nombre}: {e}")
                continue
            registros.append({"CAMPO": campo, "MOTOR": nombre, "N_REAL": n_real,
                              **{k: (round(v, 4) if pd.notna(v) else None)
                                 for k, v in met.items()}})

    res = pd.DataFrame(registros)
    res.to_csv(RESULTADOS / "benchmark_modelo1.csv", index=False, encoding="utf-8-sig")
    print(f"  Detalle: {RESULTADOS / 'benchmark_modelo1.csv'}  ({len(res)} filas)")

    # Resumen agregado por motor (mediana de skill/mae_rel y % monotono / % skill>0)
    resumen = (res.groupby("MOTOR")
               .agg(skill_mediano=("skill", "median"),
                    mae_mediano=("mae", "median"),
                    rmse_mediano=("rmse", "median"),
                    pct_skill_pos=("skill", lambda s: float((s > 0).mean())),
                    pct_monotono=("mono", lambda s: float((s >= 1).mean())),
                    n_campos=("CAMPO", "nunique"))
               .round(4).reset_index()
               .sort_values("skill_mediano", ascending=False))
    resumen.to_csv(RESULTADOS / "benchmark_modelo1_resumen.csv", index=False,
                   encoding="utf-8-sig")

    print(f"\n{'='*64}\n  RESUMEN POR MOTOR (mediana sobre {res['CAMPO'].nunique()} campos)\n{'='*64}")
    print(resumen.to_string(index=False))

    # "Mejor motor" por campo segun SKILL (desempate: menor MAE)
    res_ok = res.dropna(subset=["skill"])
    if not res_ok.empty:
        idx = res_ok.sort_values(["skill", "mae"], ascending=[False, True]) \
                    .groupby("CAMPO").head(1)
        print(f"\n  Motor ganador por campo (frecuencia, criterio SKILL):")
        print(idx["MOTOR"].value_counts().to_string())

    print("\n  Generando plots del gate dorado...")
    for campo in GATE_DORADO:
        dfc = modelo.preparar_datos_campo(df, campo)
        if int((~dfc["ES_SINTETICO"]).sum()) >= 1:
            plot_gate(campo, dfc)

    print("\n=== benchmark_modelo1.py — Completado ===")
