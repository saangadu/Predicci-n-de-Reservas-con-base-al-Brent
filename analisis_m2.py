"""
analisis_m2.py — Investigacion M2 por campo: familias Brent->Neto x dataset (OFFLINE)

Directriz usuario 2026-07-15: el Theil-Sen lineal deja campos grandes con R2 < 0.9
(CHICHIMENE 0.77, AKACIAS 0.77, ...) y los Descuentos de Calidad/Transporte estan
aislados del flujo de prediccion desde 2026-06-12. Cada campo puede tener su propia
funcion de correlacion si mejora SU metrica out-of-sample.

BUSQUEDA POR CAMPO en dos ejes cruzados (ver m2_familias.py):
  Eje 1 — familia de funcion: THEILSEN (champion) vs DESCOMPUESTO (usa el Descuento
          de Calidad modelado vs Brent), SEGMENTADA, HUBER, CUADRATICA_MONOTONA.
  Eje 2 — dataset de entrenamiento: solo-HIST (champion, decision 2026-06-12) vs
          HIST + sensibilidades del Consolidado (challenger, directriz de
          selectividad por campo 2026-07-15). Las sensibilidades entran SOLO al
          train; el LOO siempre testea sobre cierres HIST (comparacion justa).

REGLA DE ADOPCION (s12, misma vara que hibrido M1):
  mejora MAE_LOO > 5% sobre el champion (THEILSEN solo-HIST)
  + la curva final pasa las invariantes fisicas (monotonia en $40-120, Neto>=0,
    descuento implicito acotado) — ver m2_familias.pasa_fisica().

SALIDAS (todas en Calidad; Produccion NO se toca):
  resultados_calidad/benchmark_m2.csv          evidencia completa (todas las celdas)
  resultados_calidad/seleccion_metodos_m2.csv  registro de adopcion (ESTADO=ADOPTADO_CALIDAD)
  resultados_calidad/plots_m2/<campo>.png      curva champion vs ganadora
  docs/informe_m2_descuentos.html              informe ejecutivo

El dispatch en el pipeline lo hace 03b_correlacion_brent.py bajo el flag
PRED_M2_SELECCION (auto-ON si PRED_TRACK=calidad via track.FLAGS_ON_EN_CALIDAD;
OFF en Produccion hasta ratificacion).
"""
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import m2_familias as F
import seleccion_reglas as R

BASE = Path(__file__).parent
CAL  = BASE / "resultados_calidad"
PLOTS = CAL / "plots_m2"
PLOTS.mkdir(parents=True, exist_ok=True)

MEJORA_MIN = R.MEJORA_ADOPT   # >5% MAE_LOO vs champion (s12)
N_MIN_HIST = R.N_MIN_HIST_M2
TOP_MATERIALES = 20  # prioridad de reporte en consola


def _datos_campo(df: pd.DataFrame, campo: str) -> dict:
    """Separa los puntos del campo en HIST (cierres, con descuentos) y CONS
    (sensibilidades del Consolidado, sin descuentos)."""
    sub = df[(df["CAMPO"] == campo) & (~df["ES_SINTETICO"])
             & df["BRENT_FLAT_USD_BBL"].notna() & df["PRECIO_NETO_USD_BBL"].notna()]
    hist = sub[sub["ES_BASELINE"]]
    cons = sub[~sub["ES_BASELINE"]]
    return {
        "b_hist": hist["BRENT_FLAT_USD_BBL"].values.astype(float),
        "y_hist": hist["PRECIO_NETO_USD_BBL"].values.astype(float),
        # Descuentos alineados a los puntos HIST (canal de la familia DESCOMPUESTO)
        "extra_hist": {
            "desc_cal": hist["DESCUENTO_CALIDAD_USD_BBL"].values.astype(float),
            "desc_tra": hist["DESCUENTO_TRANSPORTE_USD_BBL"].values.astype(float),
        },
        "b_cons": cons["BRENT_FLAT_USD_BBL"].values.astype(float),
        "y_cons": cons["PRECIO_NETO_USD_BBL"].values.astype(float),
    }


def _evaluar_celda(familia: str, dataset: str, d: dict) -> dict | None:
    """Evalua una celda (familia x dataset): LOO sobre HIST + ajuste final + fisica.
    Retorna dict con metricas y parametros, o None si la celda no es evaluable."""
    usa_cons = dataset == "HIST+CONSOLIDADO"
    if usa_cons and len(d["b_cons"]) == 0:
        return None   # sin sensibilidades el challenger de dataset no existe
    mae_loo, r2_loo = F.loo_familia(
        familia, d["b_hist"], d["y_hist"], d["extra_hist"],
        b_cons=d["b_cons"] if usa_cons else None,
        y_cons=d["y_cons"] if usa_cons else None)
    if pd.isna(mae_loo):
        return None
    # Ajuste FINAL con todo el dataset elegido (lo que se despacharia en 03b)
    b_fit = np.concatenate([d["b_hist"], d["b_cons"]]) if usa_cons else d["b_hist"]
    y_fit = np.concatenate([d["y_hist"], d["y_cons"]]) if usa_cons else d["y_hist"]
    extra_fit = d["extra_hist"]
    if usa_cons:
        extra_fit = {k: np.concatenate([np.asarray(v, dtype=float),
                                        np.full(len(d["b_cons"]), np.nan)])
                     for k, v in d["extra_hist"].items()}
    params = F.ajustar(familia, b_fit, y_fit, extra_fit)
    if params is None:
        return None
    return {
        "MAE_LOO": mae_loo, "R2_LOO": r2_loo, "PARAMS": params,
        "PASA_FISICA": F.pasa_fisica(familia, params),
        "BETA_EQ": F.beta_equivalente(familia, params),
    }


def _plot_campo(campo: str, d: dict, cham: dict, gana: dict,
                fam_g: str, ds_g: str) -> None:
    """Curva champion (Theil-Sen solo-HIST) vs ganadora, con los puntos de ambos
    datasets. Referencia Neto=Brent (descuento cero)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(d["b_hist"], d["y_hist"], color="steelblue", s=60, zorder=5,
               label=f"HIST cierres (n={len(d['b_hist'])})")
    if len(d["b_cons"]):
        ax.scatter(d["b_cons"], d["y_cons"], color="gray", s=25, alpha=0.5,
                   marker="^", zorder=4,
                   label=f"Consolidado sens. (n={len(d['b_cons'])})")
    lo = min(d["b_hist"].min(), 45.0) - 5
    hi = max(d["b_hist"].max(), 90.0) + 10
    grid = np.linspace(lo, hi, 120)
    ax.plot(grid, F.predecir("THEILSEN", cham["PARAMS"], grid), color="darkorange",
            lw=2, label=f"Champion THEILSEN (MAE_LOO={cham['MAE_LOO']:.2f})")
    ax.plot(grid, F.predecir(fam_g, gana["PARAMS"], grid), color="seagreen",
            lw=2, ls="--", label=f"{fam_g} · {ds_g} (MAE_LOO={gana['MAE_LOO']:.2f})")
    ax.plot(grid, grid, color="gray", lw=0.8, alpha=0.5, label="Neto=Brent (desc=0)")
    ax.set_xlabel("Brent Flat (USD/bbl)"); ax.set_ylabel("Precio Aceite (USD/bbl)")
    ax.set_title(f"{campo} — M2 champion vs ganador")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS / f"{campo.replace(' ', '_').replace('/', '_')}.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)


def _informe_html(dfb: pd.DataFrame, dsel: pd.DataFrame, n_campos: int) -> None:
    """Informe ejecutivo con la evidencia del benchmark y las adopciones."""
    ad = dsel.to_html(index=False, float_format=lambda v: f"{v:.3f}") if len(dsel) \
        else "<p>(ninguna celda supero el umbral de adopcion)</p>"
    mejores = dfb[dfb["ES_MEJOR"]].sort_values("BASELINE_MBPE", ascending=False)
    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<title>Investigacion M2 — familias Brent→Neto con Descuento de Calidad</title>
<style>
 body {{ font-family: Segoe UI, sans-serif; margin: 2em auto; max-width: 1100px; color: #222; }}
 table {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
 th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
 th {{ background: #f0f3f5; }} td:first-child, th:first-child {{ text-align: left; }}
 h1 {{ font-size: 1.4em; }} h2 {{ font-size: 1.1em; margin-top: 2em; }}
 .nota {{ background: #fff8e0; padding: 0.8em 1em; border-left: 4px solid #d4a017; }}
</style></head><body>
<h1>Investigacion M2 — familias Brent→Neto con Descuento de Calidad ({date.today()})</h1>
<p class="nota">Regla de adopcion (s12): mejora MAE_LOO &gt; {MEJORA_MIN:.0%} vs champion
(THEILSEN solo-HIST) + invariantes fisicas (monotonia $40–120, Neto≥0, descuento
implicito acotado). El LOO testea SOLO cierres HIST; las sensibilidades del
Consolidado solo pueden entrar al train (comparacion justa entre datasets).
Universo = portafolio con modelo (cobertura EN_PREDICCION). Adopciones =
ESTADO=ADOPTADO_CALIDAD (solo track Calidad bajo PRED_M2_SELECCION).</p>
<h2>Adopciones ({len(dsel)} campos)</h2>
{ad}
<h2>Mejor celda por campo (aunque no llegue al umbral) — {n_campos} campos evaluados</h2>
{mejores[["CAMPO", "BASELINE_MBPE", "FAMILIA", "DATASET", "MAE_LOO", "R2_LOO",
          "MEJORA_PCT", "PASA_FISICA"]].to_html(index=False)}
<h2>Evidencia completa (todas las celdas familia × dataset)</h2>
{dfb.to_html(index=False)}
</body></html>"""
    ruta = BASE / "docs" / "informe_m2_descuentos.html"
    ruta.write_text(html, encoding="utf-8")
    print(f"Informe: {ruta}")


if __name__ == "__main__":
    print("=== analisis_m2.py — Investigacion M2: familias x dataset (portafolio) ===\n")
    # Tablon de PRODUCCION como fuente de datos (los puntos HIST/CONS son identicos
    # en ambos tracks; el analisis es offline y no toca artefactos)
    df = pd.read_parquet(BASE / "datos" / "staging" / "tablon_unico.parquet")
    mtx = pd.read_csv(BASE / "resultados" / "output_matriz_prediccion.csv")
    base_map = mtx.drop_duplicates("CAMPO").set_index("CAMPO")[
        "VOLUMEN_1P_BASELINE_MBPE"].fillna(0.0)

    # Universo s12: portafolio con modelo ∩ N_HIST>=5 (materialidad = baseline)
    portafolio = R.campos_portafolio_con_modelo(BASE)
    hist_n = (df[(~df["ES_SINTETICO"]) & df["ES_BASELINE"]
                 & df["BRENT_FLAT_USD_BBL"].notna() & df["PRECIO_NETO_USD_BBL"].notna()]
              .groupby("CAMPO").size())
    campos = [c for c in portafolio if int(hist_n.get(c, 0)) >= N_MIN_HIST]
    # Campos del portafolio sin N suficiente: quedan fuera del barrido (fallback M2)
    skip_n = [c for c in portafolio if int(hist_n.get(c, 0)) < N_MIN_HIST]
    if skip_n:
        print(f"  [INFO] {len(skip_n)} campos portafolio con N_HIST < {N_MIN_HIST} "
              f"(skip M2, conservan Theil-Sen/fallback): {skip_n[:6]}...")
    print(f"Universo: {len(campos)} campos portafolio con N_HIST >= {N_MIN_HIST}\n")

    bench, adopciones = [], []
    for rank, campo in enumerate(campos, start=1):
        d = _datos_campo(df, campo)
        baseline = float(base_map.get(campo, 0.0))

        # Champion: THEILSEN solo-HIST (la celda de referencia = Produccion actual)
        cham = _evaluar_celda("THEILSEN", "HIST", d)
        if cham is None:
            continue

        celdas = {}
        for fam in F.FAMILIAS_CANDIDATAS:
            for ds in ("HIST", "HIST+CONSOLIDADO"):
                if fam == "THEILSEN" and ds == "HIST":
                    celdas[(fam, ds)] = cham
                    continue
                celdas[(fam, ds)] = _evaluar_celda(fam, ds, d)

        # Mejor celda = menor MAE_LOO entre las que pasan la fisica
        validas = {k: v for k, v in celdas.items()
                   if v is not None and v["PASA_FISICA"]}
        mejor_k = min(validas, key=lambda k: validas[k]["MAE_LOO"], default=None)

        for (fam, ds), v in celdas.items():
            if v is None:
                continue
            mejora = ((cham["MAE_LOO"] - v["MAE_LOO"]) / cham["MAE_LOO"]
                      if cham["MAE_LOO"] > 0 else np.nan)
            es_mejor = (fam, ds) == mejor_k
            # >5% estricto (MEJORA_ADOPT) sobre el champion; no banda de revision
            adoptar = bool(es_mejor and (fam, ds) != ("THEILSEN", "HIST")
                           and R.estado_por_mejora(mejora, checks_ok=True)
                           is not None)
            bench.append({
                "CAMPO": campo, "RANK_MATERIALIDAD": rank,
                "BASELINE_MBPE": round(baseline, 1),
                "N_HIST": len(d["b_hist"]), "N_CONS": len(d["b_cons"]),
                "FAMILIA": fam, "DATASET": ds,
                "MAE_LOO": round(v["MAE_LOO"], 3),
                "R2_LOO": round(v["R2_LOO"], 4) if pd.notna(v["R2_LOO"]) else None,
                "BETA_EQ": round(v["BETA_EQ"], 4),
                "MEJORA_PCT": round(mejora * 100, 1) if pd.notna(mejora) else None,
                "PASA_FISICA": v["PASA_FISICA"],
                "ES_MEJOR": es_mejor, "ADOPTAR": adoptar,
            })
            if adoptar:
                adopciones.append({
                    "CAMPO": campo, "METODO": fam, "DATASET": ds,
                    "BASELINE_MBPE": round(baseline, 1),
                    "MAE_LOO_CHAMPION": round(cham["MAE_LOO"], 3),
                    "MAE_LOO_METODO": round(v["MAE_LOO"], 3),
                    "R2_LOO_METODO": round(v["R2_LOO"], 4) if pd.notna(v["R2_LOO"]) else None,
                    "MEJORA_PCT": round(mejora * 100, 1),
                    "M2_PARAMS": F.params_a_json(v["PARAMS"]),
                    "ESTADO": "ADOPTADO_CALIDAD",
                    "FECHA": str(date.today()),
                })
                _plot_campo(campo, d, cham, v, fam, ds)
                print(f"  [ADOPTAR] {campo:<24} {fam} · {ds}  "
                      f"MAE_LOO {cham['MAE_LOO']:.2f} -> {v['MAE_LOO']:.2f} "
                      f"({mejora:+.0%})")

    dfb = pd.DataFrame(bench)
    dfb.to_csv(CAL / "benchmark_m2.csv", index=False, encoding="utf-8-sig")
    print(f"\nBenchmark: {CAL / 'benchmark_m2.csv'} ({len(dfb)} celdas)")

    dsel = pd.DataFrame(adopciones)
    if len(dsel):
        dsel = dsel.sort_values("BASELINE_MBPE", ascending=False)
    dsel.to_csv(CAL / "seleccion_metodos_m2.csv", index=False, encoding="utf-8-sig")
    print(f"Seleccion: {CAL / 'seleccion_metodos_m2.csv'} ({len(dsel)} adopciones)")

    _informe_html(dfb, dsel, n_campos=len(campos))

    # Resumen top-20 materiales (prioridad del usuario)
    print(f"\n=== Top-{TOP_MATERIALES} materiales: mejor celda por campo ===")
    top = dfb[(dfb["RANK_MATERIALIDAD"] <= TOP_MATERIALES) & dfb["ES_MEJOR"]]
    print(top[["CAMPO", "BASELINE_MBPE", "FAMILIA", "DATASET",
               "MAE_LOO", "R2_LOO", "MEJORA_PCT", "ADOPTAR"]].to_string(index=False))
