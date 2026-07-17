"""experimento_estadistico.py — A/B de los candidatos ESTADISTICOS (s9, offline).

Familia permitida (directriz usuario 2026-07-12): ponderaciones/tratamientos que conservan
la isotonica+escalera. Tres candidatos, mismo protocolo LOYO honesto (train transformado,
error evaluado contra los deltas REALES sin transformar):

  C1 — Centrado parcial por η² (shrinkage de offsets de vigencia): fraccion ∈
       {0, 0.25, 0.5, η², 1.0} sobre los campos con sesgo>15% y baseline≥5.
  C2 — Pesos sinteticos: masa por nivel = 1.0×real (actual) vs 0.5×real vs decaimiento
       exponencial por distancia a la banda real. Ola 1 completa; el gate dorado no
       puede degradar.
  C3 — Peso reducido (0.5) a puntos reales con NIVEL_DEFINICIONAL ≠ '' (quiebres
       definicionales). Si no hay puntos flaggeados en la ola 1 → N/A documentado.

Salida: resultados_calidad/experimento_estadistico.csv + resumen en consola.
"""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from motores_modelo1 import MotorIsotonico

m3 = importlib.import_module("03_modelo")
FEATURE, TARGET = m3.FEATURE, m3.TARGET

BASE = Path(__file__).parent
CAL = BASE / "resultados_calidad"

BASELINE_MIN = 10.0
EXCLUIR = {"FLOREÑA", "NARE UNIFICADO", "INFANTAS"}
GATE_DORADO = {"RUBIALES", "CASTILLA", "CAÑO SUR ESTE", "CASTILLA NORTE", "AKACIAS",
               "CHICHIMENE", "LA CIRA", "CUPIAGUA", "YARIGUI-CANTAGALLO"}

tab_prod = pd.read_parquet(BASE / "datos" / "staging" / "tablon_unico.parquet")
tab_cal = pd.read_parquet(BASE / "datos" / "staging_calidad" / "tablon_unico.parquet")
met_prod = pd.read_csv(BASE / "resultados" / "metricas.csv")
mtx_prod = pd.read_csv(BASE / "resultados" / "output_matriz_prediccion.csv")
mtx_prod = mtx_prod[mtx_prod["MOTOR"] == "Isotonica"]


def reales(tab, campo):
    r = tab[(tab["CAMPO"] == campo) & (~tab["ES_SINTETICO"]) & (~tab["ES_BASELINE"])
            & tab[TARGET].notna() & tab[FEATURE].notna()].copy()
    r["ANIO"] = r["VIGENCIA"].astype(str).str.slice(0, 4)
    return r.reset_index(drop=True)


def sinteticos(tab, campo):
    return tab[(tab["CAMPO"] == campo) & (tab["ES_SINTETICO"])
               & tab[TARGET].notna() & tab[FEATURE].notna()].copy().reset_index(drop=True)


def loyo_custom(real, sint, y_train_all=None, w_sint=None, w_real=None):
    """LOYO honesto: train = sinteticos(w_sint) + otros años (targets y_train_all si se
    pasan — p.ej. centrados parciales); error vs los deltas RAW del año excluido."""
    y_tr_all = real[TARGET].values if y_train_all is None else np.asarray(y_train_all)
    anios = real["ANIO"].values
    uni = sorted(set(anios))
    if len(real) < 2 or len(uni) < 2:
        return np.nan
    xs = sint[FEATURE].values if len(sint) else np.empty(0)
    ys = sint[TARGET].values if len(sint) else np.empty(0)
    if w_sint is None:
        ws = m3.pesos_sinteticos_tramo(sint, len(real))[0] if len(sint) else np.empty(0)
    else:
        ws = w_sint
    wr = np.ones(len(real)) if w_real is None else np.asarray(w_real)
    errs = []
    for a in uni:
        te = anios == a
        tr = ~te
        if tr.sum() < 2:
            continue
        xt = np.concatenate([xs, real.loc[tr, FEATURE].values]) if len(xs) else real.loc[tr, FEATURE].values
        yt = np.concatenate([ys, y_tr_all[tr]]) if len(ys) else y_tr_all[tr]
        wt = np.concatenate([ws, wr[tr]]) if len(ws) else wr[tr]
        yh = MotorIsotonico().fit(xt, yt, sample_weight=wt).predict(real.loc[te, FEATURE].values)
        errs.extend(np.abs(np.asarray(yh) - real.loc[te, TARGET].values).tolist())
    return float(np.mean(errs)) if errs else np.nan


def eta2_de(real):
    y = real[TARGET]
    grp = real["ANIO"]
    if len(y) < 4 or grp.nunique() < 2:
        return np.nan
    gm = y.mean()
    ss_tot = float(((y - gm) ** 2).sum())
    if ss_tot <= 0:
        return np.nan
    return float(sum(len(g) * (g.mean() - gm) ** 2 for _, g in y.groupby(grp)) / ss_tot)


def centrado(real, fraccion):
    y = real[TARGET]
    grp = real["ANIO"]
    offset = grp.map(y.groupby(grp).mean()) - y.mean()
    return (y - fraccion * offset).values


if __name__ == "__main__":
    base_map = mtx_prod.groupby("CAMPO")["VOLUMEN_1P_BASELINE_MBPE"].first()
    ola1 = sorted([c for c in base_map[base_map >= BASELINE_MIN].index if c not in EXCLUIR],
                  key=lambda c: -base_map[c])
    filas = []

    # ── C1: centrado parcial (campos con sesgo material) ─────────────────────
    print("=== C1 — Centrado parcial por η² (train centrado, error vs RAW) ===")
    sesgo_map = {}
    for _, r in met_prod.iterrows():
        bl, dr = r.get("BASELINE_LATEST"), r.get("DELTA_REF_ISO")
        if pd.notna(bl) and bl > 0 and pd.notna(dr):
            sesgo_map[r["CAMPO"]] = abs(dr) / bl
    cand_c1 = [c for c in ola1 if sesgo_map.get(c, 0) > 0.15 and base_map[c] >= 5.0]
    print(f"  candidatos (sesgo>15%): {cand_c1}")
    for campo in cand_c1:
        r = reales(tab_cal, campo)          # track calidad (escalera perfil)
        s = sinteticos(tab_cal, campo)
        e2 = eta2_de(r)
        base = loyo_custom(r, s)
        for frac, tag in [(0.25, "f=0.25"), (0.5, "f=0.50"),
                          (e2, f"f=eta2({e2:.2f})" if pd.notna(e2) else "f=eta2(nan)"),
                          (1.0, "f=1.00 (total)")]:
            if pd.isna(frac):
                continue
            mae = loyo_custom(r, s, y_train_all=centrado(r, frac))
            mejora = (base - mae) / base * 100 if base and base > 0 else np.nan
            filas.append({"EXPERIMENTO": "C1_centrado_parcial", "CAMPO": campo,
                          "VARIANTE": tag, "MAE_LOYO": round(mae, 3),
                          "MAE_BASE": round(base, 3), "MEJORA_PCT": round(mejora, 1)})
            print(f"  {campo:18s} {tag:16s} MAE={mae:.3f} (base {base:.3f}, {mejora:+.1f}%)")

    # ── C2: variantes de pesos sinteticos (ola 1) ─────────────────────────────
    print("\n=== C2 — Pesos sinteticos: 1.0x (actual) vs 0.5x vs decaimiento ===")
    agg = {"actual": [], "media": [], "decae": []}
    peor_gd = {"media": 0.0, "decae": 0.0}
    for campo in ola1:
        r = reales(tab_cal, campo)
        s = sinteticos(tab_cal, campo)
        if len(r) < 2 or r["ANIO"].nunique() < 2:
            continue
        w_act = m3.pesos_sinteticos_tramo(s, len(r))[0] if len(s) else np.empty(0)
        mae_act = loyo_custom(r, s, w_sint=w_act)
        variantes = {"media": w_act * 0.5}
        if len(s):
            banda_lo = r[FEATURE].min()
            dist = np.maximum(banda_lo - s[FEATURE].values, 0.0)
            variantes["decae"] = w_act * np.exp(-dist / 10.0)
        for tag, w in variantes.items():
            mae = loyo_custom(r, s, w_sint=w)
            mejora = (mae_act - mae) / mae_act * 100 if mae_act and mae_act > 0 else np.nan
            filas.append({"EXPERIMENTO": "C2_pesos_sinteticos", "CAMPO": campo,
                          "VARIANTE": tag, "MAE_LOYO": round(mae, 3),
                          "MAE_BASE": round(mae_act, 3), "MEJORA_PCT": round(mejora, 1)})
            w_c = float(base_map[campo])
            agg[tag].append((mae, w_c))
            if campo in GATE_DORADO and pd.notna(mejora) and mejora < peor_gd.get(tag, 0):
                peor_gd[tag] = mejora
        agg["actual"].append((mae_act, float(base_map[campo])))
    for tag, pares in agg.items():
        pares = [(m, w) for m, w in pares if pd.notna(m)]
        pond = sum(m * w for m, w in pares) / sum(w for _, w in pares) if pares else np.nan
        print(f"  {tag:8s} MAE_LOYO ponderado ola1 = {pond:.3f}"
              + (f"  (peor campo gate dorado: {peor_gd.get(tag, 0):+.1f}%)" if tag != "actual" else ""))

    # ── C3: peso por calidad del punto (NIVEL_DEFINICIONAL) ──────────────────
    print("\n=== C3 — Peso 0.5 a puntos con NIVEL_DEFINICIONAL ===")
    n_flag_total = 0
    for campo in ola1:
        r = reales(tab_cal, campo)
        flag = r["NIVEL_DEFINICIONAL"].fillna("").astype(str).str.len() > 0
        n_flag = int(flag.sum())
        n_flag_total += n_flag
        if n_flag == 0:
            continue
        s = sinteticos(tab_cal, campo)
        base = loyo_custom(r, s)
        w_real = np.where(flag.values, 0.5, 1.0)
        mae = loyo_custom(r, s, w_real=w_real)
        mejora = (base - mae) / base * 100 if base and base > 0 else np.nan
        filas.append({"EXPERIMENTO": "C3_peso_definicional", "CAMPO": campo,
                      "VARIANTE": f"w=0.5 ({n_flag} pts)", "MAE_LOYO": round(mae, 3),
                      "MAE_BASE": round(base, 3), "MEJORA_PCT": round(mejora, 1)})
        print(f"  {campo:18s} {n_flag} pts flaggeados: MAE {base:.3f} -> {mae:.3f} ({mejora:+.1f}%)")
    if n_flag_total == 0:
        print("  N/A — ningun punto real de la ola 1 tiene NIVEL_DEFINICIONAL (los quiebres")
        print("  definicionales conocidos, ej. 2026_REGALIAS, no caen en estos campos/quarters).")
        filas.append({"EXPERIMENTO": "C3_peso_definicional", "CAMPO": "(ola 1)",
                      "VARIANTE": "N/A sin puntos flaggeados", "MAE_LOYO": None,
                      "MAE_BASE": None, "MEJORA_PCT": None})

    df = pd.DataFrame(filas)
    df.to_csv(CAL / "experimento_estadistico.csv", index=False, encoding="utf-8-sig")
    print(f"\nGuardado: {CAL / 'experimento_estadistico.csv'} ({len(df)} filas)")
