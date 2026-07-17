"""analisis_selectivo.py — Dossier por campo + benchmark selectivo de métodos (LOYO).

Ola 1: campos con baseline >= 10 MBPE (excl. FLOREÑA/NARE/INFANTAS, en revisión).
Para cada campo:
  1) Dossier diagnóstico: puntos por vigencia, η², sesgo, solapa, forma del FC
     (BK, equilibrio, perfil BK_P10..P90), planitud del deck, N, arquetipo.
  2) Benchmark selectivo: MAE_LOYO del champion (isotónica + escalera cliff) vs los
     métodos HIPOTETIZADOS para ese campo (no torneo abierto). Regla de adopción:
     mejora MAE_LOYO >= 15% sin violar invariantes.

Fuentes: tablon Producción (escalera cliff) + tablon Calidad (escalera perfil + BK_P).
NO toca Producción. Salidas en resultados_calidad/.
"""
import importlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import motores_modelo1 as M

m3 = importlib.import_module("03_modelo")
FEATURE, TARGET = m3.FEATURE, m3.TARGET

BASE = Path(__file__).parent
CAL  = BASE / "resultados_calidad"
PLOTS = CAL / "plots_seleccion"
PLOTS.mkdir(parents=True, exist_ok=True)

BASELINE_MIN = 10.0
EXCLUIR = {"FLOREÑA", "NARE UNIFICADO", "INFANTAS"}
MEJORA_MIN = 0.15   # 15% de mejora en MAE_LOYO para considerar adopción

tab_prod = pd.read_parquet(BASE / "datos" / "staging" / "tablon_unico.parquet")
tab_cal  = pd.read_parquet(BASE / "datos" / "staging_calidad" / "tablon_unico.parquet")
met_prod = pd.read_csv(BASE / "resultados" / "metricas.csv")
mtx_prod = pd.read_csv(BASE / "resultados" / "output_matriz_prediccion.csv")
mtx_prod = mtx_prod[mtx_prod["MOTOR"] == "Isotonica"]

AX = "#8A968E"
YEAR_COL = {"2024": "#3B6EA5", "2025": "#C77D2E", "2026": "#B0413E"}


# ── helpers de datos por campo ───────────────────────────────────────────────
def reales(tab, campo):
    r = tab[(tab["CAMPO"] == campo) & (~tab["ES_SINTETICO"]) & (~tab["ES_BASELINE"])
            & tab[TARGET].notna() & tab[FEATURE].notna()].copy()
    r["ANIO"] = r["VIGENCIA"].astype(str).str.slice(0, 4)
    return r


def sinteticos(tab, campo):
    return tab[(tab["CAMPO"] == campo) & (tab["ES_SINTETICO"])
               & tab[TARGET].notna() & tab[FEATURE].notna()].copy()


def perfil_dict(campo):
    r = tab_cal[(tab_cal["CAMPO"] == campo)]
    out = {}
    for pct in (10, 25, 50, 75, 90):
        col = f"BK_P{pct}"
        if col in r.columns:
            v = r[col].dropna()
            if len(v):
                out[pct] = float(v.iloc[0])
    return out


def anclas(campo):
    r = tab_prod[tab_prod["CAMPO"] == campo]
    a = r.dropna(subset=["BK_ANCLA_FIN_USD_BBL", "BK_ANCLA_PDP_USD_BBL"], how="all")
    if a.empty:
        return np.nan, np.nan
    f = a.sort_values("VIGENCIA_BREAKEVEN", ascending=False).iloc[0]
    return float(f["BK_ANCLA_FIN_USD_BBL"]), float(f["BK_ANCLA_PDP_USD_BBL"])


def _centrar(real):
    y = real[TARGET]
    grp = real["ANIO"]
    return (y - (grp.map(y.groupby(grp).mean()) - y.mean())).values


# ── LOYO genérico por método ─────────────────────────────────────────────────
def loyo(real, sint, factory, center=False):
    """MAE_LOYO: deja un AÑO fuera, entrena con sintéticos + otros años, predice el año."""
    real = real.reset_index(drop=True)
    y_all = _centrar(real) if center else real[TARGET].values
    anios = real["ANIO"].values
    uni = sorted(set(anios))
    if len(real) < 2 or len(uni) < 2:
        return np.nan
    xs = sint[FEATURE].values if len(sint) else np.empty(0)
    ys = sint[TARGET].values if len(sint) else np.empty(0)
    ws = m3.pesos_sinteticos_tramo(sint, len(real))[0] if len(sint) else np.empty(0)
    errs = []
    for a in uni:
        te = anios == a
        tr = ~te
        if tr.sum() < 2:
            continue
        xt = np.concatenate([xs, real.loc[tr, FEATURE].values]) if len(xs) else real.loc[tr, FEATURE].values
        yt = np.concatenate([ys, y_all[tr]]) if len(ys) else y_all[tr]
        wt = np.concatenate([ws, np.ones(tr.sum())]) if len(ws) else np.ones(tr.sum())
        mo = factory().fit(xt, yt, sample_weight=wt)
        yh = mo.predict(real.loc[te, FEATURE].values)
        errs.extend(np.abs(np.asarray(yh) - real.loc[te, TARGET].values).tolist())
    return float(np.mean(errs)) if errs else np.nan


def loyo_fcperfil(real, perfil, baseline, bk_sup, bk_inf):
    """FCPerfil no ajusta datos → su 'LOYO' = MAE de la curva FC vs todos los reales."""
    if not perfil or not np.isfinite(bk_sup) or not np.isfinite(bk_inf) or bk_sup <= bk_inf:
        return np.nan
    mo = M.MotorFCPerfil(perfil, baseline, bk_sup, bk_inf)
    yh = mo.predict(real[FEATURE].values)
    return float(np.mean(np.abs(yh - real[TARGET].values)))


# ── arquetipo (heurística de naturaleza del campo) ───────────────────────────
def arquetipo(campo, eta2, sesgo, rango_rel, insensible):
    if insensible:
        return "gas/insensible"
    if rango_rel < 0.03:
        return "deck-plano"
    if eta2 >= 0.4 and sesgo > 0.15:
        return "confound-dominante"
    if sesgo > 0.15:
        return "maduro-recuperacion"
    return "sensible-precio"


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_map = (mtx_prod.groupby("CAMPO")["VOLUMEN_1P_BASELINE_MBPE"].first())
    campos = sorted([c for c in base_map[base_map >= BASELINE_MIN].index
                     if c not in EXCLUIR], key=lambda c: -base_map[c])
    print(f"Ola 1: {len(campos)} campos (baseline >= {BASELINE_MIN})\n")

    dossier, bench = [], []
    for campo in campos:
        r_p = reales(tab_prod, campo)
        s_cliff = sinteticos(tab_prod, campo)
        s_perfil = sinteticos(tab_cal, campo)
        if r_p.empty:
            continue
        mrow = met_prod[met_prod["CAMPO"] == campo]
        baseline = float(base_map[campo])
        p_ref = float(mrow["P_REF_USD_BBL"].iloc[0]) if len(mrow) and pd.notna(mrow["P_REF_USD_BBL"].iloc[0]) else np.nan
        bk_sup, bk_inf = anclas(campo)
        per = perfil_dict(campo)

        # diagnóstico
        y = r_p[TARGET]
        gm = y.mean()
        grp = r_p["ANIO"]
        n_anio = grp.nunique()
        ss_tot = float(((y - gm) ** 2).sum())
        eta2 = (sum(len(g) * (g.mean() - gm) ** 2 for _, g in y.groupby(grp)) / ss_tot
                if ss_tot > 0 and n_anio >= 2 else np.nan)
        dref = float(mrow["DELTA_REF_ISO"].iloc[0]) if len(mrow) and pd.notna(mrow["DELTA_REF_ISO"].iloc[0]) else np.nan
        sesgo = abs(dref) / baseline if pd.notna(dref) and baseline > 0 else np.nan
        rango_rel = float(y.max() - y.min()) / baseline if baseline > 0 else np.nan
        insensible = bool(tab_prod[(tab_prod.CAMPO == campo)]["BRENT_INSENSITIVE"].dropna().astype(bool).any()) \
            if "BRENT_INSENSITIVE" in tab_prod.columns else False
        arq = arquetipo(campo, eta2 if pd.notna(eta2) else 0, sesgo if pd.notna(sesgo) else 0,
                        rango_rel if pd.notna(rango_rel) else 1, insensible)
        niv = mtx_prod[mtx_prod.CAMPO == campo]["NIVEL_CONFIANZA"].iloc[0]

        dossier.append({
            "CAMPO": campo, "BASELINE_MBPE": round(baseline, 1), "N_REAL": len(r_p),
            "N_ANIOS": n_anio, "ETA2": round(eta2, 3) if pd.notna(eta2) else None,
            "SESGO": round(sesgo, 3) if pd.notna(sesgo) else None,
            "RANGO_REL": round(rango_rel, 3) if pd.notna(rango_rel) else None,
            "ARQUETIPO": arq, "NIVEL_PROD": niv, "P_REF": round(p_ref, 1) if pd.notna(p_ref) else None,
            "BK_SUP": round(bk_sup, 1) if pd.notna(bk_sup) else None,
            "BK_INF": round(bk_inf, 1) if pd.notna(bk_inf) else None,
            "TIENE_PERFIL": bool(per),
        })

        # métodos HIPOTETIZADOS según arquetipo
        champ = loyo(r_p, s_cliff, M.MotorIsotonico, center=False)
        cand = {"champion": champ}
        cand["perfil"] = loyo(r_p, s_perfil, M.MotorIsotonico, center=False) if len(s_perfil) else np.nan
        cand["HuberIso"] = loyo(r_p, s_cliff, M.MotorHuberIso, center=False)
        if arq in ("confound-dominante",):
            cand["reanclaje"] = loyo(r_p, s_cliff, M.MotorIsotonico, center=True)
        if arq in ("sensible-precio", "maduro-recuperacion"):
            cand["LinealRobusto"] = loyo(r_p, pd.DataFrame(columns=r_p.columns), M.MotorLinealRobusto)
            cand["Sigmoide"] = loyo(r_p, s_cliff, M.MotorSigmoide)
        if arq in ("gas/insensible", "deck-plano", "maduro-recuperacion"):
            cand["FCPerfil"] = loyo_fcperfil(r_p, per, baseline, bk_sup, bk_inf)

        best = min((k for k in cand if k != "champion" and pd.notna(cand[k])),
                   key=lambda k: cand[k], default=None)
        for met, mae in cand.items():
            mejora = (champ - mae) / champ if met != "champion" and pd.notna(mae) and pd.notna(champ) and champ > 0 else np.nan
            bench.append({
                "CAMPO": campo, "BASELINE_MBPE": round(baseline, 1), "ARQUETIPO": arq,
                "METODO": met, "MAE_LOYO": round(mae, 3) if pd.notna(mae) else None,
                "MEJORA_PCT": round(mejora * 100, 1) if pd.notna(mejora) else None,
                "ES_MEJOR": (met == best),
                "ADOPTAR": bool(met == best and pd.notna(mejora) and mejora >= MEJORA_MIN),
            })

        # plot diagnóstico 3 paneles
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        fig.patch.set_alpha(0)
        for ax in axes:
            ax.patch.set_alpha(0)
            for sp in ax.spines.values():
                sp.set_color(AX); sp.set_linewidth(0.6)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.tick_params(colors=AX, labelsize=7.5)
            ax.grid(True, alpha=0.13, color=AX, lw=0.5)
        # P1: deck por año
        for a, g in r_p.groupby("ANIO"):
            axes[0].scatter(g[FEATURE], g[TARGET], s=34, color=YEAR_COL.get(a, AX),
                            edgecolor="white", linewidth=0.5, label=a)
        axes[0].axhline(0, color=AX, ls=":", lw=0.7)
        axes[0].set_title(f"Deck por año  (η²={eta2:.2f})" if pd.notna(eta2) else "Deck por año",
                          fontsize=9, color=AX, loc="left")
        axes[0].legend(fontsize=6.5, frameon=False, labelcolor=AX)
        axes[0].set_ylabel("Δ (MBPE)", color=AX, fontsize=8)
        # P2: curva champion (delta) en producción
        sc = mtx_prod[mtx_prod.CAMPO == campo].sort_values("PRECIO_NETO_EFECTIVO_USD_BBL")
        if not sc.empty:
            axes[1].plot(sc["PRECIO_NETO_EFECTIVO_USD_BBL"],
                         sc["VOLUMEN_1P_PREDICHO_MBPE"] - baseline, color="#B0413E", lw=2)
        axes[1].scatter(r_p[FEATURE], r_p[TARGET], s=20, color=AX, alpha=0.6, zorder=3)
        axes[1].axhline(0, color=AX, ls=":", lw=0.7)
        axes[1].set_title(f"Curva actual  (sesgo {sesgo*100:.0f}%)" if pd.notna(sesgo) else "Curva actual",
                          fontsize=9, color=AX, loc="left")
        axes[1].set_xlabel("Precio Neto", color=AX, fontsize=8)
        # P3: perfil FC
        if per and pd.notna(bk_sup) and pd.notna(bk_inf) and bk_sup > bk_inf:
            fc = M.MotorFCPerfil(per, baseline, bk_sup, bk_inf)
            xx = np.linspace(bk_inf, max(bk_sup, r_p[FEATURE].max()), 60)
            axes[2].plot(xx, fc.predict(xx), color="#1B6535", lw=2)
            axes[2].set_title("Perfil FC (curva financiera)", fontsize=9, color=AX, loc="left")
        else:
            axes[2].text(0.5, 0.5, "sin perfil FC", ha="center", va="center",
                         color=AX, transform=axes[2].transAxes, fontsize=9)
            axes[2].set_title("Perfil FC", fontsize=9, color=AX, loc="left")
        axes[2].axhline(0, color=AX, ls=":", lw=0.7)
        axes[2].set_xlabel("Precio Neto", color=AX, fontsize=8)
        fig.suptitle(f"{campo}   ·   {baseline:.0f} MBPE   ·   {arq}", fontsize=10,
                     color=AX, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(PLOTS / f"{campo.replace(' ', '_').replace('/', '_')}.png",
                    dpi=115, transparent=True, bbox_inches="tight")
        plt.close(fig)

    dfd = pd.DataFrame(dossier)
    dfb = pd.DataFrame(bench)
    dfd.to_csv(CAL / "dossier_campos.csv", index=False, encoding="utf-8-sig")
    dfb.to_csv(CAL / "benchmark_selectivo.csv", index=False, encoding="utf-8-sig")
    print("Dossier:", CAL / "dossier_campos.csv")
    print("Benchmark:", CAL / "benchmark_selectivo.csv")
    print("\n=== Arquetipos ===")
    print(dfd.groupby("ARQUETIPO")["CAMPO"].apply(lambda s: ", ".join(s)).to_string())
    print("\n=== Adopciones candidatas (mejora MAE_LOYO >= 15%) ===")
    ad = dfb[dfb["ADOPTAR"]]
    print(ad[["CAMPO", "BASELINE_MBPE", "ARQUETIPO", "METODO", "MAE_LOYO", "MEJORA_PCT"]].to_string(index=False)
          if len(ad) else "  (ninguna supera el umbral)")
    print("\n=== Mejor método por campo (aunque no llegue a 15%) ===")
    mejores = dfb[dfb["ES_MEJOR"]].sort_values("BASELINE_MBPE", ascending=False)
    print(mejores[["CAMPO", "BASELINE_MBPE", "METODO", "MAE_LOYO", "MEJORA_PCT"]].to_string(index=False))

    # ── Registro de selección (fuente de verdad del dispatch) ────────────────
    # perfil = base global del track Calidad (no requiere override por campo).
    # Directriz usuario 2026-07-12: solo la FAMILIA ESTADISTICA es adoptable —
    # ponderaciones/tratamientos que conservan la isotónica+escalera (re-anclaje,
    # centrado parcial, pesos). Los swaps de motor (FCPerfil/Sigmoide/LinealRobusto/
    # HuberIso) se benchmarkean como evidencia pero NUNCA se adoptan: cambian la
    # forma de la curva, que es semántica de negocio ratificable, no hiperparámetro.
    OVERRIDE = {"reanclaje"}
    HIP = {
        "FCPerfil": "Deck poco informativo / campo maduro: el volumen económico a cada "
                    "precio sale del FC certificado (perfil BK_P10-90), no de una regresión "
                    "con N pequeño. Curva financiera pura, sin sobreajuste posible.",
        "Sigmoide": "Respuesta al precio con saturación suave (piso de abandono + techo "
                    "geológico): la logística de 4 parámetros impone ambas asíntotas físicas "
                    "con menos grados de libertad que la isotónica escalonada.",
        "LinealRobusto": "Respuesta al precio aproximadamente lineal y limpia: Theil-Sen "
                    "(mediana de pendientes) resiste outliers y no inventa escalones donde "
                    "solo hay tendencia; colapsa a plano si la pendiente no es positiva.",
        "HuberIso": "1-2 vigencias atípicas por recertificación: IRLS-Huber baja el peso "
                    "del punto outlier según su residual, sin borrar la vigencia ni centrar "
                    "el año entero (menos agresivo que re-anclaje).",
        "reanclaje": "Confound de vigencia material (η² alto, sesgo>15%): el offset entre "
                    "decks fabrica un escalón de precio falso; centrar por vigencia conserva "
                    "solo la pendiente intra-año (elasticidad real ≈ 0).",
    }
    reg = []
    for _, r in dfb[dfb["ADOPTAR"] & dfb["METODO"].isin(OVERRIDE)].iterrows():
        champ = dfb[(dfb.CAMPO == r.CAMPO) & (dfb.METODO == "champion")]["MAE_LOYO"].iloc[0]
        reg.append({
            "CAMPO": r["CAMPO"], "METODO": r["METODO"], "ARQUETIPO": r["ARQUETIPO"],
            "BASELINE_MBPE": r["BASELINE_MBPE"],
            "MAE_LOYO_CHAMPION": champ, "MAE_LOYO_METODO": r["MAE_LOYO"],
            "MEJORA_PCT": r["MEJORA_PCT"], "HIPOTESIS": HIP.get(r["METODO"], ""),
            "ESTADO": "ADOPTADO", "FECHA": "2026-07-11",
        })
    dreg = pd.DataFrame(reg).sort_values("BASELINE_MBPE", ascending=False)
    dreg.to_csv(CAL / "seleccion_metodos.csv", index=False, encoding="utf-8-sig")
    print("\n=== Registro de selección (overrides del default perfil+isotónica) ===")
    print(dreg[["CAMPO", "METODO", "MEJORA_PCT", "ESTADO"]].to_string(index=False))
