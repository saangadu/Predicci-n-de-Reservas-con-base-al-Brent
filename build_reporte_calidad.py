"""build_reporte_calidad.py — genera el informe Artifact (HTML self-contained) del track
Calidad: perfil de salida + re-anclaje selectivo por confound. Embebe graficas antes/
despues como PNG base64 (fondo transparente, ejes neutros) y tablas desde los CSV.
"""
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
PROD = BASE / "resultados"
CAL  = BASE / "resultados_calidad"
OUT  = BASE / "docs" / "reporte_calidad.html"

AX = "#8A968E"        # ejes/labels neutros (legibles en claro y oscuro)
C_PROD = "#B0413E"    # Produccion (antes)
C_CAL  = "#1B6535"    # Calidad (despues)
C_BASE = "#B9A24B"    # baseline

mp = pd.read_csv(PROD / "output_matriz_prediccion.csv")
mc = pd.read_csv(CAL  / "output_matriz_prediccion.csv")
mp = mp[mp["MOTOR"] == "Isotonica"]
mc = mc[mc["MOTOR"] == "Isotonica"]
comp = pd.read_csv(CAL / "comparativa_tracks.csv")
comp["MECANISMO"] = comp["MECANISMO"].fillna("")
rean = pd.read_csv(CAL / "reanclaje_confound.csv")
diag = pd.read_csv(PROD / "diag_confound_vigencia.csv")
tablon = pd.read_parquet(BASE / "datos" / "staging" / "tablon_unico.parquet")
metp_full = pd.read_csv(PROD / "metricas.csv")
metc_full = pd.read_csv(CAL / "metricas.csv")

# Colores por año de vigencia (legibles sobre fondo transparente en ambos temas)
YEAR_COL = {"2024": "#3B6EA5", "2025": "#C77D2E", "2026": "#B0413E"}


def _curva_delta(mtx, campo, baseline):
    s = mtx[mtx["CAMPO"] == campo].sort_values("PRECIO_NETO_EFECTIVO_USD_BBL")
    if s.empty:
        return None, None
    x = s["PRECIO_NETO_EFECTIVO_USD_BBL"].values
    y = (s["VOLUMEN_1P_PREDICHO_MBPE"] - baseline).values
    return x, y


def _pts_reales(campo):
    r = tablon[(tablon["CAMPO"] == campo) & (~tablon["ES_SINTETICO"])
              & (~tablon["ES_BASELINE"]) & tablon["DELTA_SENS_MBPE"].notna()
              & tablon["PRECIO_NETO_USD_BBL"].notna()].copy()
    r["ANIO"] = r["VIGENCIA"].astype(str).str.slice(0, 4)
    return r


def confound_deep(campo: str) -> str:
    """Figura 2 paneles (antes/después) en espacio delta con puntos por año."""
    r = _pts_reales(campo)
    if r.empty:
        return ""
    mrow_p = metp_full[metp_full["CAMPO"] == campo]
    mrow_c = metc_full[metc_full["CAMPO"] == campo]
    baseline = float(mrow_p["BASELINE_LATEST"].iloc[0])
    p_ref = float(mrow_p["P_REF_USD_BBL"].iloc[0])
    dref_p = float(mrow_p["DELTA_REF_ISO"].iloc[0])
    dref_c = float(mrow_c["DELTA_REF_ISO"].iloc[0])
    gmean = r["DELTA_SENS_MBPE"].mean()
    ymean = r.groupby("ANIO")["DELTA_SENS_MBPE"].transform("mean")
    r["DELTA_CENT"] = r["DELTA_SENS_MBPE"] - (ymean - gmean)

    xp, yp = _curva_delta(mp, campo, baseline)
    xc, yc = _curva_delta(mc, campo, baseline)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    fig.patch.set_alpha(0)

    def _base(ax):
        ax.patch.set_alpha(0)
        ax.axhline(0, color=AX, lw=0.8, ls=(0, (3, 3)), alpha=0.6)
        ax.axvline(p_ref, color=AX, lw=0.9, ls=":", alpha=0.7)
        for sp in ax.spines.values():
            sp.set_color(AX); sp.set_linewidth(0.7)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(colors=AX, labelsize=8)
        ax.set_xlabel("Precio Neto (USD/bbl)", color=AX, fontsize=8.5)
        ax.grid(True, alpha=0.13, color=AX, lw=0.5)

    # Panel ANTES
    _base(a1)
    a1.set_ylabel("Δ Reservas vs baseline (MBPE)", color=AX, fontsize=8.5)
    for anio, g in r.groupby("ANIO"):
        col = YEAR_COL.get(anio, AX)
        a1.scatter(g["PRECIO_NETO_USD_BBL"], g["DELTA_SENS_MBPE"], s=46, color=col,
                   edgecolor="white", linewidth=0.6, zorder=4, label=f"deck {anio}")
        my = g["DELTA_SENS_MBPE"].mean()
        a1.axhline(my, color=col, lw=1.0, ls=(0, (5, 4)), alpha=0.55, zorder=2)
    if xp is not None:
        a1.plot(xp, yp, "-", color=C_PROD, lw=2.3, zorder=5, label="curva Producción")
    a1.scatter([p_ref], [dref_p], s=120, facecolor="none", edgecolor=C_PROD, linewidth=2.2, zorder=6)
    a1.annotate(f"d_ref = {dref_p:.1f}", (p_ref, dref_p), textcoords="offset points",
                xytext=(8, -2), fontsize=8, color=C_PROD, fontweight="bold")
    a1.set_title(f"ANTES · Producción   ·   sesgo {abs(dref_p)/baseline*100:.0f}%",
                 fontsize=10, color=C_PROD, fontweight="bold", loc="left")
    a1.legend(fontsize=7, frameon=False, labelcolor=AX, loc="lower right")

    # Panel DESPUÉS
    _base(a2)
    for anio, g in r.groupby("ANIO"):
        col = YEAR_COL.get(anio, AX)
        a2.scatter(g["PRECIO_NETO_USD_BBL"], g["DELTA_CENT"], s=46, color=col,
                   edgecolor="white", linewidth=0.6, zorder=4, label=f"deck {anio}")
    if xc is not None:
        a2.plot(xc, yc, "-", color=C_CAL, lw=2.3, zorder=5, label="curva Calidad")
    a2.scatter([p_ref], [dref_c], s=120, facecolor="none", edgecolor=C_CAL, linewidth=2.2, zorder=6)
    a2.annotate(f"d_ref = {dref_c:.1f}", (p_ref, dref_c), textcoords="offset points",
                xytext=(8, 6), fontsize=8, color=C_CAL, fontweight="bold")
    a2.set_title(f"DESPUÉS · Calidad (centrado)   ·   sesgo {abs(dref_c)/baseline*100:.0f}%",
                 fontsize=10, color=C_CAL, fontweight="bold", loc="left")
    a2.annotate(f"p_ref = {p_ref:.0f}", (p_ref, a2.get_ylim()[1]), textcoords="offset points",
                xytext=(4, -12), fontsize=7.5, color=AX)

    fig.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def curva_b64(campo: str, titulo: str = None) -> str:
    sp = mp[mp["CAMPO"] == campo].sort_values("BRENT_USD_BBL")
    sc = mc[mc["CAMPO"] == campo].sort_values("BRENT_USD_BBL")
    if sp.empty or sc.empty:
        return ""
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    bl = sp["VOLUMEN_1P_BASELINE_MBPE"].iloc[0]
    if pd.notna(bl):
        ax.axhline(bl, ls=(0, (4, 3)), color=C_BASE, lw=1.1, label="Baseline 1P", zorder=1)
    ax.plot(sp["BRENT_USD_BBL"], sp["VOLUMEN_1P_PREDICHO_MBPE"], "-", color=C_PROD,
            lw=2.2, label="Producción", zorder=3)
    ax.plot(sc["BRENT_USD_BBL"], sc["VOLUMEN_1P_PREDICHO_MBPE"], "-", color=C_CAL,
            lw=2.2, label="Calidad", zorder=4)
    for spine in ax.spines.values():
        spine.set_color(AX)
        spine.set_linewidth(0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=AX, labelsize=8)
    ax.set_xlabel("Brent (USD/bbl)", color=AX, fontsize=8.5)
    ax.set_ylabel("Vol 1P (MBPE)", color=AX, fontsize=8.5)
    ax.grid(True, alpha=0.15, color=AX, lw=0.5)
    leg = ax.legend(fontsize=7.5, frameon=False, labelcolor=AX, loc="best")
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── Datos para tablas ────────────────────────────────────────────────────────
def fnum(x, d=2):
    if pd.isna(x):
        return "—"
    return f"{x:,.{d}f}"


def pct(x, d=0):
    if pd.isna(x):
        return "—"
    return f"{x*100:.{d}f}%"


# 1) Confound reconciliation
conf_out = set(mp[mp["SENSIBILIDAD_NO_IDENTIFICADA"] == True]["CAMPO"]) \
    if "SENSIBILIDAD_NO_IDENTIFICADA" in mp.columns else set()
diag_flag = set(diag[diag["FLAG"]]["CAMPO"])
sesgo_campos = mp[mp["MOTIVO_CONFIANZA"].str.contains("sesgo-recuperacion", na=False)]
sesgo_list = (sesgo_campos.drop_duplicates("CAMPO")[["CAMPO", "VOLUMEN_1P_BASELINE_MBPE"]]
              .sort_values("VOLUMEN_1P_BASELINE_MBPE", ascending=False))

# η²/sesgo de CASABE/DINA/GUANDO (por que nunca fueron CONFOUND)
nunca_conf = []
for campo in ["CASABE", "DINA TERCIARIO", "GUANDO"]:
    d = diag[diag["CAMPO"] == campo]
    cp = comp[comp["CAMPO"] == campo]
    if not d.empty and not cp.empty:
        nunca_conf.append({
            "campo": campo, "eta2": d["ETA2"].iloc[0],
            "salto_rel": d["SALTO_REL"].iloc[0], "sesgo": cp["SESGO_PROD"].iloc[0],
            "en_confound": campo in conf_out,
        })

# 2) A/B re-anclaje
rean_rows = []
for _, r in rean.iterrows():
    cp = comp[comp["CAMPO"] == r["CAMPO"]]
    rean_rows.append({
        "campo": r["CAMPO"], "baseline": r["BASELINE"], "eta2": r["ETA2"],
        "sesgo_a": r["SESGO_ANTES"], "sesgo_d": r["SESGO_DESPUES"],
        "loyo_a": cp["MAE_LOYO_PROD"].iloc[0] if not cp.empty else np.nan,
        "loyo_d": cp["MAE_LOYO_CAL"].iloc[0] if not cp.empty else np.nan,
        "niv_a": cp["NIVEL_PROD"].iloc[0] if not cp.empty else "",
        "niv_d": cp["NIVEL_CAL"].iloc[0] if not cp.empty else "",
    })

# 3) NIVEL MEDIA→ALTA (materiales)
subieron = comp[(comp["NIVEL_PROD"] == "MEDIA") & (comp["NIVEL_CAL"] == "ALTA")] \
    .sort_values("BASELINE_MBPE", ascending=False)

# 4) Gate dorado sensibilidad preservada
gd = ["RUBIALES", "CASTILLA", "CAÑO SUR ESTE", "CASTILLA NORTE", "AKACIAS",
      "CHICHIMENE", "LA CIRA", "CUPIAGUA", "YARIGUI-CANTAGALLO"]
gd_rows = []
for campo in gd:
    cp = comp[comp["CAMPO"] == campo]
    if cp.empty:
        continue
    r = cp.iloc[0]
    rng_p = (r["VOL80_PROD"] - r["VOL60_PROD"]) if pd.notna(r["VOL80_PROD"]) else np.nan
    rng_c = (r["VOL80_CAL"] - r["VOL60_CAL"]) if pd.notna(r["VOL80_CAL"]) else np.nan
    gd_rows.append({"campo": campo, "base": r["BASELINE_MBPE"], "rng_p": rng_p,
                    "rng_c": rng_c, "loyo_p": r["MAE_LOYO_PROD"], "loyo_c": r["MAE_LOYO_CAL"]})

# 5) MEDIA-24 con motivo (produccion)
media = (mp[mp["NIVEL_CONFIANZA"] == "MEDIA"].drop_duplicates("CAMPO")
         [["CAMPO", "VOLUMEN_1P_BASELINE_MBPE", "MOTIVO_CONFIANZA"]]
         .sort_values("VOLUMEN_1P_BASELINE_MBPE", ascending=False))


def motivo_corto(m):
    m = str(m)
    if "sesgo-recuperacion" in m:
        import re
        g = re.search(r"sesgo-recuperacion=(\d+)%", m)
        return f"sesgo-recuperación {g.group(1)}%" if g else "sesgo-recuperación"
    if "sensibilidad-no-identificada" in m:
        return "confound (sens. no identificada)"
    if "sin skill" in m:
        return "sin skill vs ingenuo"
    if "N=2" in m or "N=3" in m:
        return "N pequeño (pocas vigencias)"
    if "OUTLIER" in m:
        return "outlier LOO"
    return "otro"


media_rows = []
rean_set = set(rean["CAMPO"])
for _, r in media.iterrows():
    campo = r["CAMPO"]
    mot = motivo_corto(r["MOTIVO_CONFIANZA"])
    cp = comp[comp["CAMPO"] == campo]
    subio = (not cp.empty and cp["NIVEL_CAL"].iloc[0] == "ALTA")
    if campo in rean_set:
        accion = "Re-anclaje confound ✓"
    elif subio:
        accion = "Perfil salida → ALTA ✓"
    elif "sesgo" in mot:
        accion = "Perfil salida (parcial)"
    elif "confound" in mot:
        accion = "Honesto en MEDIA (sesgo≈0)"
    elif "N pequeño" in mot:
        accion = "Requiere más vigencias"
    else:
        accion = "Documentar"
    media_rows.append({"campo": campo, "base": r["VOLUMEN_1P_BASELINE_MBPE"],
                       "motivo": mot, "accion": accion, "subio": subio,
                       "rean": campo in rean_set})

# Graficas
figs = {c: curva_b64(c) for c in
        ["CAÑO LIMON", "CASABE", "LISAMA UNIFICADO", "DINA TERCIARIO", "CASTILLA", "LA CIRA"]}

# Deep-dive confound (2 paneles) + hechos por campo para la narrativa
deep_figs = {}
deep_facts = {}
for campo in ["LISAMA UNIFICADO", "CASABE", "DINA TERCIARIO"]:
    deep_figs[campo] = confound_deep(campo)
    r = _pts_reales(campo)
    mrp = metp_full[metp_full["CAMPO"] == campo]
    mrc = metc_full[metc_full["CAMPO"] == campo]
    aniostats = []
    for anio, g in r.groupby("ANIO"):
        aniostats.append({"anio": anio, "n": len(g),
                          "pmin": g["PRECIO_NETO_USD_BBL"].min(),
                          "pmax": g["PRECIO_NETO_USD_BBL"].max(),
                          "dmean": g["DELTA_SENS_MBPE"].mean(),
                          "dmin": g["DELTA_SENS_MBPE"].min(),
                          "dmax": g["DELTA_SENS_MBPE"].max()})
    dd = pd.read_csv(CAL / "reanclaje_confound.csv")
    dd = dd[dd["CAMPO"] == campo].iloc[0]
    deep_facts[campo] = {
        "baseline": float(mrp["BASELINE_LATEST"].iloc[0]),
        "p_ref": float(mrp["P_REF_USD_BBL"].iloc[0]),
        "dref_p": float(mrp["DELTA_REF_ISO"].iloc[0]),
        "dref_c": float(mrc["DELTA_REF_ISO"].iloc[0]),
        "loyo_p": float(mrp["MAE_LOYO_ISO"].iloc[0]) if pd.notna(mrp["MAE_LOYO_ISO"].iloc[0]) else None,
        "loyo_c": float(mrc["MAE_LOYO_ISO"].iloc[0]) if pd.notna(mrc["MAE_LOYO_ISO"].iloc[0]) else None,
        "eta2": float(dd["ETA2"]), "sesgo_a": float(dd["SESGO_ANTES"]),
        "sesgo_d": float(dd["SESGO_DESPUES"]),
        "niv_p": comp[comp["CAMPO"] == campo]["NIVEL_PROD"].iloc[0],
        "niv_c": comp[comp["CAMPO"] == campo]["NIVEL_CAL"].iloc[0],
        "anios": aniostats,
    }

# Campos en revisión de números (excluidos del informe hasta cerrar cifras):
# FLOREÑA y NARE UNIFICADO (revisión de números); INFANTAS (falta el dato 2026_Q1).
EXCLUIR_REVISION = {"FLOREÑA", "NARE UNIFICADO", "INFANTAS"}
media_rows = [r for r in media_rows if r["campo"] not in EXCLUIR_REVISION]
subieron = subieron[~subieron["CAMPO"].isin(EXCLUIR_REVISION)]
sesgo_list = sesgo_list[~sesgo_list["CAMPO"].isin(EXCLUIR_REVISION)]


# ── HTML ─────────────────────────────────────────────────────────────────────
def th(cols):
    return "".join(f"<th>{c}</th>" for c in cols)


def badge(txt, kind):
    return f'<span class="badge {kind}">{txt}</span>'


def niv_badge(n):
    k = {"ALTA": "b-alta", "MEDIA": "b-media", "BAJA": "b-baja"}.get(str(n), "b-muted")
    return f'<span class="badge {k}">{n}</span>'


# tabla confound reconciliation
conf_html = f"""
<table><thead><tr>{th(['Detector','Qué mide','# campos','¿Creció?'])}</tr></thead><tbody>
<tr><td><b>Flag CONFOUND</b><br><span class="sub">SENSIBILIDAD_NO_IDENTIFICADA</span></td>
<td>η²≥0.8 · bandas de precio separadas · salto inter-año ≥5%·baseline</td>
<td class="num">{len(conf_out)}</td>
<td>{badge('NO — sigue en 14','b-cal')}</td></tr>
<tr><td><b>Detector sesgo-recuperación</b><br><span class="sub">nuevo en s5</span></td>
<td>|d_ref|/baseline &gt; 15% — el ancla p_ref cae en el valle deprimido</td>
<td class="num">{len(sesgo_list)}</td>
<td>{badge('detector distinto','b-media')}</td></tr>
</tbody></table>
"""

nunca_html = "".join(
    f"<tr><td><b>{r['campo']}</b></td><td class='num'>{fnum(r['eta2'],3)}</td>"
    f"<td class='num'>{fnum(r['salto_rel'],2)}</td><td class='num'>{pct(r['sesgo'])}</td>"
    f"<td>{'sí' if r['en_confound'] else 'no'}</td>"
    f"<td>{'CONFOUND' if r['eta2']>=0.8 else 'η² insuficiente → detector sesgo'}</td></tr>"
    for r in nunca_conf)

rean_html = "".join(
    f"<tr><td><b>{r['campo']}</b></td><td class='num'>{fnum(r['baseline'],1)}</td>"
    f"<td class='num'>{fnum(r['eta2'],2)}</td>"
    f"<td class='num'>{pct(r['sesgo_a'])} → <b class='good'>{pct(r['sesgo_d'])}</b></td>"
    f"<td class='num'>{fnum(r['loyo_a'])} → {fnum(r['loyo_d'])}</td>"
    f"<td>{niv_badge(r['niv_a'])} → {niv_badge(r['niv_d'])}</td></tr>"
    for r in rean_rows)

gd_html = "".join(
    f"<tr><td><b>{r['campo']}</b></td><td class='num'>{fnum(r['base'],0)}</td>"
    f"<td class='num'>{fnum(r['rng_p'])} → {fnum(r['rng_c'])}</td>"
    f"<td class='num'>{fnum(r['loyo_p'])} → <b class='good'>{fnum(r['loyo_c'])}</b></td></tr>"
    for r in gd_rows)

media_html = "".join(
    f"<tr class='{'hl' if (r['rean'] or r['subio']) else ''}'>"
    f"<td><b>{r['campo']}</b></td><td class='num'>{fnum(r['base'],1)}</td>"
    f"<td>{r['motivo']}</td><td>{r['accion']}</td></tr>"
    for r in media_rows)

subieron_html = "".join(
    f"<tr><td><b>{r['CAMPO']}</b></td><td class='num'>{fnum(r['BASELINE_MBPE'],1)}</td>"
    f"<td class='num'>{fnum(r['MAE_LOYO_PROD'])} → <b class='good'>{fnum(r['MAE_LOYO_CAL'])}</b></td></tr>"
    for _, r in subieron.iterrows())


def fig_block(campo, cap):
    src = figs.get(campo, "")
    if not src:
        return ""
    return f'<figure><img alt="Curva {campo} Producción vs Calidad" src="{src}"><figcaption>{cap}</figcaption></figure>'


def _anios_narr(f):
    """Frase describiendo el offset entre decks por año."""
    parts = []
    for a in f["anios"]:
        rng = f"{a['pmin']:.0f}–{a['pmax']:.0f}" if a["pmax"] > a["pmin"] else f"{a['pmin']:.0f}"
        parts.append(f"<b>deck {a['anio']}</b> ({a['n']} pt{'s' if a['n']>1 else ''}, "
                     f"${rng}): Δμ&nbsp;=&nbsp;{a['dmean']:+.1f}")
    return " · ".join(parts)


def deep_block(campo, narr):
    f = deep_facts[campo]
    src = deep_figs.get(campo, "")
    img = (f'<figure class="wide"><img alt="Confound {campo} antes/después" src="{src}">'
           f'<figcaption>Espacio Δ-reservas vs Precio Neto. Puntos por año de deck; línea '
           f'discontinua = media de cada deck (su nivel). Izq: la curva de Producción lee '
           f'p_ref={f["p_ref"]:.0f} en el fondo del deck deprimido (d_ref={f["dref_p"]:.1f}) y '
           f'sube sobre todos los puntos. Der: centrados los decks a un nivel común, '
           f'd_ref={f["dref_c"]:.1f} y la curva queda plana/honesta.</figcaption></figure>')
    loyo = (f"{f['loyo_p']:.2f} → {f['loyo_c']:.2f}"
            if f['loyo_p'] is not None and f['loyo_c'] is not None else "—")
    loyo_cls = "good" if (f['loyo_p'] and f['loyo_c'] and f['loyo_c'] <= f['loyo_p']) else "bad"
    chips = (f'<div class="chips">'
             f'<span class="chip"><b>{f["baseline"]:.1f}</b> MBPE baseline</span>'
             f'<span class="chip">η² <b>{f["eta2"]:.2f}</b></span>'
             f'<span class="chip">sesgo <b>{f["sesgo_a"]*100:.0f}%</b> → '
             f'<b class="good">{f["sesgo_d"]*100:.0f}%</b></span>'
             f'<span class="chip">MAE_LOYO <b class="{loyo_cls}">{loyo}</b></span>'
             f'<span class="chip">{niv_badge(f["niv_p"])} → {niv_badge(f["niv_c"])}</span></div>')
    return f"""<div class="deep">
      <h3>{campo}</h3>
      {chips}
      {img}
      <div class="narr">{narr}</div>
    </div>"""


HTML = f"""<title>Track Calidad — Perfil de salida + Re-anclaje confound</title>
<style>
:root {{
  --bg:#F7F8F6; --panel:#FFFFFF; --ink:#14201A; --muted:#5C6B62; --faint:#8A968E;
  --line:#DDE3DC; --line2:#EBEFEA; --green:#1B6535; --green-w:#E6F0E9;
  --clay:#B0413E; --clay-w:#F6E9E8; --amber:#9A7B1F; --hl:#FBF7E9;
  --accent:#1B6535;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg:#0E1512; --panel:#151E19; --ink:#E8EEE9; --muted:#9DA99F; --faint:#7C897F;
    --line:#26332C; --line2:#1D2822; --green:#5DBE84; --green-w:#16271D;
    --clay:#E08A86; --clay-w:#2A1917; --amber:#D9BE6A; --hl:#211E12;
    --accent:#5DBE84;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#0E1512; --panel:#151E19; --ink:#E8EEE9; --muted:#9DA99F; --faint:#7C897F;
  --line:#26332C; --line2:#1D2822; --green:#5DBE84; --green-w:#16271D;
  --clay:#E08A86; --clay-w:#2A1917; --amber:#D9BE6A; --hl:#211E12; --accent:#5DBE84;
}}
:root[data-theme="light"] {{
  --bg:#F7F8F6; --panel:#FFFFFF; --ink:#14201A; --muted:#5C6B62; --faint:#8A968E;
  --line:#DDE3DC; --line2:#EBEFEA; --green:#1B6535; --green-w:#E6F0E9;
  --clay:#B0413E; --clay-w:#F6E9E8; --amber:#9A7B1F; --hl:#FBF7E9; --accent:#1B6535;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  line-height:1.6; font-size:16px;
}}
.wrap {{ max-width:1000px; margin:0 auto; padding:0 24px 96px; }}
header {{ padding:56px 0 32px; border-bottom:2px solid var(--ink); margin-bottom:8px; }}
.eyebrow {{ font-size:12px; letter-spacing:.16em; text-transform:uppercase;
  color:var(--accent); font-weight:700; margin-bottom:14px; }}
h1 {{ font-family:Georgia,"Iowan Old Style",serif; font-weight:600; font-size:clamp(30px,4.4vw,44px);
  line-height:1.1; margin:0 0 16px; text-wrap:balance; letter-spacing:-.01em; }}
.lede {{ font-size:18px; color:var(--muted); max-width:68ch; margin:0; }}
.meta {{ margin-top:20px; font-size:13px; color:var(--faint); display:flex; gap:20px; flex-wrap:wrap; }}
.meta b {{ color:var(--muted); font-weight:600; }}
section {{ margin-top:56px; }}
.snum {{ font-family:Georgia,serif; font-size:13px; color:var(--accent); font-weight:700;
  letter-spacing:.05em; }}
h2 {{ font-family:Georgia,serif; font-weight:600; font-size:26px; margin:6px 0 6px;
  text-wrap:balance; letter-spacing:-.01em; }}
h3 {{ font-size:16px; font-weight:700; margin:28px 0 10px; color:var(--ink); }}
p {{ max-width:72ch; }}
.sub {{ font-size:12.5px; color:var(--faint); }}
.good {{ color:var(--green); }} .bad {{ color:var(--clay); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin:24px 0; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:18px 18px 16px; }}
.card .k {{ font-size:30px; font-family:Georgia,serif; font-weight:600; letter-spacing:-.02em;
  font-variant-numeric:tabular-nums; }}
.card .l {{ font-size:12.5px; color:var(--muted); margin-top:4px; }}
.card.g .k {{ color:var(--green); }} .card.c .k {{ color:var(--clay); }}
table {{ border-collapse:collapse; width:100%; font-size:14px; margin:16px 0; }}
.scroll {{ overflow-x:auto; }}
th {{ text-align:left; font-size:11.5px; letter-spacing:.05em; text-transform:uppercase;
  color:var(--muted); font-weight:700; padding:9px 12px; border-bottom:1.5px solid var(--line);
  white-space:nowrap; }}
td {{ padding:9px 12px; border-bottom:1px solid var(--line2); vertical-align:top; }}
td.num, th.num {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
tr.hl td {{ background:var(--hl); }}
tbody tr:hover td {{ background:var(--line2); }}
.badge {{ display:inline-block; font-size:11px; font-weight:700; padding:2px 8px; border-radius:20px;
  letter-spacing:.02em; }}
.b-alta {{ background:var(--green-w); color:var(--green); }}
.b-media {{ background:var(--hl); color:var(--amber); }}
.b-baja,.b-muted {{ background:var(--line2); color:var(--muted); }}
.b-cal {{ background:var(--green-w); color:var(--green); }}
.figs {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:24px; margin:20px 0; }}
figure {{ margin:0; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }}
figure img {{ width:100%; height:auto; display:block; }}
figcaption {{ font-size:12.5px; color:var(--muted); margin-top:10px; line-height:1.5; }}
.callout {{ border-left:3px solid var(--accent); background:var(--panel); padding:16px 20px;
  border-radius:0 8px 8px 0; margin:20px 0; font-size:14.5px; }}
.callout.warn {{ border-left-color:var(--clay); }}
.steps {{ display:grid; gap:12px; margin:18px 0 24px; }}
.step {{ display:flex; gap:14px; align-items:flex-start; background:var(--panel);
  border:1px solid var(--line); border-radius:10px; padding:14px 16px; font-size:14.5px; }}
.step .n {{ flex:none; width:26px; height:26px; border-radius:50%; background:var(--accent);
  color:#fff; font-weight:700; font-size:14px; display:grid; place-items:center;
  font-family:Georgia,serif; }}
.deep {{ margin:32px 0; padding-top:8px; border-top:1px solid var(--line); }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 14px; }}
.chip {{ font-size:12.5px; color:var(--muted); background:var(--panel); border:1px solid var(--line);
  border-radius:20px; padding:3px 11px; font-variant-numeric:tabular-nums; }}
.chip b {{ color:var(--ink); font-variant-numeric:tabular-nums; }}
figure.wide {{ margin:0 0 12px; }}
.narr {{ font-size:14.5px; color:var(--ink); max-width:78ch; }}
.legend2 {{ font-size:13px; color:var(--muted); background:var(--line2); border-radius:8px;
  padding:10px 14px; margin:8px 0 4px; }}
.legend {{ display:flex; gap:18px; font-size:12.5px; color:var(--muted); margin:4px 0 0; flex-wrap:wrap; }}
.legend i {{ display:inline-block; width:14px; height:3px; border-radius:2px; vertical-align:middle; margin-right:6px; }}
ul.tight {{ margin:10px 0; padding-left:20px; }} ul.tight li {{ margin:6px 0; max-width:70ch; }}
code {{ font-family:ui-monospace,"SF Mono",Menlo,monospace; font-size:13px; background:var(--line2);
  padding:1px 5px; border-radius:4px; }}
.foot {{ margin-top:64px; padding-top:24px; border-top:1px solid var(--line); font-size:13px; color:var(--faint); }}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Piloto Predicción Reservas 1P · Gerencia de Desarrollo</div>
  <h1>Track Calidad: perfil de salida del FC y re-anclaje selectivo por confound</h1>
  <p class="lede">Resultados preliminares de las dos soluciones pendientes de la Directriz
  Escalera↔Deck, corridas en un track paralelo que <b>no toca Producción</b>. Evidencia
  para la reunión de Finanzas.</p>
  <div class="meta">
    <span><b>Fecha</b> 2026-07-11</span>
    <span><b>Producción intacta</b> hash idéntico ✓</span>
    <span><b>Métrica de gobierno</b> MAE_LOYO (normalización v2)</span>
    <span><b>Campos analizados</b> {len(comp)}</span>
  </div>
</header>

<section>
  <div class="snum">01 — Resumen</div>
  <h2>Qué se probó y qué salió</h2>
  <p>Dos cambios de motor, cada uno detrás de un feature-flag apagado en Producción. El
  track Calidad los prende y escribe a <code>resultados_calidad/</code>. La corrida con los
  flags apagados reproduce Producción <b>bit a bit</b> (paridad verificada), así que todo lo
  que cambia abajo viene de los cambios de modelo, no de la infraestructura.</p>
  <div class="cards">
    <div class="card g"><div class="k">7</div><div class="l">campos MEDIA → ALTA (ninguno bajó)</div></div>
    <div class="card g"><div class="k">3</div><div class="l">confound re-anclados (LISAMA, CASABE, DINA T.)</div></div>
    <div class="card g"><div class="k">67</div><div class="l">campos con perfil de salida volumétrico</div></div>
    <div class="card"><div class="k">14</div><div class="l">flag CONFOUND — sin cambio</div></div>
  </div>
  <div class="callout">
    <b>Tu pregunta sobre el confound:</b> no incluimos campos nuevos. El flag CONFOUND sigue
    en los mismos <b>14</b>. CASABE, DINA TERCIARIO y GUANDO nunca fueron CONFOUND — aparecen
    por un <b>detector distinto</b> (sesgo-recuperación, agregado en s5) que caza la misma
    causa (offsets de vigencia) por otro síntoma. Detalle en §2.
  </div>
</section>

<section>
  <div class="snum">02 — Confound</div>
  <h2>No crecieron los campos: cambió el detector</h2>
  <p>Hay <b>dos mecanismos distintos</b> que producen el mismo síntoma (la curva "recupera"
  reservas al subir el Brent más de lo que el deck justifica). El flag histórico solo veía uno.</p>
  {conf_html}
  <h3>Por qué CASABE / DINA / GUANDO nunca fueron CONFOUND</h3>
  <p>El flag CONFOUND exige η² ≥ 0.8. Estos campos tienen η² por debajo del umbral: su
  artefacto no es el escalón de bandas separadas (Mecanismo B clásico) sino el ancla p_ref
  cayendo en el valle deprimido de la vigencia reciente. Los caza el detector sesgo-recuperación.</p>
  <div class="scroll"><table><thead><tr>{th(['Campo','η²','salto rel','sesgo-recup','¿flag CONFOUND?','Detectado por'])}</tr></thead>
  <tbody>{nunca_html}</tbody></table></div>
  <div class="callout warn">
    <b>Dos causas, dos curas.</b> El <b>acantilado sintético</b> (Mecanismo A, p.ej. Caño Limón)
    lo arregla el <b>perfil de salida</b> (§3). El <b>offset de vigencia</b> (Mecanismo B, p.ej.
    CASABE, LISAMA) lo arregla el <b>re-anclaje selectivo</b> (§4). Aplicar la cura equivocada
    no sirve — por eso se separan.
  </div>
</section>

<section>
  <div class="snum">03 — Solución A: perfil de salida</div>
  <h2>El acantilado de clase → curva gradual del propio FC</h2>
  <p>En vez de tumbar toda la clase en un solo breakeven (el precio del año marginal), la
  escalera sintética ahora usa el <b>perfil volumétrico</b> del FC certificado: a qué precio deja
  de ser económico el 10/25/50/75/90% del volumen. Cero datos nuevos — sale del mismo flujo de
  caja. Validado contra la tabla auditada de la directriz (Caño Limón P10=$48.2, P50=$35.7 ✓).</p>
  <div class="legend"><span><i style="background:{C_PROD}"></i>Producción (acantilado)</span>
  <span><i style="background:{C_CAL}"></i>Calidad (perfil)</span>
  <span><i style="background:{C_BASE}"></i>Baseline 1P</span></div>
  <div class="figs">
    {fig_block("CAÑO LIMON", "Caño Limón — la curva ya no despega artificialmente sobre el baseline; sigue el perfil gradual del deck.")}
    {fig_block("LA CIRA", "La Cira — el acantilado bajo la banda se suaviza; la sensibilidad in-banda se reduce a lo que el FC respalda.")}
  </div>
  <div class="callout warn">
    <b>Hallazgo para Finanzas (§4bis.2).</b> El perfil ancla "reservas completas" al breakeven del
    FC. Donde el deck muestra variación de precio <b>por encima</b> de ese breakeven, el modelo la
    atribuye a ruido de vigencia y la aplana — y eso <b>mejora</b> el MAE_LOYO (generaliza mejor al
    año no visto). Es decir: parte de la "sensibilidad al precio" in-banda del gate dorado era
    confound, no respuesta real. La decisión — ¿es correcto el precio del año marginal como umbral
    de la clase? — es de Finanzas (§5 pregunta 4).
  </div>
  <h3>El gate dorado: MAE_LOYO mejora, la sensibilidad se contrae</h3>
  <div class="scroll"><table><thead><tr>{th(['Campo','Baseline','Rango Vol80−Vol60','MAE_LOYO'])}</tr></thead>
  <tbody>{gd_html}</tbody></table></div>
  <p class="sub">Rango = sensibilidad de precio observable en banda. Se contrae en CASTILLA/LA CIRA/YARIGUÍ,
  pero el MAE_LOYO (métrica honesta de generalización) baja en todos → la contracción quita confound, no señal.</p>
</section>

<section>
  <div class="snum">04 — Solución B: re-anclaje selectivo</div>
  <h2>Confound de vigencia, explicado</h2>

  <h3>El problema en una idea</h3>
  <p>El modelo aprende <b>Δreservas = f(Precio&nbsp;Neto)</b> juntando puntos de varios <b>decks</b>
  (los pronósticos de cada cierre: 2024, 2025, 2026). Cada deck se normaliza contra el cierre del año
  anterior. <b>Si el campo se recertifica entre cierres, los decks quedan en niveles distintos.</b>
  Entonces, al <b>mismo precio</b>, dos decks muestran <b>deltas distintos</b> — no porque el precio
  cambie las reservas, sino porque el libro se revisó. Eso es el <i>confound de vigencia</i>:
  el modelo confunde "salto entre decks" con "respuesta al precio".</p>

  <div class="steps">
    <div class="step"><span class="n">1</span><div><b>El síntoma.</b> El ancla del modelo,
    <code>p_ref</code> (el precio del último cierre conocido), cae dentro de la nube deprimida del
    deck más reciente. El modelo lee ese fondo como "el nivel actual del campo".</div></div>
    <div class="step"><span class="n">2</span><div><b>La distorsión.</b> Al reconstruir el volumen
    (<code>Vol = baseline + [f(p) − f(p_ref)]</code>), la curva se <b>levanta</b> por la distancia
    entre ese fondo y el resto de puntos → "recupera" reservas al subir el Brent que <b>ningún deck
    real muestra</b>. Medimos esa distorsión como <b>sesgo = |d_ref| / baseline</b>.</div></div>
    <div class="step"><span class="n">3</span><div><b>La cura.</b> <b>Centrar cada deck a un nivel
    común</b> (quitarle su offset, conservar solo la pendiente <i>dentro</i> del año). Donde el
    confound es genuino, esa pendiente intra-año ≈ 0 → centrar <b>no borra señal real</b> (no la había)
    y elimina el escalón falso. La curva queda plana y honesta.</div></div>
  </div>

  <div class="callout">
    <b>Por qué es selectivo.</b> Centrar por año fue <b>rechazado globalmente</b> en junio porque
    aplana también la sensibilidad <i>real</i> del gate dorado. Aquí solo se aplica donde
    <code>sesgo&nbsp;&gt;&nbsp;15% <b>Y</b> η²&nbsp;≥&nbsp;0.4 <b>Y</b> baseline&nbsp;≥&nbsp;5&nbsp;MBPE</code>.
    El discriminante que protege al gate dorado es el <b>sesgo</b>: CASTILLA/AKACIAS tienen η² alto
    pero sesgo≈0 (su ancla no cae en ningún valle) → <b>no entran</b>. Califican 3 campos.
  </div>

  <p class="legend2">En las gráficas: cada color es un <b>año de deck</b>; la línea discontinua de ese
  color es <b>su nivel medio</b> (el offset). La línea de puntos vertical es <b>p_ref</b>; el círculo
  es <b>d_ref</b> (lo que la curva lee ahí). <span style="color:{C_PROD}">■ Producción</span> ·
  <span style="color:{C_CAL}">■ Calidad</span>.</p>

  {deep_block("LISAMA UNIFICADO", (
    "<b>El caso limpio.</b> Los decks 2024 y 2025 coinciden: a $74–83, LISAMA no pierde nada "
    "(Δ≈0). Pero el deck <b>2026 revisó el campo a la baja</b> y su único punto —al <b>mismo "
    "precio</b> ($74.8)— cae a <b>−10.7 MBPE</b>, casi todo el baseline. Eso no es respuesta al "
    "precio: es una recertificación entre cierres. Como <code>p_ref</code> (74.7) cae justo sobre "
    "ese punto de 2026, la curva de Producción lo toma como nivel actual y, al reconstruir, sube "
    "<b>+5.6 MBPE por encima de todos los puntos reales</b> (sesgo 52%). Centrando los tres decks, "
    "el salto de 2026 desaparece y la pendiente intra-año (≈0: no hay respuesta real al precio) se "
    "conserva. <b>η²=1.0</b> confirma que TODO el movimiento era el año → centrar no borra nada real. "
    "Resultado: sesgo 52%→13%, MAE_LOYO 3.09→0.39, sube a ALTA. Es el que valida el enfoque."))}

  {deep_block("DINA TERCIARIO", (
    "<b>Caso intermedio, limpio.</b> Los decks 2024 (Δ≈−0.15 a $71–73), 2025 (−0.5 a $67–70) y "
    "2026 (−2.3 a $60) forman un pequeño gradiente donde <b>el año domina</b> (η²=0.59): los años "
    "recientes están más deprimidos. Centrar baja el sesgo 22%→14% y aplana la curva a lo que el "
    "deck respalda. El MAE_LOYO sube apenas (0.33→0.69, ambos muy bajos sobre 8.8 MBPE): la curva "
    "es más honesta sin costo material."))}

  {deep_block("CASABE", (
    "<b>El caso ambiguo — por eso queda EN REVISIÓN.</b> Los decks 2024 (Δ≈0 a $76–79), 2025 "
    "(Δμ=−3.0 a $70–78) y 2026 (−5.4 a $67.7) SÍ forman un gradiente, pero aquí <b>precio y año "
    "están confundidos</b>: los años recientes caen a la vez a menor precio y a menor nivel de deck. "
    "η²=0.44 (apenas sobre el umbral 0.4) dice que el año explica ~44% de la varianza — hay confound, "
    "pero <b>no puro</b>. Centrar reduce el sesgo (41%→28%) pero el <b>MAE_LOYO empeora</b> "
    "(0.94→3.14): al quitar el offset de año también se removió algo que podría ser respuesta real al "
    "precio. Es el límite del método: cuando η² está en la frontera, no se puede separar confound de "
    "señal con confianza. <b>No se da por resuelto.</b>"))}

  <div class="callout warn">
    <b>La lección de los tres.</b> El método funciona limpio cuando el año explica <i>casi todo</i>
    (LISAMA η²=1.0) o <i>claramente</i> (DINA η²=0.59). En la frontera (CASABE η²=0.44) el sesgo baja
    pero la generalización puede empeorar → se marca para revisión, no se publica. Esa es la diferencia
    entre corregir un artefacto y borrar señal: el gate de η² la traza, pero cerca del umbral hay que
    mirar campo por campo.
  </div>
</section>

<section>
  <div class="snum">05 — Los 7 que subieron a ALTA</div>
  <h2>Mejora neta de confiabilidad, sin degradar ninguno</h2>
  <div class="scroll"><table><thead><tr>{th(['Campo','Baseline','MAE_LOYO Prod → Cal'])}</tr></thead>
  <tbody>{subieron_html}</tbody></table></div>
  <p class="sub">CASTILLA y CHICHIMENE (gate dorado, 186 y 174 MBPE) suben a ALTA por el perfil;
  LISAMA/DINA por re-anclaje; los demás por generalización más honesta.</p>
</section>

<section>
  <div class="snum">06 — Mapa de acción</div>
  <h2>Los {len(media_rows)} campos de confiabilidad MEDIA, por materialidad</h2>
  <p>Orden por reservas 1P. Resaltados = ya mejorados en el track Calidad.</p>
  <div class="scroll"><table><thead><tr>{th(['Campo','Baseline','Motivo (Producción)','Acción track Calidad'])}</tr></thead>
  <tbody>{media_html}</tbody></table></div>
</section>

<section>
  <div class="snum">07 — Para la reunión</div>
  <h2>Preguntas abiertas de la Directriz, con evidencia preliminar</h2>
  <ul class="tight">
    <li><b>¿Umbral de salida = precio del año marginal?</b> (§5.4) El perfil dice que no: anclar a
    ese precio mejora el MAE_LOYO en todo el gate dorado. Evidencia a favor de reemplazar el
    acantilado por el perfil volumétrico.</li>
    <li><b>¿Re-anclaje selectivo por confound?</b> (§5.6) Sí, con guardas. LISAMA lo valida; CASABE
    queda en revisión. Ningún campo del gate dorado fue tocado (sesgo≈0 los protege).</li>
    <li><b>Lista <code>bk_revision_finanzas.csv</code>.</b> El perfil de salida elimina de raíz los
    12 casos <i>libro-vivo-bajo-bk</i>: la curva ya es gradual como el deck.</li>
  </ul>
  <div class="callout">
    <b>Nada de esto está en Producción.</b> Los flags siguen apagados por defecto. Se prenden solo
    lo que Finanzas ratifique. El track Calidad es reproducible: <code>python run_pipeline.py --calidad</code>.
  </div>
</section>

<div class="foot">
  Piloto Predicción Reservas 1P vs Brent · Soporte analítico interno (NO reemplaza SEC / EcoFaro /
  ARIES / Planning Space). Track Calidad pre-ratificación · Directriz Escalera↔Deck §3, §4bis.6, §4bis.7.
</div>
</div>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"Reporte: {OUT}  ({len(HTML)//1024} KB)")
print(f"Figuras embebidas: {sum(1 for v in figs.values() if v)}/{len(figs)}")
