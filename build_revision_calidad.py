"""build_revision_calidad.py — Revisión consolidada del track Calidad (3 cambios).

Los 3 cambios sustanciales de Calidad vs Producción:
  1. Perfil de salida BK_P10–P90  (motor_breakeven → 05 → 01 → 02: valores de la escalera)
  2. Re-anclaje selectivo          (03: LISAMA — centrado de offsets de vigencia)
  3. Bandas de incertidumbre LOYO  (03 residuales → 04: VOL_P10/P90_MBPE)

Revisión pipeline etapa por etapa + impacto de portafolio + campos materiales. Self-contained.
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
CAL = BASE / "resultados_calidad"
OUT = BASE / "docs" / "revision_calidad.html"

AX = "#5C6B62"; C_PROD = "#B0413E"; C_CAL = "#1B6535"; C_BASE = "#B9A24B"
YEAR_COL = {"2024": "#3B6EA5", "2025": "#C77D2E", "2026": "#B0413E"}
NIV_COL = {"ALTA": "#1B6535", "MEDIA": "#B9A24B", "BAJA": "#B0413E",
           "INSENSIBLE_PRECIO": "#7A5EA6", "SIN_MODELO": "#8A968E"}

mp = pd.read_csv(PROD / "output_matriz_prediccion.csv"); mp = mp[mp.MOTOR == "Isotonica"]
mc = pd.read_csv(CAL / "output_matriz_prediccion.csv"); mc = mc[mc.MOTOR == "Isotonica"]
metp = pd.read_csv(PROD / "metricas.csv")
metc = pd.read_csv(CAL / "metricas.csv")
dossier = pd.read_csv(CAL / "dossier_campos.csv")
reg = pd.read_csv(CAL / "seleccion_metodos.csv")
EXCLUIR = {"FLOREÑA", "NARE UNIFICADO", "INFANTAS"}
tablon = pd.read_parquet(BASE / "datos" / "staging" / "tablon_unico.parquet")


def fnum(x, d=2):
    return "—" if pd.isna(x) else f"{x:,.{d}f}"


def _st(ax):
    ax.patch.set_alpha(0)
    for s in ax.spines.values():
        s.set_color(AX); s.set_linewidth(0.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(colors=AX, labelsize=8)
    ax.grid(True, alpha=0.15, color=AX, lw=0.5)


def _b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=132, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def deck_reales(campo):
    r = tablon[(tablon.CAMPO == campo) & (~tablon.ES_SINTETICO) & (~tablon.ES_BASELINE)
               & tablon.DELTA_SENS_MBPE.notna() & tablon.PRECIO_NETO_USD_BBL.notna()].copy()
    r["A"] = r.VIGENCIA.astype(str).str.slice(0, 4)
    return r


def fig_campo(campo, con_banda=True):
    """Curva prod vs cal (+ banda LOYO si aplica) + deck real, en Precio Neto → Vol."""
    sp = mp[mp.CAMPO == campo].sort_values("PRECIO_NETO_EFECTIVO_USD_BBL")
    sc = mc[mc.CAMPO == campo].sort_values("PRECIO_NETO_EFECTIVO_USD_BBL")
    if sp.empty or sc.empty:
        return ""
    bl = float(sc.VOLUMEN_1P_BASELINE_MBPE.iloc[0]) if pd.notna(sc.VOLUMEN_1P_BASELINE_MBPE.iloc[0]) else np.nan
    r = deck_reales(campo)
    fig, ax = plt.subplots(figsize=(6.9, 3.6)); fig.patch.set_alpha(0)
    _st(ax)
    x = sc.PRECIO_NETO_EFECTIVO_USD_BBL
    if con_banda and "VOL_P10_MBPE" in sc.columns and sc.VOL_P10_MBPE.notna().any():
        ax.fill_between(x, sc.VOL_P10_MBPE.astype(float), sc.VOL_P90_MBPE.astype(float),
                        color=C_CAL, alpha=0.16, linewidth=0, label="Banda LOYO P10–P90")
    ax.plot(sp.PRECIO_NETO_EFECTIVO_USD_BBL, sp.VOLUMEN_1P_PREDICHO_MBPE, "-",
            color=C_PROD, lw=2.0, label="Producción", zorder=4)
    ax.plot(x, sc.VOLUMEN_1P_PREDICHO_MBPE, "-", color=C_CAL, lw=2.4, label="Calidad", zorder=5)
    if pd.notna(bl):
        ax.axhline(bl, ls=(0, (4, 3)), color=C_BASE, lw=1.0, label="Baseline 1P")
        for a, g in r.groupby("A"):
            ax.scatter(g.PRECIO_NETO_USD_BBL, g.DELTA_SENS_MBPE + bl, s=42,
                       color=YEAR_COL.get(a, AX), edgecolor="white", linewidth=0.6,
                       zorder=6, label=f"deck {a}")
    ax.set_xlabel("Precio Neto (USD/bbl)", color=AX, fontsize=8.5)
    ax.set_ylabel("Vol 1P (MBPE)", color=AX, fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, labelcolor=AX, loc="best", ncol=2)
    return _b64(fig)


def fig_niveles():
    orden = ["ALTA", "MEDIA", "BAJA", "INSENSIBLE_PRECIO", "SIN_MODELO"]
    np_ = mp.groupby("CAMPO").NIVEL_CONFIANZA.first().value_counts()
    nc_ = mc.groupby("CAMPO").NIVEL_CONFIANZA.first().value_counts()
    fig, ax = plt.subplots(figsize=(8.4, 3.1)); fig.patch.set_alpha(0)
    _st(ax)
    xx = np.arange(len(orden))
    ax.bar(xx - 0.19, [np_.get(o, 0) for o in orden], width=0.36, color=C_PROD, alpha=0.85, label="Producción")
    ax.bar(xx + 0.19, [nc_.get(o, 0) for o in orden], width=0.36, color=C_CAL, alpha=0.85, label="Calidad")
    for i, o in enumerate(orden):
        ax.text(i - 0.19, np_.get(o, 0) + 0.8, str(np_.get(o, 0)), ha="center", fontsize=8, color=C_PROD)
        ax.text(i + 0.19, nc_.get(o, 0) + 0.8, str(nc_.get(o, 0)), ha="center", fontsize=8, color=C_CAL)
    ax.set_xticks(xx); ax.set_xticklabels(orden, fontsize=8.5)
    ax.set_ylabel("# campos", color=AX, fontsize=9)
    ax.legend(fontsize=8.5, frameon=False, labelcolor=AX)
    return _b64(fig)


def fig_portfolio():
    campos = dossier.CAMPO.tolist()
    gp = mp[mp.CAMPO.isin(campos)].groupby("BRENT_USD_BBL").VOLUMEN_1P_PREDICHO_MBPE.sum()
    gc = mc[mc.CAMPO.isin(campos)].groupby("BRENT_USD_BBL").VOLUMEN_1P_PREDICHO_MBPE.sum()
    fig, ax = plt.subplots(figsize=(8.4, 3.3)); fig.patch.set_alpha(0)
    _st(ax)
    ax.plot(gp.index, gp.values, "-", color=C_PROD, lw=2.4, label="Producción")
    ax.plot(gc.index, gc.values, "-", color=C_CAL, lw=2.4, label="Calidad (3 cambios)")
    ax.set_xlabel("Brent (USD/bbl)", color=AX, fontsize=9)
    ax.set_ylabel("Σ Vol 1P ola 1 (MBPE)", color=AX, fontsize=9)
    ax.legend(fontsize=9, frameon=False, labelcolor=AX)
    return _b64(fig)


# ── métricas de portafolio ────────────────────────────────────────────────────
campos = dossier.CAMPO.tolist()
mloyo_p = dict(zip(metp.CAMPO, metp.MAE_LOYO_ISO))
mloyo_c = dict(zip(metc.CAMPO, metc.MAE_LOYO_ISO))
w = {c: float(dossier[dossier.CAMPO == c].BASELINE_MBPE.iloc[0]) for c in campos}


def wavg(m):
    num = den = 0.0
    for c in campos:
        v = m.get(c)
        if v is not None and pd.notna(v):
            num += v * w[c]; den += w[c]
    return num / den if den else np.nan


loyo_p, loyo_c = wavg(mloyo_p), wavg(mloyo_c)
niv_p = mp.groupby("CAMPO").NIVEL_CONFIANZA.first()
niv_c = mc.groupby("CAMPO").NIVEL_CONFIANZA.first()
subieron = [c for c in campos if niv_p.get(c) == "MEDIA" and niv_c.get(c) == "ALTA"]
rean_campo = set(reg[reg.ESTADO == "ADOPTADO"].CAMPO)
n_banda = mc[mc.VOL_P10_MBPE.notna()].CAMPO.nunique() if "VOL_P10_MBPE" in mc.columns else 0

# residuales para ancho de banda
rp = dict(zip(metc.CAMPO, metc.get("LOYO_RESID_P10", pd.Series(dtype=float))))
rq = dict(zip(metc.CAMPO, metc.get("LOYO_RESID_P90", pd.Series(dtype=float))))


def vol_en(m, campo, brent):
    s = m[m.CAMPO == campo]
    if s.empty:
        return np.nan
    i = (s.BRENT_USD_BBL - brent).abs().idxmin()
    return float(s.loc[i, "VOLUMEN_1P_PREDICHO_MBPE"])


# tabla campos materiales
mat_rows = ""
for c in sorted(campos, key=lambda c: -w[c]):
    if c in EXCLUIR:
        continue
    base = w[c]
    metodo = "re-anclaje" if c in rean_campo else ("perfil" if c in dossier.CAMPO.values else "—")
    v60p, v60c = vol_en(mp, c, 60), vol_en(mc, c, 60)
    v80p, v80c = vol_en(mp, c, 80), vol_en(mc, c, 80)
    ancho = (rq.get(c, np.nan) - rp.get(c, np.nan))
    ancho_rel = ancho / base if pd.notna(ancho) and base > 0 else np.nan
    np_n, nc_n = niv_p.get(c), niv_c.get(c)
    niv_txt = (f'<span class="badge" style="background:{NIV_COL.get(np_n,AX)}22;color:{NIV_COL.get(np_n,AX)}">{np_n}</span>'
               + (f' → <span class="badge" style="background:{NIV_COL.get(nc_n,AX)}22;color:{NIV_COL.get(nc_n,AX)}">{nc_n}</span>' if nc_n != np_n else ""))
    mat_rows += (f"<tr><td><b>{c}</b></td><td class='num'>{fnum(base,1)}</td>"
                 f"<td>{metodo}</td>"
                 f"<td class='num'>{fnum(v60p,1)}→{fnum(v60c,1)}</td>"
                 f"<td class='num'>{fnum(v80p,1)}→{fnum(v80c,1)}</td>"
                 f"<td class='num'>±{fnum(ancho/2,1)} ({fnum(ancho_rel*100,0)}%)</td>"
                 f"<td>{niv_txt}</td></tr>")

fig_niv = fig_niveles()
fig_port = fig_portfolio()
fig_perfil = fig_campo("PALAGUA", con_banda=True)     # perfil (deck plano → FC)
fig_reanc = fig_campo("LISAMA UNIFICADO", con_banda=True)   # re-anclaje
fig_banda = fig_campo("CHICHIMENE", con_banda=True)   # banda ancha material


def th(cols):
    return "".join(f"<th>{c}</th>" for c in cols)


HTML = f"""<title>Revisión del track Calidad — 3 cambios consolidados</title>
<style>
:root {{ --bg:#F7F8F6; --panel:#FFF; --ink:#14201A; --muted:#5C6B62; --faint:#8A968E;
  --line:#DDE3DC; --line2:#EBEFEA; --green:#1B6535; --green-w:#E6F0E9; --clay:#B0413E; --hl:#FBF7E9; --accent:#1B6535; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0E1512; --panel:#151E19; --ink:#E8EEE9;
  --muted:#9DA99F; --faint:#7C897F; --line:#26332C; --line2:#1D2822; --green:#5DBE84;
  --green-w:#16271D; --clay:#E08A86; --hl:#211E12; --accent:#5DBE84; }} }}
:root[data-theme="dark"] {{ --bg:#0E1512; --panel:#151E19; --ink:#E8EEE9; --muted:#9DA99F;
  --faint:#7C897F; --line:#26332C; --line2:#1D2822; --green:#5DBE84; --green-w:#16271D;
  --clay:#E08A86; --hl:#211E12; --accent:#5DBE84; }}
:root[data-theme="light"] {{ --bg:#F7F8F6; --panel:#FFF; --ink:#14201A; --muted:#5C6B62;
  --faint:#8A968E; --line:#DDE3DC; --line2:#EBEFEA; --green:#1B6535; --green-w:#E6F0E9;
  --clay:#B0413E; --hl:#FBF7E9; --accent:#1B6535; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); line-height:1.6;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
.wrap {{ max-width:1000px; margin:0 auto; padding:0 24px 90px; }}
header {{ padding:54px 0 28px; border-bottom:2px solid var(--ink); }}
.eyebrow {{ font-size:12px; letter-spacing:.15em; text-transform:uppercase; color:var(--accent); font-weight:700; margin-bottom:12px; }}
h1 {{ font-family:Georgia,serif; font-weight:600; font-size:clamp(29px,4.3vw,43px); line-height:1.12; margin:0 0 14px; text-wrap:balance; }}
.lede {{ font-size:18px; color:var(--muted); max-width:70ch; margin:0; }}
.meta {{ margin-top:18px; font-size:13px; color:var(--faint); display:flex; gap:20px; flex-wrap:wrap; }}
.meta b {{ color:var(--muted); }}
section {{ margin-top:50px; }}
.snum {{ font-family:Georgia,serif; font-size:13px; color:var(--accent); font-weight:700; }}
h2 {{ font-family:Georgia,serif; font-weight:600; font-size:25px; margin:6px 0 8px; text-wrap:balance; }}
h3 {{ font-size:15.5px; margin:22px 0 8px; }}
p {{ max-width:76ch; font-size:14.5px; }}
.sub {{ font-size:12.5px; color:var(--faint); }}
.good {{ color:var(--green); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin:22px 0; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:17px; }}
.card .k {{ font-size:26px; font-family:Georgia,serif; font-weight:600; font-variant-numeric:tabular-nums; }}
.card .l {{ font-size:12.5px; color:var(--muted); margin-top:4px; }}
.card.g .k {{ color:var(--green); }}
.three {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:18px 0; }}
@media (max-width:720px) {{ .three {{ grid-template-columns:1fr; }} }}
.chg {{ background:var(--panel); border:1px solid var(--line); border-top:3px solid var(--accent);
  border-radius:8px; padding:15px 16px; }}
.chg h4 {{ margin:0 0 6px; font-size:14.5px; }} .chg p {{ font-size:13px; margin:0; color:var(--muted); }}
table {{ border-collapse:collapse; width:100%; font-size:13px; margin:12px 0; }}
.scroll {{ overflow-x:auto; }}
th {{ text-align:left; font-size:10.5px; letter-spacing:.04em; text-transform:uppercase; color:var(--muted);
  font-weight:700; padding:8px 10px; border-bottom:1.5px solid var(--line); white-space:nowrap; }}
td {{ padding:7px 10px; border-bottom:1px solid var(--line2); vertical-align:top; }}
td.num,th.num {{ font-variant-numeric:tabular-nums; white-space:nowrap; text-align:right; }}
tbody tr:hover td {{ background:var(--line2); }}
.badge {{ display:inline-block; font-size:10.5px; font-weight:700; padding:1px 7px; border-radius:20px; }}
figure {{ margin:0; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:13px; }}
figure img {{ width:100%; height:auto; display:block; }}
figcaption {{ font-size:12.5px; color:var(--muted); margin-top:9px; line-height:1.5; }}
.figs {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(310px,1fr)); gap:20px; margin:16px 0; }}
.callout {{ border-left:3px solid var(--accent); background:var(--panel); padding:15px 19px;
  border-radius:0 8px 8px 0; margin:18px 0; font-size:14px; }}
.foot {{ margin-top:56px; padding-top:20px; border-top:1px solid var(--line); font-size:13px; color:var(--faint); }}
</style>
<div class="wrap">
<header>
  <div class="eyebrow">Piloto Predicción Reservas 1P · Gerencia de Desarrollo</div>
  <h1>El track Calidad, consolidado: tres cambios y su efecto en el portafolio</h1>
  <p class="lede">Revisión de punta a punta del pipeline con las tres incorporaciones que
  sobrevivieron el filtro estadístico — perfil de salida, re-anclaje y bandas LOYO — y cómo
  mueven el portafolio general y los campos materiales. Producción intacta.</p>
  <div class="meta">
    <span><b>Fecha</b> 2026-07-12</span>
    <span><b>Métrica</b> MAE_LOYO ponderado por materialidad</span>
    <span><b>Gate</b> tests_calidad (invariantes + no-regresión)</span>
  </div>
</header>

<section>
  <div class="snum">01 — Los tres cambios</div>
  <h2>Qué entra a Calidad, y dónde vive en el pipeline</h2>
  <div class="three">
    <div class="chg"><h4>① Perfil de salida BK_P10–P90</h4><p>Reemplaza el acantilado de clase por
    la curva volumétrica del FC certificado. Cambia los VALORES de la escalera sintética, no el
    motor. <b>05 breakeven → 01 ETL → 02 sintéticos.</b></p></div>
    <div class="chg"><h4>② Re-anclaje selectivo</h4><p>Centra los offsets de vigencia donde el
    confound es puro (LISAMA). Ponderación/tratamiento; la isotónica sigue igual. <b>03 modelo.</b></p></div>
    <div class="chg"><h4>③ Bandas de incertidumbre LOYO</h4><p>Cuantiles P10/P90 de los residuales
    de validación cruzada → rango honesto por campo. No cambia la curva. <b>03 residuales → 04 export.</b></p></div>
  </div>
  <div class="cards">
    <div class="card g"><div class="k">{fnum(loyo_p,2)}→{fnum(loyo_c,2)}</div><div class="l">MAE_LOYO ponderado ola 1 (−{fnum((loyo_p-loyo_c)/loyo_p*100,0)}%)</div></div>
    <div class="card g"><div class="k">{len(subieron)}</div><div class="l">campos materiales MEDIA→ALTA</div></div>
    <div class="card"><div class="k">{n_banda}</div><div class="l">campos con banda de incertidumbre</div></div>
    <div class="card"><div class="k">{len(rean_campo)}</div><div class="l">override de re-anclaje (LISAMA)</div></div>
  </div>
</section>

<section>
  <div class="snum">02 — Revisión del pipeline, etapa por etapa</div>
  <h2>Qué toca cada cambio (y qué NO se tocó)</h2>
  <div class="scroll"><table><thead><tr>{th(['Etapa','Archivo','Qué cambia en Calidad','Invariante conservado'])}</tr></thead><tbody>
  <tr><td><b>0.5 Breakeven</b></td><td>motor_breakeven.py, 05</td>
  <td>① calcula perfil BK_P10–P90 del FC (`perfil_salida`); columnas nuevas en breakeven_resultados</td>
  <td>golden RUBIALES/CASTILLA ±0.5; combinación multi-FC</td></tr>
  <tr><td><b>1 ETL</b></td><td>01_etl.py</td>
  <td>① propaga BK_P* del D-PDP al tablón (guard por presencia de columna)</td>
  <td>orden canónico; agregación v3; homologación UNIFICADO</td></tr>
  <tr><td><b>2 Sintéticos</b></td><td>02_synthetic.py</td>
  <td>① escalera = curva de perfil en vez de acantilado (ALERTA=PERFIL_SALIDA)</td>
  <td>guard de banda; hard-zero; monotonía; cap §4.3</td></tr>
  <tr><td><b>3b M2</b></td><td>03b_correlacion_brent.py</td>
  <td>sin cambios (Theil-Sen ya robusto)</td><td>β&gt;0; fallback k·Brent taggeado</td></tr>
  <tr><td><b>3 Modelo</b></td><td>03_modelo.py</td>
  <td>② re-anclaje LISAMA (registro); ③ residuales LOYO P10/P90 a metricas</td>
  <td>ancla C5; LOO/LOYO; isotónica primaria intacta</td></tr>
  <tr><td><b>4 Export</b></td><td>04_pbi_export.py</td>
  <td>③ columnas VOL_P10/P90_MBPE (guard por presencia)</td>
  <td>hard-zero H1; esquema estable en Producción</td></tr>
  <tr><td><b>Gate</b></td><td>tests_calidad/</td>
  <td>invariantes por campo + no-regresión + aislamiento del dispatch</td>
  <td>NORTE = solo Producción (intacto)</td></tr>
  </tbody></table></div>
  <div class="callout"><b>Descartado en la auditoría (con evidencia):</b> swaps de motor
  (plano/recta/sigmoide — cambian la forma), centrado parcial η² (empeora CASABE), pesos por
  calidad del punto (sin efecto), y tuning de pesos sintéticos (son inertes por diseño).</div>
</section>

<section>
  <div class="snum">03 — Impacto en el portafolio</div>
  <h2>Confiabilidad y volumen agregado</h2>
  <figure><img alt="niveles" src="{fig_niv}"><figcaption>Distribución de niveles de confianza
  (portafolio completo): Producción vs Calidad. El movimiento neto es hacia ALTA sin degradar
  ningún campo.</figcaption></figure>
  <figure style="margin-top:16px"><img alt="portafolio" src="{fig_port}"><figcaption>Σ Vol 1P
  de la ola 1 vs Brent. El ancla agregada a BRENT_REF se conserva (Σ pred = Σ baseline); las
  diferencias son de forma donde el deck no respalda sensibilidad.</figcaption></figure>
</section>

<section>
  <div class="snum">04 — Campos materiales</div>
  <h2>Efecto campo por campo (orden de materialidad)</h2>
  <div class="scroll"><table><thead><tr>{th(['Campo','Baseline','Cambio','Vol@Brent60 P→C','Vol@Brent80 P→C','Banda ±(%base)','Confianza'])}</tr></thead>
  <tbody>{mat_rows}</tbody></table></div>
  <p class="sub">Vol P→C = Producción → Calidad. Banda = medio-ancho P10–P90 (incertidumbre LOYO)
  y su % del baseline. La banda angosta acompaña a los ALTA; la ancha, a los MEDIA.</p>
</section>

<section>
  <div class="snum">05 — Un ejemplo por cambio</div>
  <h2>Los tres, vistos en un campo material</h2>
  <div class="figs">
    <figure><img alt="perfil PALAGUA" src="{fig_perfil}"><figcaption><b>① Perfil — PALAGUA</b>:
    deck plano; el perfil del FC reemplaza la rampa isotónica interpolada. Banda angosta
    (predice bien el año no visto).</figcaption></figure>
    <figure><img alt="reanclaje LISAMA" src="{fig_reanc}"><figcaption><b>② Re-anclaje — LISAMA</b>:
    el salto de deck 2026 se centra; la curva deja de sobre-recuperar. Sube a ALTA.</figcaption></figure>
    <figure><img alt="banda CHICHIMENE" src="{fig_banda}"><figcaption><b>③ Banda — CHICHIMENE</b>:
    banda ancha (±5 MBPE) honesta — el campo tiene sensibilidad real pero con incertidumbre de
    vigencia visible.</figcaption></figure>
  </div>
</section>

<div class="foot">
  Piloto Predicción Reservas 1P vs Brent · Soporte analítico interno (NO reemplaza SEC / EcoFaro /
  ARIES / Planning Space). Track Calidad consolidado (perfil + re-anclaje + bandas LOYO), s9
  (2026-07-12). Producción intacta (flags off). Detalle por método: c2_pesos_sinteticos,
  c4_bandas_loyo, informe_seleccion_metodos.
</div>
</div>
"""
OUT.write_text(HTML, encoding="utf-8")
print(f"OK: {OUT} ({len(HTML)//1024} KB)")
print(f"MAE_LOYO pond {loyo_p:.2f}->{loyo_c:.2f} | MEDIA->ALTA {len(subieron)} | "
      f"banda {n_banda} campos | reanclaje {len(rean_campo)}")
