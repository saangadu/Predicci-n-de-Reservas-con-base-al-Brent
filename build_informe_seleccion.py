"""build_informe_seleccion.py — Informe Artifact s9: auditoría estadística del pipeline.

Narrativa post-rollback (directriz usuario 2026-07-12): los swaps de motor se DESCARTAN
(cambian la forma de la curva, no la ponderación); se conservan re-anclaje (LISAMA) y
perfil de salida BK_P10-P90; se auditan las palancas ESTADISTICAS del pipeline (C1-C3,
A/B LOYO) y se agregan bandas de incertidumbre LOYO (C4) al export Calidad.
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
OUT = BASE / "docs" / "informe_seleccion_metodos.html"

AX = "#8A968E"; C_PROD = "#B0413E"; C_CAL = "#1B6535"; C_BASE = "#B9A24B"
YEAR_COL = {"2024": "#3B6EA5", "2025": "#C77D2E", "2026": "#B0413E"}

dossier = pd.read_csv(CAL / "dossier_campos.csv")
reg = pd.read_csv(CAL / "seleccion_metodos.csv")
exp = pd.read_csv(CAL / "experimento_estadistico.csv")
mp = pd.read_csv(PROD / "output_matriz_prediccion.csv"); mp = mp[mp.MOTOR == "Isotonica"]
mc = pd.read_csv(CAL / "output_matriz_prediccion.csv"); mc = mc[mc.MOTOR == "Isotonica"]
metp = pd.read_csv(PROD / "metricas.csv")
metc = pd.read_csv(CAL / "metricas.csv")
tablon = pd.read_parquet(BASE / "datos" / "staging" / "tablon_unico.parquet")


def fnum(x, d=2):
    return "—" if pd.isna(x) else f"{x:,.{d}f}"


def deck_reales(campo):
    r = tablon[(tablon.CAMPO == campo) & (~tablon.ES_SINTETICO) & (~tablon.ES_BASELINE)
               & tablon.DELTA_SENS_MBPE.notna() & tablon.PRECIO_NETO_USD_BBL.notna()].copy()
    r["A"] = r.VIGENCIA.astype(str).str.slice(0, 4)
    return r


def _estilo(ax):
    ax.patch.set_alpha(0)
    for s in ax.spines.values():
        s.set_color(AX); s.set_linewidth(0.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(colors=AX, labelsize=8)
    ax.grid(True, alpha=0.14, color=AX, lw=0.5)


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def fig_banda(campo):
    """Curva Calidad con banda LOYO P10-P90 + deck real por año."""
    sc = mc[mc.CAMPO == campo].sort_values("PRECIO_NETO_EFECTIVO_USD_BBL")
    if sc.empty or "VOL_P10_MBPE" not in sc.columns or sc["VOL_P10_MBPE"].isna().all():
        return ""
    r = deck_reales(campo)
    bl = float(sc.VOLUMEN_1P_BASELINE_MBPE.iloc[0])
    fig, ax = plt.subplots(figsize=(6.9, 3.6)); fig.patch.set_alpha(0)
    _estilo(ax)
    x = sc.PRECIO_NETO_EFECTIVO_USD_BBL
    ax.fill_between(x, sc.VOL_P10_MBPE.astype(float), sc.VOL_P90_MBPE.astype(float),
                    color=C_CAL, alpha=0.18, linewidth=0, label="Banda LOYO P10–P90")
    ax.plot(x, sc.VOLUMEN_1P_PREDICHO_MBPE, "-", color=C_CAL, lw=2.3, label="Curva Calidad", zorder=5)
    ax.axhline(bl, ls=(0, (4, 3)), color=C_BASE, lw=1.1, label="Baseline 1P")
    for a, g in r.groupby("A"):
        ax.scatter(g.PRECIO_NETO_USD_BBL, g.DELTA_SENS_MBPE + bl, s=44,
                   color=YEAR_COL.get(a, AX), edgecolor="white", linewidth=0.6,
                   zorder=6, label=f"deck {a}")
    ax.set_xlabel("Precio Neto (USD/bbl)", color=AX, fontsize=8.5)
    ax.set_ylabel("Vol 1P (MBPE)", color=AX, fontsize=8.5)
    ax.legend(fontsize=7.2, frameon=False, labelcolor=AX, loc="best", ncol=2)
    return _b64(fig)


def fig_c1_casabe():
    """MAE_LOYO vs fracción de centrado (CASABE y LISAMA): la evidencia del rechazo."""
    fig, ax = plt.subplots(figsize=(6.9, 3.4)); fig.patch.set_alpha(0)
    _estilo(ax)
    for campo, col in [("CASABE", C_PROD), ("LISAMA UNIFICADO", C_CAL)]:
        sub = exp[(exp.EXPERIMENTO == "C1_centrado_parcial") & (exp.CAMPO == campo)]
        if sub.empty:
            continue
        base = float(sub.MAE_BASE.iloc[0])
        fr, mae = [0.0], [base]
        for _, r in sub.iterrows():
            v = r.VARIANTE
            f = (0.25 if "0.25" in v else 0.5 if "0.50" in v else
                 1.0 if "1.00" in v else float(v.split("(")[1].rstrip(")")))
            fr.append(f); mae.append(float(r.MAE_LOYO))
        o = np.argsort(fr)
        ax.plot(np.array(fr)[o], np.array(mae)[o], "o-", color=col, lw=2, ms=5, label=campo)
    ax.set_xlabel("Fracción del offset removida (0 = sin centrar, 1 = centrado total)",
                  color=AX, fontsize=8.5)
    ax.set_ylabel("MAE_LOYO (MBPE)", color=AX, fontsize=8.5)
    ax.legend(fontsize=8, frameon=False, labelcolor=AX)
    return _b64(fig)


def portfolio_curve(campos):
    gp = mp[mp.CAMPO.isin(campos)].groupby("BRENT_USD_BBL").VOLUMEN_1P_PREDICHO_MBPE.sum()
    gc = mc[mc.CAMPO.isin(campos)].groupby("BRENT_USD_BBL").VOLUMEN_1P_PREDICHO_MBPE.sum()
    fig, ax = plt.subplots(figsize=(8.4, 3.6)); fig.patch.set_alpha(0)
    _estilo(ax)
    ax.plot(gp.index, gp.values, "-", color=C_PROD, lw=2.4, label="Producción")
    ax.plot(gc.index, gc.values, "-", color=C_CAL, lw=2.4,
            label="Calidad (perfil + re-anclaje LISAMA)")
    ax.set_xlabel("Brent (USD/bbl)", color=AX, fontsize=9)
    ax.set_ylabel("Σ Vol 1P ola 1 (MBPE)", color=AX, fontsize=9)
    ax.legend(fontsize=9, frameon=False, labelcolor=AX)
    return _b64(fig)


# ── métricas de portafolio (post-rollback) ───────────────────────────────────
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
niv_p = mp[mp.CAMPO.isin(campos)].groupby("CAMPO").NIVEL_CONFIANZA.first()
niv_c = mc[mc.CAMPO.isin(campos)].groupby("CAMPO").NIVEL_CONFIANZA.first()
subieron = [c for c in campos if niv_p.get(c) == "MEDIA" and niv_c.get(c) == "ALTA"]

fig_port = portfolio_curve(campos)
fig_c1 = fig_c1_casabe()
BANDAS_CAMPOS = ["CASTILLA", "CHICHIMENE", "QUIFA", "LISAMA UNIFICADO"]
figs_banda = {c: fig_banda(c) for c in BANDAS_CAMPOS}


def fig_niveles():
    """Distribución de niveles de confianza: Producción vs Calidad (portafolio completo)."""
    orden = ["ALTA", "MEDIA", "BAJA", "INSENSIBLE_PRECIO", "SIN_MODELO"]
    np_ = mp.groupby("CAMPO").NIVEL_CONFIANZA.first().value_counts()
    nc_ = mc.groupby("CAMPO").NIVEL_CONFIANZA.first().value_counts()
    fig, ax = plt.subplots(figsize=(8.2, 3.2)); fig.patch.set_alpha(0)
    _estilo(ax)
    xx = np.arange(len(orden))
    ax.bar(xx - 0.19, [np_.get(o, 0) for o in orden], width=0.36, color=C_PROD,
           alpha=0.85, label="Producción")
    ax.bar(xx + 0.19, [nc_.get(o, 0) for o in orden], width=0.36, color=C_CAL,
           alpha=0.85, label="Calidad (3 cambios)")
    for i, o in enumerate(orden):
        a, b = np_.get(o, 0), nc_.get(o, 0)
        ax.text(i - 0.19, a + 0.8, str(a), ha="center", fontsize=8, color=C_PROD)
        ax.text(i + 0.19, b + 0.8, str(b), ha="center", fontsize=8, color=C_CAL)
    ax.set_xticks(xx); ax.set_xticklabels(orden, fontsize=8.5)
    ax.set_ylabel("# campos", color=AX, fontsize=9)
    ax.legend(fontsize=8.5, frameon=False, labelcolor=AX)
    return _b64(fig)


def th(cols):
    return "".join(f"<th>{c}</th>" for c in cols)


# tabla rechazados
rech = reg[reg.ESTADO == "RECHAZADO"].sort_values("BASELINE_MBPE", ascending=False)
rech_html = "".join(
    f"<tr><td><b>{r.CAMPO}</b></td><td>{r.METODO}</td>"
    f"<td class='num good'>+{fnum(r.MEJORA_PCT,0)}%</td>"
    f"<td class='bad'>RECHAZADO</td></tr>"
    for _, r in rech.iterrows())

# tabla C1
c1 = exp[exp.EXPERIMENTO == "C1_centrado_parcial"]
c1_html = "".join(
    f"<tr><td><b>{r.CAMPO}</b></td><td>{r.VARIANTE}</td>"
    f"<td class='num'>{fnum(r.MAE_BASE,3)} → {fnum(r.MAE_LOYO,3)}</td>"
    f"<td class='num {'good' if r.MEJORA_PCT and r.MEJORA_PCT>0 else 'bad'}'>"
    f"{r.MEJORA_PCT:+.1f}%</td></tr>"
    for _, r in c1.iterrows() if pd.notna(r.MEJORA_PCT))

# tabla C2 agregada
c2 = exp[exp.EXPERIMENTO == "C2_pesos_sinteticos"]
c2_rows = []
for tag in ["media", "decae"]:
    sub = c2[c2.VARIANTE == tag]
    pares = [(m, w.get(c, 0)) for c, m in zip(sub.CAMPO, sub.MAE_LOYO) if pd.notna(m)]
    pond = sum(m_ * w_ for m_, w_ in pares) / sum(w_ for _, w_ in pares) if pares else np.nan
    c2_rows.append((tag, pond))
base_pares = [(m, w.get(c, 0)) for c, m in zip(c2[c2.VARIANTE == "media"].CAMPO,
                                               c2[c2.VARIANTE == "media"].MAE_BASE) if pd.notna(m)]
c2_base = sum(m_ * w_ for m_, w_ in base_pares) / sum(w_ for _, w_ in base_pares) if base_pares else np.nan

HTML = f"""<title>Auditoría estadística del pipeline — Track Calidad</title>
<style>
:root {{ --bg:#F7F8F6; --panel:#FFFFFF; --ink:#14201A; --muted:#5C6B62; --faint:#8A968E;
  --line:#DDE3DC; --line2:#EBEFEA; --green:#1B6535; --green-w:#E6F0E9; --clay:#B0413E;
  --clay-w:#F6E9E8; --hl:#FBF7E9; --accent:#1B6535; }}
@media (prefers-color-scheme: dark) {{ :root {{
  --bg:#0E1512; --panel:#151E19; --ink:#E8EEE9; --muted:#9DA99F; --faint:#7C897F;
  --line:#26332C; --line2:#1D2822; --green:#5DBE84; --green-w:#16271D; --clay:#E08A86;
  --clay-w:#2A1917; --hl:#211E12; --accent:#5DBE84; }} }}
:root[data-theme="dark"] {{ --bg:#0E1512; --panel:#151E19; --ink:#E8EEE9; --muted:#9DA99F;
  --faint:#7C897F; --line:#26332C; --line2:#1D2822; --green:#5DBE84; --green-w:#16271D;
  --clay:#E08A86; --clay-w:#2A1917; --hl:#211E12; --accent:#5DBE84; }}
:root[data-theme="light"] {{ --bg:#F7F8F6; --panel:#FFFFFF; --ink:#14201A; --muted:#5C6B62;
  --faint:#8A968E; --line:#DDE3DC; --line2:#EBEFEA; --green:#1B6535; --green-w:#E6F0E9;
  --clay:#B0413E; --clay-w:#F6E9E8; --hl:#FBF7E9; --accent:#1B6535; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; line-height:1.6; }}
.wrap {{ max-width:1040px; margin:0 auto; padding:0 24px 96px; }}
header {{ padding:56px 0 30px; border-bottom:2px solid var(--ink); }}
.eyebrow {{ font-size:12px; letter-spacing:.16em; text-transform:uppercase; color:var(--accent);
  font-weight:700; margin-bottom:14px; }}
h1 {{ font-family:Georgia,serif; font-weight:600; font-size:clamp(30px,4.4vw,44px); line-height:1.1;
  margin:0 0 16px; text-wrap:balance; letter-spacing:-.01em; }}
.lede {{ font-size:18px; color:var(--muted); max-width:70ch; margin:0; }}
.meta {{ margin-top:20px; font-size:13px; color:var(--faint); display:flex; gap:20px; flex-wrap:wrap; }}
.meta b {{ color:var(--muted); }}
section {{ margin-top:52px; }}
.snum {{ font-family:Georgia,serif; font-size:13px; color:var(--accent); font-weight:700; letter-spacing:.05em; }}
h2 {{ font-family:Georgia,serif; font-weight:600; font-size:26px; margin:6px 0 8px; text-wrap:balance; }}
h3 {{ font-size:16px; font-weight:700; margin:26px 0 10px; }}
p {{ max-width:76ch; }}
.sub {{ font-size:12.5px; color:var(--faint); }}
.good {{ color:var(--green); }} .bad {{ color:var(--clay); font-weight:700; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; margin:22px 0; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:17px; }}
.card .k {{ font-size:27px; font-family:Georgia,serif; font-weight:600; font-variant-numeric:tabular-nums; }}
.card .l {{ font-size:12.5px; color:var(--muted); margin-top:4px; }}
.card.g .k {{ color:var(--green); }} .card.c .k {{ color:var(--clay); }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; margin:14px 0; }}
.scroll {{ overflow-x:auto; }}
th {{ text-align:left; font-size:11px; letter-spacing:.04em; text-transform:uppercase; color:var(--muted);
  font-weight:700; padding:9px 11px; border-bottom:1.5px solid var(--line); white-space:nowrap; }}
td {{ padding:8px 11px; border-bottom:1px solid var(--line2); vertical-align:top; }}
td.num,th.num {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
tbody tr:hover td {{ background:var(--line2); }}
.verd {{ display:inline-block; font-size:11px; font-weight:700; padding:2px 9px; border-radius:20px; }}
.v-ok {{ background:var(--green-w); color:var(--green); }}
.v-no {{ background:var(--clay-w); color:var(--clay); }}
.figs {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:22px; margin:18px 0; }}
figure {{ margin:0; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }}
figure img {{ width:100%; height:auto; display:block; }}
figcaption {{ font-size:12.5px; color:var(--muted); margin-top:10px; line-height:1.5; }}
.callout {{ border-left:3px solid var(--accent); background:var(--panel); padding:16px 20px;
  border-radius:0 8px 8px 0; margin:20px 0; font-size:14.5px; }}
.callout.warn {{ border-left-color:var(--clay); }}
.foot {{ margin-top:60px; padding-top:22px; border-top:1px solid var(--line); font-size:13px; color:var(--faint); }}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Piloto Predicción Reservas 1P · Gerencia de Desarrollo</div>
  <h1>Auditoría estadística del pipeline: ponderar la evidencia, no cambiar la curva</h1>
  <p class="lede">Rollback de los métodos que cambiaban la forma del modelo, y auditoría A/B
  de las palancas estadísticas reales del pipeline — con veredictos honestos: dos rechazadas,
  una validada, y bandas de incertidumbre nuevas. Track Calidad; Producción intacta.</p>
  <div class="meta">
    <span><b>Directriz</b> solo ponderación/tratamiento (familia re-anclaje)</span>
    <span><b>Conservado</b> perfil BK_P10–P90 + re-anclaje LISAMA</span>
    <span><b>Métrica</b> MAE_LOYO, train transformado vs error RAW</span>
  </div>
</header>

<section>
  <div class="snum">01 — El rollback, con autocrítica</div>
  <h2>Por qué se descartan métodos que "ganaban"</h2>
  <p>Los 6 overrides de motor de la búsqueda anterior mejoraban el MAE_LOYO — y aun así se
  descartan. La razón: <b>cambiaban la forma de la curva</b> (plano, recta, sigmoide), y la
  forma no es un hiperparámetro. La isotónica + escalera ES la semántica financiera del motor
  (cómo se desincorporan reservas por clase al caer el precio); una recta de Theil-Sen puede
  cross-validar mejor y aún así contar una historia que finanzas no puede ratificar. La mejora
  legítima opera <b>sobre la evidencia</b> (pesos, offsets, incertidumbre), no sobre el modelo.</p>
  <div class="scroll"><table><thead><tr>{th(['Campo','Método descartado','Ganancia LOYO que se deja','Veredicto'])}</tr></thead>
  <tbody>{rech_html}</tbody></table></div>
  <div class="callout">
    <b>Lo que sobrevive el filtro:</b> el <b>perfil de salida BK_P10–P90</b> (no cambia el motor —
    cambia los puntos de anclaje sintéticos por la economía del FC certificado) y el
    <b>re-anclaje de LISAMA</b> (ponderación pura: centra los offsets de vigencia, la isotónica
    sigue siendo la isotónica). Ambos son de la familia estadística.
  </div>
</section>

<section>
  <div class="snum">02 — Auditoría por etapa</div>
  <h2>Dónde viven las palancas estadísticas del pipeline</h2>
  <div class="scroll"><table><thead><tr>{th(['Etapa','Palanca estadística','Hallazgo crítico','Acción'])}</tr></thead><tbody>
  <tr><td><b>05 Breakeven</b></td><td>Perfil de salida BK_P10–P90</td>
  <td>Auditado contra colas de aceite minúsculo: BK_P10 máx 105.7, <b>0/116 sospechosos</b> — robusto</td>
  <td><span class="verd v-ok">SE CONSERVA</span></td></tr>
  <tr><td><b>01 ETL</b></td><td>Flags TARGET_NULO / NIVEL_DEFINICIONAL</td>
  <td>Se flaggean pero no modulan pesos del entrenamiento</td><td>Probado en C3 → sin efecto</td></tr>
  <tr><td><b>02 Sintéticos</b></td><td>Pesos por nivel (balance de masa, W_MIN=0.05)</td>
  <td>Nunca A/B-eado desde Path D</td><td>Probado en C2 → <span class="verd v-ok">VALIDADO</span></td></tr>
  <tr><td><b>03 Modelo</b></td><td>Peso por vigencia (uniforme) · centrado todo-o-nada</td>
  <td>Recencia ya rechazada (s5, respetado); shrinkage parcial nunca probado</td>
  <td>Probado en C1 → <span class="verd v-no">RECHAZADO</span></td></tr>
  <tr><td><b>03b M2</b></td><td>Theil-Sen (mediana de pendientes)</td>
  <td>Ya es el estimador robusto correcto para N pequeño</td><td>Sin cambio</td></tr>
  <tr><td><b>04 Confiabilidad</b></td><td>ALTA/MEDIA/BAJA sin cuantificación</td>
  <td>El nivel es un juicio discreto; no dice CUÁNTO puede errar la curva</td>
  <td>C4 → <span class="verd v-ok">BANDAS LOYO NUEVAS</span></td></tr>
  <tr><td><b>NORTE / tests</b></td><td>tests_calidad tiers 1–3</td>
  <td>El gate del track exige mejora benchmarkeada + invariantes por campo</td><td>Vigente</td></tr>
  </tbody></table></div>
</section>

<section>
  <div class="snum">03 — C1: centrado parcial por η²</div>
  <h2>La generalización elegante del re-anclaje… que los datos rechazaron</h2>
  <p>Hipótesis: si el centrado total borra señal en los casos frontera (CASABE, η²=0.44),
  remover solo la <b>fracción η² del offset</b> (shrinkage tipo empirical-Bayes) debería quedarse
  con lo mejor de ambos mundos. El A/B dice que no:</p>
  <figure><img alt="MAE vs fracción de centrado" src="{fig_c1}">
  <figcaption>MAE_LOYO vs fracción del offset removida. <b>LISAMA</b> (η²=1.0): más centrado =
  mejor — el offset es 100% año, el centrado total ya adoptado es óptimo. <b>CASABE</b> (η²=0.44):
  TODA fracción empeora (f=η²: −41%, f=1: −94%) — precio y año están mezclados y cualquier
  centrado rompe señal real. No existe punto intermedio bueno.</figcaption></figure>
  <div class="scroll"><table><thead><tr>{th(['Campo','Variante','MAE_LOYO base → variante','Δ'])}</tr></thead>
  <tbody>{c1_html}</tbody></table></div>
  <div class="callout warn">
    <b>Veredicto: RECHAZADO.</b> El discriminante del re-anclaje no es η² continuo sino el A/B
    por campo: donde el año explica TODO (LISAMA), centrar total; donde explica parte (CASABE),
    no centrar nada — el campo queda honesto en MEDIA con su sesgo documentado. El shrinkage era
    teoría atractiva; la validación cruzada la mató. Eso también es un resultado.
  </div>
</section>

<section>
  <div class="snum">04 — C2 y C3: los pesos existentes, validados</div>
  <h2>El balance de masa de la escalera sobrevive su primer A/B</h2>
  <div class="cards">
    <div class="card g"><div class="k">Δ=0.000</div><div class="l">las 3 variantes empatan EXACTO en los 20 campos</div></div>
    <div class="card g"><div class="k">×0.05–×5</div><div class="l">prueba extrema: la curva no se mueve nada</div></div>
    <div class="card"><div class="k">0.0%</div><div class="l">degradación del gate dorado</div></div>
  </div>
  <p><b>C2:</b> el esquema de pesos sintéticos (masa por nivel = masa real, vigente desde Path D)
  nunca había sido A/B-eado. El deep-dive mostró algo más fuerte que un empate: los pesos hoy son
  <b>inertes</b> — la isotónica (PAVA) solo los usa al agrupar violaciones de monotonía, y el
  guard de banda + el cap §4.3 ya eliminaron esos conflictos aguas arriba. Incluso con pesos
  ×0.05–×5 la curva no cambia 0.001 MBPE. Lo que ancla el tramo bajo son los <b>VALORES</b> de
  la escalera (perfil BK_P10–P90) + hard-zero, no los pesos. <i>Fe de erratas: la versión previa
  reportó "decaimiento peor (1.522)"; era un artefacto de composición (CUSIANA/CUPIAGUA SUR sin
  sintéticos no entraban en esa variante).</i> Detalle completo: <code>c2_pesos_sinteticos</code>.</p>
  <p><b>C3 (peso 0.5 a puntos con quiebre definicional):</b> cada campo de la ola tiene exactamente
  1 punto flaggeado (2026_REGALIAS). Efecto: ±0–5%, nada se acerca al umbral de 15%. La razón es
  instructiva: el LOYO ya aísla ese año como fold completo — bajarle el peso en el train apenas
  mueve la curva. <b>Sin efecto → no se adopta</b> (cambio sin beneficio = complejidad gratis).</p>
</section>

<section>
  <div class="snum">05 — C4: bandas de incertidumbre LOYO</div>
  <h2>Cuánto puede errar la curva, campo por campo</h2>
  <p>La mejora estadística nueva que SÍ entra: la confiabilidad deja de ser solo una etiqueta
  (ALTA/MEDIA/BAJA) y se convierte en un <b>rango empírico</b>. Los residuales del LOYO — cuánto
  se desvió cada año real de lo que la curva (entrenada sin él) predijo — dan los cuantiles
  P10/P90 de una banda alrededor de la curva. No cambia la curva: la rodea de honestidad.
  Columnas nuevas <code>VOL_P10_MBPE</code>/<code>VOL_P90_MBPE</code> en el export Calidad
  (motor primario; banda colapsa a 0 bajo el piso de abandono — ahí la incertidumbre es del BK,
  no de la curva).</p>
  <div class="figs">
    {"".join(f'<figure><img alt="banda {c}" src="{figs_banda[c]}"><figcaption><b>{c}</b> — banda P10–P90 de residuales LOYO alrededor de la curva Calidad, con el deck real por vigencia. Banda angosta = la curva predice bien años no vistos; banda ancha = incertidumbre honesta visible.</figcaption></figure>' for c in BANDAS_CAMPOS if figs_banda.get(c))}
  </div>
  <p class="sub">Aviso PBI: 2 columnas nuevas en <code>output_matriz_prediccion.csv</code> del track
  Calidad → el tablero necesitará DAX nuevo (área sombreada) cuando se ratifique.</p>
</section>

<section>
  <div class="snum">06 — Impacto en el portafolio (post-rollback)</div>
  <h2>Qué queda en el track Calidad</h2>
  <div class="cards">
    <div class="card g"><div class="k">{fnum(loyo_p,2)}→{fnum(loyo_c,2)}</div><div class="l">MAE_LOYO ponderado ola 1 (perfil + LISAMA)</div></div>
    <div class="card g"><div class="k">{len(subieron)}</div><div class="l">campos MEDIA→ALTA</div></div>
    <div class="card"><div class="k">1</div><div class="l">override vigente (LISAMA/re-anclaje)</div></div>
    <div class="card"><div class="k">2</div><div class="l">candidatos rechazados por A/B (C1, C3)</div></div>
  </div>
  <figure><img alt="portafolio" src="{fig_port}">
  <figcaption>Σ Vol 1P de la ola 1 vs Brent: Producción vs Calidad (perfil de salida + re-anclaje
  LISAMA). El ancla agregada a BRENT_REF se conserva; las diferencias son de forma donde el deck
  no respalda sensibilidad.</figcaption></figure>
  <div class="callout">
    <b>Lectura honesta del rollback:</b> el MAE_LOYO ponderado del track queda en
    {fnum(loyo_c,2)} (vs {fnum(loyo_p,2)} de Producción) — menor que con los motores descartados
    ({'1.47'}), y esa diferencia es el <b>precio de conservar la semántica</b> isotónica+escalera.
    Se paga a sabiendas: un número mejor con una curva que finanzas no puede ratificar no es una
    mejora del sistema.
  </div>
</section>

<div class="foot">
  Piloto Predicción Reservas 1P vs Brent · Soporte analítico interno (NO reemplaza SEC / EcoFaro /
  ARIES / Planning Space). Track Calidad pre-ratificación · s9 (2026-07-12). Protocolo A/B: LOYO
  con train transformado y error contra deltas reales sin transformar; adopción ≥15% + invariantes.
  Evidencia completa: seleccion_metodos.csv (adoptados y rechazados), experimento_estadistico.csv.
</div>
</div>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"Informe: {OUT} ({len(HTML)//1024} KB)")
print(f"Rechazados: {len(rech)} | override vigente: {(reg.ESTADO=='ADOPTADO').sum()} | "
      f"MAE_LOYO pond {loyo_p:.2f}->{loyo_c:.2f} | MEDIA->ALTA: {len(subieron)}")
