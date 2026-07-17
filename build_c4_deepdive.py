"""build_c4_deepdive.py — Mini-informe: C4 a fondo (bandas de incertidumbre LOYO).

Cómo se calculan (aritmética completa con CASTILLA), qué significan (validación cruzada
con NIVEL_CONFIANZA), cómo implementarlas en el tablero PBI y cómo usarlas en análisis
futuros (gate cuantitativo, lectura CAPEX, backtest de calibración).
"""
import base64
import importlib
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from motores_modelo1 import MotorIsotonico

m3 = importlib.import_module("03_modelo")
BASE = Path(__file__).parent
OUT = BASE / "docs" / "c4_bandas_loyo.html"

AX = "#5C6B62"; C_G = "#1B6535"; C_R = "#B0413E"; C_B = "#3B6EA5"; C_Y = "#B9A24B"
YEAR_COL = {"2024": "#3B6EA5", "2025": "#C77D2E", "2026": "#B0413E"}
NIV_COL = {"ALTA": "#1B6535", "MEDIA": "#B9A24B", "BAJA": "#B0413E"}

t = pd.read_parquet(BASE / "datos" / "staging_calidad" / "tablon_unico.parquet")
mc = pd.read_csv(BASE / "resultados_calidad" / "output_matriz_prediccion.csv")
mc = mc[mc.MOTOR == "Isotonica"]
met = pd.read_csv(BASE / "resultados_calidad" / "metricas.csv")
EXCLUIR = {"FLOREÑA", "NARE UNIFICADO", "INFANTAS"}


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=135, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _st(ax):
    for s in ax.spines.values():
        s.set_color(AX); s.set_linewidth(0.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(colors=AX, labelsize=8)
    ax.grid(True, alpha=0.18, color=AX, lw=0.5)


# ── Datos CASTILLA: residuales fold por fold ─────────────────────────────────
campo = "CASTILLA"
sub = t[(t.CAMPO == campo) & t.DELTA_SENS_MBPE.notna() & t.PRECIO_NETO_USD_BBL.notna()
        & (~t.ES_BASELINE)]
r = sub[~sub.ES_SINTETICO].reset_index(drop=True)
s = sub[sub.ES_SINTETICO].reset_index(drop=True)
r["A"] = r.VIGENCIA.astype(str).str.slice(0, 4)
ws = m3.pesos_sinteticos_tramo(s, len(r))[0]
folds = []
for a in sorted(r.A.unique()):
    te = (r.A == a).values; tr = ~te
    xt = np.concatenate([s.PRECIO_NETO_USD_BBL.values, r.loc[tr, "PRECIO_NETO_USD_BBL"].values])
    yt = np.concatenate([s.DELTA_SENS_MBPE.values, r.loc[tr, "DELTA_SENS_MBPE"].values])
    wt = np.concatenate([ws, np.ones(tr.sum())])
    mo = MotorIsotonico().fit(xt, yt, sample_weight=wt)
    for _, row in r[te].iterrows():
        yh = float(mo.predict([row.PRECIO_NETO_USD_BBL])[0])
        folds.append({"anio": a, "precio": row.PRECIO_NETO_USD_BBL,
                      "real": row.DELTA_SENS_MBPE, "pred": yh,
                      "resid": row.DELTA_SENS_MBPE - yh})
fdf = pd.DataFrame(folds)
q10, q90 = np.quantile(fdf.resid, 0.10), np.quantile(fdf.resid, 0.90)

# Fig 1: mecánica — real vs pred por fold, residuales como segmentos
fig, ax = plt.subplots(figsize=(9.4, 3.9))
_st(ax)
for _, row in fdf.iterrows():
    ax.plot([row.precio, row.precio], [row.pred, row.real], "-", color=AX, lw=1.0,
            alpha=0.6, zorder=2)
for a, g in fdf.groupby("anio"):
    ax.scatter(g.precio, g.real, s=52, color=YEAR_COL[a], edgecolor="white",
               linewidth=0.7, zorder=4, label=f"real deck {a}")
    ax.scatter(g.precio, g.pred, s=42, marker="x", color=YEAR_COL[a], zorder=3, lw=1.6)
ax.annotate("× = lo que la curva (entrenada SIN ese año) predijo\n● = lo que el deck real dijo\nsegmento = residual",
            xy=(0.02, 0.06), xycoords="axes fraction", fontsize=8, color=AX)
ax.set_title(f"CASTILLA — los 9 residuales LOYO: real vs predicho, año por año",
             fontsize=10, color=AX, loc="left")
ax.set_xlabel("Precio Neto (USD/bbl)", color=AX, fontsize=9)
ax.set_ylabel("Δ reservas (MBPE)", color=AX, fontsize=9)
ax.legend(fontsize=8, frameon=False, labelcolor=AX, loc="lower right")
fig.tight_layout()
fig1 = _b64(fig)

# Fig 2: banda CASTILLA (curva + banda + deck)
sc = mc[mc.CAMPO == campo].sort_values("PRECIO_NETO_EFECTIVO_USD_BBL")
bl = float(sc.VOLUMEN_1P_BASELINE_MBPE.iloc[0])
fig, ax = plt.subplots(figsize=(9.4, 4.1))
_st(ax)
x = sc.PRECIO_NETO_EFECTIVO_USD_BBL
ax.fill_between(x, sc.VOL_P10_MBPE.astype(float), sc.VOL_P90_MBPE.astype(float),
                color=C_G, alpha=0.18, linewidth=0, label=f"banda P10–P90  [{q10:+.2f}, {q90:+.2f}]")
ax.plot(x, sc.VOLUMEN_1P_PREDICHO_MBPE, "-", color=C_G, lw=2.4, label="curva Calidad", zorder=5)
ax.axhline(bl, ls=(0, (4, 3)), color=C_Y, lw=1.2, label=f"baseline 1P = {bl:.1f}")
for a, g in r.groupby("A"):
    ax.scatter(g.PRECIO_NETO_USD_BBL, g.DELTA_SENS_MBPE + bl, s=46, color=YEAR_COL[a],
               edgecolor="white", linewidth=0.6, zorder=6, label=f"deck {a}")
ax.set_title("CASTILLA — la banda alrededor de la curva (P10 = piso empírico, P90 = techo)",
             fontsize=10, color=AX, loc="left")
ax.set_xlabel("Precio Neto (USD/bbl)", color=AX, fontsize=9)
ax.set_ylabel("Vol 1P (MBPE)", color=AX, fontsize=9)
ax.legend(fontsize=7.6, frameon=False, labelcolor=AX, loc="lower right", ncol=2)
fig.tight_layout()
fig2 = _b64(fig)

# Fig 3: ancho relativo por campo (ola 1), color por nivel
rp = dict(zip(met.CAMPO, met.get("LOYO_RESID_P10")))
rq = dict(zip(met.CAMPO, met.get("LOYO_RESID_P90")))
g = mc[mc.VOL_P10_MBPE.notna()].groupby("CAMPO").agg(
    base=("VOLUMEN_1P_BASELINE_MBPE", "first"), niv=("NIVEL_CONFIANZA", "first"))
g["ancho_rel"] = g.index.map(lambda c: (rq.get(c, np.nan) - rp.get(c, np.nan))) / g.base
ola = g[(g.base >= 10) & (~g.index.isin(EXCLUIR))].sort_values("ancho_rel")
fig, ax = plt.subplots(figsize=(9.4, 4.6))
_st(ax)
cols = [NIV_COL.get(n, AX) for n in ola.niv]
ax.barh(range(len(ola)), ola.ancho_rel * 100, color=cols, alpha=0.85)
ax.set_yticks(range(len(ola)))
ax.set_yticklabels(ola.index, fontsize=7.5)
ax.axvline(10, color=C_Y, lw=1.2, ls="--")
ax.axvline(25, color=C_R, lw=1.2, ls="--")
ax.annotate("umbral propuesto ALTA <10%", xy=(10.4, 0.4), fontsize=8, color=C_Y)
ax.annotate("MEDIA <25%", xy=(25.4, 0.4), fontsize=8, color=C_R)
ax.set_title("Ancho de banda relativo (P90−P10)/baseline — ola 1, color = nivel de confianza actual",
             fontsize=10, color=AX, loc="left")
ax.set_xlabel("% del baseline", color=AX, fontsize=9)
fig.tight_layout()
fig3 = _b64(fig)

# tabla fold-por-fold
folds_html = "".join(
    f"<tr><td>{f['anio']}</td><td class='num'>{f['precio']:.1f}</td>"
    f"<td class='num'>{f['real']:+.2f}</td><td class='num'>{f['pred']:+.2f}</td>"
    f"<td class='num'><b>{f['resid']:+.2f}</b></td></tr>" for f in folds)

# tabla lectura Brent 60/70/80 CASTILLA
lect = []
scb = mc[mc.CAMPO == campo]
for b in [60, 70, 80]:
    i = (scb.BRENT_USD_BBL - b).abs().idxmin()
    row = scb.loc[i]
    lect.append((b, row.VOLUMEN_1P_PREDICHO_MBPE, row.VOL_P10_MBPE, row.VOL_P90_MBPE))
lect_html = "".join(
    f"<tr><td class='num'>{b}</td><td class='num'><b>{v:.1f}</b></td>"
    f"<td class='num'>{p10:.1f}</td><td class='num'>{p90:.1f}</td>"
    f"<td>«con ~80% de confianza empírica, entre {p10:.0f} y {p90:.0f} MBPE»</td></tr>"
    for b, v, p10, p90 in lect)

# distribución por nivel
dist = g[~g.index.isin(EXCLUIR)].groupby("niv").ancho_rel.median() * 100

HTML = f"""<title>C4 a fondo — Bandas de incertidumbre LOYO</title>
<style>
body {{ margin:0; background:#fff; color:#14201A; line-height:1.6;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
.wrap {{ max-width:920px; margin:0 auto; padding:36px 26px 80px; }}
.eyebrow {{ font-size:11px; letter-spacing:.15em; text-transform:uppercase; color:#1B6535;
  font-weight:700; margin-bottom:10px; }}
h1 {{ font-family:Georgia,serif; font-size:31px; line-height:1.15; margin:0 0 12px; }}
h2 {{ font-family:Georgia,serif; font-size:21px; margin:34px 0 8px; }}
h3 {{ font-size:15px; margin:22px 0 6px; }}
p {{ max-width:78ch; font-size:14.5px; }}
.sub {{ font-size:12px; color:#8A968E; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; margin:12px 0; }}
th {{ text-align:left; font-size:10.5px; letter-spacing:.04em; text-transform:uppercase;
  color:#5C6B62; padding:7px 10px; border-bottom:1.5px solid #DDE3DC; }}
td {{ padding:6px 10px; border-bottom:1px solid #EBEFEA; }}
td.num {{ font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap; }}
img {{ width:100%; height:auto; display:block; border:1px solid #DDE3DC; border-radius:8px;
  margin:12px 0 6px; }}
.cap {{ font-size:12.5px; color:#5C6B62; margin:4px 0 18px; }}
.box {{ border-left:3px solid #1B6535; background:#F3F7F4; padding:13px 17px;
  border-radius:0 8px 8px 0; margin:16px 0; font-size:14px; }}
.box.warn {{ border-left-color:#B0413E; background:#FAF1F0; }}
code, pre {{ font-family:ui-monospace,"Cascadia Code",Consolas,monospace; font-size:12px; }}
pre {{ background:#F3F5F2; border:1px solid #DDE3DC; border-radius:8px; padding:12px 14px;
  overflow-x:auto; line-height:1.5; }}
ol,ul {{ font-size:14px; max-width:78ch; }}
li {{ margin:5px 0; }}
</style>
<div class="wrap">
<div class="eyebrow">Piloto Predicción 1P · Deep-dive C4 · 2026-07-12</div>
<h1>Bandas de incertidumbre LOYO: cómo se calculan, cómo van al tablero y cómo usarlas</h1>
<p>La idea en una frase: <b>la mejor estimación de cuánto puede errar la curva el próximo año
es cuánto erró en cada año pasado cuando ese año no se le mostró</b>. Eso es exactamente lo
que el LOYO ya calcula — la banda solo convierte esos errores en un rango visible.</p>

<h2>1 · La aritmética completa (CASTILLA)</h2>
<p>Tres pasos, sin cajas negras:</p>
<p><b>Paso 1 — Un residual por punto real, sin trampa.</b> Para cada vigencia (2024, 2025,
2026) se entrena la isotónica SIN ese año (sintéticos + los otros años) y se predice el año
excluido. Residual = lo que el deck real dijo − lo que la curva predijo:</p>
<table><thead><tr><th>Año excluido</th><th class="num">Precio Neto</th><th class="num">Δ real</th>
<th class="num">Δ predicho</th><th class="num">residual</th></tr></thead>
<tbody>{folds_html}</tbody></table>
<img alt="mecanica residuales" src="{fig1}">
<div class="cap">La curva sin 2024 subestima (+1.8 a +2.2: el deck real estuvo MEJOR de lo
predicho); sin 2025 falla en ambos sentidos (−2.2 a +2.4); 2026 casi clava (−0.4). Esos 9
números SON la incertidumbre observada del campo.</div>
<p><b>Paso 2 — Cuantiles empíricos.</b> De los 9 residuales: q10 = <b>{q10:+.2f}</b>,
q90 = <b>{q90:+.2f}</b> MBPE (guardados en <code>metricas.csv</code> como
<code>LOYO_RESID_P10/P90</code>).</p>
<p><b>Paso 3 — La banda alrededor de la curva</b>, con tres reglas duras:
<code>VOL_P10 = max(0, min(vol, vol+q10))</code> · <code>VOL_P90 = max(vol, vol+q90)</code> ·
bajo el piso de abandono la banda colapsa (vol=0 es afirmación del BK, no de la curva).</p>
<img alt="banda castilla" src="{fig2}">
<div class="cap">CASTILLA: a cualquier Brent, el rango honesto es la curva {q10:+.1f}/{q90:+.1f}
MBPE — un rango de 4.3 MBPE sobre un baseline de 186 (2.3%).</div>

<h2>2 · Qué significa: la banda VALIDA la clasificación de confianza</h2>
<img alt="ancho relativo por campo" src="{fig3}">
<div class="cap">El hallazgo de la ola 1: los campos ALTA (verde) tienen bandas angostas
(mediana {dist.get('ALTA', 0):.0f}% del baseline) y los MEDIA (ámbar) anchas (mediana
{dist.get('MEDIA', 0):.0f}%; BAJA: {dist.get('BAJA', 0):.0f}%). Dos sistemas independientes —
el gate cualitativo y los residuales — cuentan la misma historia. CUSIANA (17%) y CASABE (26%)
son exactamente los que el confound tiene en MEDIA.</div>
<div class="box"><b>Lectura tipo CAPEX (CASTILLA):</b></div>
<table><thead><tr><th class="num">Brent</th><th class="num">Vol predicho</th>
<th class="num">P10 (piso)</th><th class="num">P90</th><th>Frase para el comité</th></tr></thead>
<tbody>{lect_html}</tbody></table>

<h2>3 · Implementación en el tablero (cuando se ratifique)</h2>
<p>El export Calidad ya trae las 2 columnas. Pasos concretos sobre el modelo semántico
"Predicción de Reservas" (PBIP):</p>
<ol>
<li><b>Cerrar Power BI Desktop ANTES de editar</b> (regla del proyecto: Desktop abierto +
edición externa = clobber al guardar).</li>
<li><b>Power Query (partición de Predicción):</b> tipar las columnas nuevas con cultura
<b>en-US</b> — la regla crítica del modelo (es-CO convierte 58.0 → 580):
<pre>Table.TransformColumnTypes(#"paso anterior",
    {{{{"VOL_P10_MBPE", type number}}, {{"VOL_P90_MBPE", type number}}}}, "en-US")</pre></li>
<li><b>Medidas DAX</b> (la matriz es CAMPO × MOTOR × Brent; los visuales ya filtran
Isotónica):
<pre>Vol P10 := SUM ( Predicción[VOL_P10_MBPE] )
Vol P90 := SUM ( Predicción[VOL_P90_MBPE] )
Ancho Banda := [Vol P90] - [Vol P10]
Ancho Banda % :=
DIVIDE ( [Ancho Banda], SUM ( Predicción[VOLUMEN_1P_BASELINE_MBPE] ) )</pre></li>
<li><b>Visual de banda</b> (página de curva por campo): PBI no tiene "banda" nativa en el
line chart — el patrón estándar es un <b>gráfico de áreas apiladas</b>: serie 1 = [Vol P10]
con relleno transparente, serie 2 = [Ancho Banda] con relleno verde translúcido, y encima la
línea [Vol Predicho]. Alternativa rápida: llevar P10/P90 al <b>tooltip</b> y una tarjeta
"Rango @ Brent seleccionado" = <code>[Vol P10] &amp; " – " &amp; [Vol P90]</code>.</li>
<li><b>Tarjeta de confiabilidad cuantitativa</b>: [Ancho Banda %] junto al badge
ALTA/MEDIA/BAJA — el número que respalda la etiqueta.</li>
</ol>
<div class="box warn"><b>Regla de agregación — importante:</b> los cuantiles NO se suman. La
banda es válida <b>por campo</b>; en la vista de portafolio (ΣVol) NO mostrar ΣP10/ΣP90 como
banda del total (sobre-estima el ancho: ignora la diversificación entre campos). Portafolio
con banda = bootstrap conjunto, candidato de ola 2.</div>

<h2>4 · Futuros análisis</h2>
<p><b>(a) Gate cuantitativo de confiabilidad.</b> Umbral propuesto (calibrado con la
distribución actual): ALTA exige <code>Ancho Banda % &lt; 10%</code>, MEDIA &lt; 25%. Se
añadiría como criterio COMPLEMENTARIO en <code>clasificar_confianza</code> (04) tras
ratificación — hoy la correlación ya es fuerte, el gate lo haría explícito.</p>
<p><b>(b) Backtest de calibración con 2026_Q2.</b> Cuando lleguen las reservas reales del
próximo quarter, la banda se vuelve <b>testeable</b>: ~80% de los campos deberían caer dentro
de su banda P10–P90 (eso afirma un rango 80%). Si caen muchos menos, la banda es optimista;
muchos más, conservadora. Se integra al backtest G4 (snapshot vs real) como
"cobertura empírica de banda" — un número por corrida del loop rodante.</p>
<p><b>(c) Limitaciones honestas</b> (documentadas, candidatas de ola 2):
con N≈9 los cuantiles son gruesos (P10/P90 ≈ cerca del mín/máx observado);
la banda es <b>constante</b> a lo largo de la curva — no se ensancha en la zona extrapolada
(mejora natural: inflarla donde <code>ES_EXTRAPOLADO=True</code>);
y es banda de VIGENCIA (captura recertificaciones entre decks, la fuente de error dominante),
no de ruido de medición.</p>

<p class="sub">Piloto Predicción Reservas 1P · Soporte analítico interno. Fuente:
resultados_calidad/output_matriz_prediccion.csv (VOL_P10/P90_MBPE, 112 campos),
metricas.csv (LOYO_RESID_P10/P90), 03_modelo.py::residuales_loyo. Flag PRED_BANDAS_LOYO
(off en Producción hasta ratificar).</p>
</div>
"""
OUT.write_text(HTML, encoding="utf-8")
print(f"OK: {OUT} ({len(HTML)//1024} KB)")
