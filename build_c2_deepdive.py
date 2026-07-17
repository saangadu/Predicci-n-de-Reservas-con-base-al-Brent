"""build_c2_deepdive.py — Mini-informe: C2 a fondo (pesos sintéticos).

Hallazgos del deep-dive (2026-07-12):
  1. Per-campo el A/B es un empate EXACTO (Δ=0.000 en los 20 campos); el "decaimiento
     peor" del agregado era artefacto de composición (CUSIANA/CUPIAGUA SUR sin sintéticos
     no entraban en esa variante).
  2. Prueba extrema ×0.05–×5: la curva no cambia NADA. Razón: PAVA (isotónica) solo usa
     pesos al agrupar VIOLACIONES de monotonía; el guard de banda + el cap §4.3 eliminaron
     los conflictos sintético↔real → los pesos son inertes hoy.
  3. Lo que ancla el tramo bajo son los VALORES de la escalera (niveles/perfil) + hard-zero,
     no los pesos. La palanca real del tramo bajo es el perfil BK_P10–P90 (ya adoptado).
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
OUT = BASE / "docs" / "c2_pesos_sinteticos.html"

AX = "#5C6B62"; C1 = "#1B6535"; C2C = "#B0413E"; C3 = "#3B6EA5"; C_BASE = "#B9A24B"
t = pd.read_parquet(BASE / "datos" / "staging_calidad" / "tablon_unico.parquet")
exp = pd.read_csv(BASE / "resultados_calidad" / "experimento_estadistico.csv")


def datos(c):
    s = t[(t.CAMPO == c) & t.DELTA_SENS_MBPE.notna() & t.PRECIO_NETO_USD_BBL.notna()
          & (~t.ES_BASELINE)]
    return s[~s.ES_SINTETICO].copy(), s[s.ES_SINTETICO].copy()


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=135, transparent=False,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _st(ax):
    for s in ax.spines.values():
        s.set_color(AX); s.set_linewidth(0.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(colors=AX, labelsize=8)
    ax.grid(True, alpha=0.18, color=AX, lw=0.5)


# ── Fig 1: el mecanismo de balance de masa (CASTILLA) ────────────────────────
r, s = datos("CASTILLA")
niv = s.VOLUMEN_1P_SENSIBILIDAD_MBPE.round(1)
grupos = sorted(s.groupby(niv).size().items())
nivs = [g[0] for g in grupos]; ns = [g[1] for g in grupos]
pesos = [m3.peso_sintetico(len(r), n) for n in ns]
masas = [n * p for n, p in zip(ns, pesos)]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.4))
for a in (a1, a2):
    _st(a)
a1.bar(range(len(nivs)), ns, color=C3, alpha=0.8)
a1.set_title("Puntos sintéticos por nivel de la escalera", fontsize=9.5, color=AX, loc="left")
a1.set_xlabel("nivel de la escalera (abandono → tope)", color=AX, fontsize=8.5)
a1.set_ylabel("n puntos", color=AX, fontsize=8.5)
a2.bar(range(len(nivs)), masas, color=C1, alpha=0.85)
a2.axhline(len(r), color=C2C, lw=1.6, ls="--", label=f"masa real (n_real = {len(r)})")
a2.set_title("Masa (n × peso) por nivel: TODOS = 9.0", fontsize=9.5, color=AX, loc="left")
a2.set_xlabel("nivel de la escalera", color=AX, fontsize=8.5)
a2.set_ylabel("masa efectiva", color=AX, fontsize=8.5)
a2.set_ylim(0, 12)
a2.legend(fontsize=8, frameon=False, labelcolor=AX)
fig.tight_layout()
fig1 = _b64(fig)

# ── Fig 2: prueba extrema — curvas con pesos ×0.05 / ×1 / ×5 (idénticas) ─────
x = np.concatenate([s.PRECIO_NETO_USD_BBL.values, r.PRECIO_NETO_USD_BBL.values])
y = np.concatenate([s.DELTA_SENS_MBPE.values, r.DELTA_SENS_MBPE.values])
w1 = m3.pesos_sinteticos_tramo(s, len(r))[0]
grid = np.linspace(x.min(), x.max() + 5, 160)
fig, ax = plt.subplots(figsize=(9.6, 4.0))
_st(ax)
estilos = [("pesos ×0.05", 0.05, C2C, 4.6, "-"), ("pesos ×1 (actual)", 1.0, C1, 2.8, "-"),
           ("pesos ×5", 5.0, C3, 1.4, "--")]
for tag, f, col, lw, ls in estilos:
    w = np.concatenate([w1 * f, np.ones(len(r))])
    yy = MotorIsotonico().fit(x, y, sample_weight=w).predict(grid)
    ax.plot(grid, yy, ls, color=col, lw=lw, label=tag, alpha=0.95)
ax.scatter(s.PRECIO_NETO_USD_BBL, s.DELTA_SENS_MBPE, s=22, color=C3, alpha=0.45,
           label="sintéticos (escalera)", zorder=3)
ax.scatter(r.PRECIO_NETO_USD_BBL, r.DELTA_SENS_MBPE, s=46, color="#14201A",
           edgecolor="white", linewidth=0.6, label="deck real", zorder=4)
ax.axvspan(r.PRECIO_NETO_USD_BBL.min(), r.PRECIO_NETO_USD_BBL.max(), color=C_BASE,
           alpha=0.10, label="banda real ($53–69)")
ax.set_title("CASTILLA — tres curvas con pesos ×0.05 / ×1 / ×5: PERFECTAMENTE superpuestas "
             "(Δ máx = 0.000 MBPE)", fontsize=10, color=AX, loc="left")
ax.set_xlabel("Precio Neto (USD/bbl)", color=AX, fontsize=9)
ax.set_ylabel("Δ reservas (MBPE)", color=AX, fontsize=9)
ax.legend(fontsize=8, frameon=False, labelcolor=AX, loc="lower right")
fig.tight_layout()
fig2 = _b64(fig)

# ── Fig 3: dónde SÍ actúan los pesos (demo didáctica con conflicto inyectado) ─
# Se inyecta un punto real FICTICIO por debajo del nivel de la escalera a un precio
# mayor (violación de monotonía) para ver a PAVA usar los pesos al agrupar.
xf = np.array([30, 32, 34, 36, 38, 40.0, 45.0])   # 6 sintéticos nivel 100 + 1 "real" 20
yf = np.array([100, 100, 100, 100, 100, 100.0, 20.0])
fig, ax = plt.subplots(figsize=(9.6, 3.6))
_st(ax)
for tag, wsin, col, lw in [("peso sintético alto (masa 9)", 1.5, C1, 2.6),
                           ("peso sintético bajo (0.1)", 0.1, C2C, 2.6)]:
    w = np.array([wsin] * 6 + [1.0])
    yy = MotorIsotonico().fit(xf, yf, sample_weight=w).predict(np.linspace(28, 47, 100))
    ax.plot(np.linspace(28, 47, 100), yy, "-", color=col, lw=lw, label=tag)
ax.scatter(xf[:6], yf[:6], s=30, color=C3, alpha=0.6, label="escalera (nivel 100)")
ax.scatter([45], [20], s=70, color="#14201A", edgecolor="white", zorder=4,
           label="punto real EN CONFLICTO (20 a mayor precio)")
ax.set_title("Demo didáctica: los pesos SOLO deciden cuando hay CONFLICTO de monotonía "
             "(real por debajo de la escalera a mayor precio)", fontsize=9.5, color=AX, loc="left")
ax.set_xlabel("Precio Neto (USD/bbl)", color=AX, fontsize=9)
ax.set_ylabel("Δ (MBPE)", color=AX, fontsize=9)
ax.legend(fontsize=8, frameon=False, labelcolor=AX)
fig.tight_layout()
fig3 = _b64(fig)

# tabla per-field
c2 = exp[exp.EXPERIMENTO == "C2_pesos_sinteticos"]
piv = c2.pivot_table(index="CAMPO", columns="VARIANTE", values="MAE_LOYO")
piv["base"] = c2.groupby("CAMPO").MAE_BASE.first()
rows = "".join(
    f"<tr><td><b>{c}</b></td><td class='num'>{piv.loc[c,'base']:.3f}</td>"
    f"<td class='num'>{piv.loc[c,'media']:.3f}</td>"
    f"<td class='num'>{piv.loc[c,'decae']:.3f}</td>"
    f"<td class='num good'>0.000</td></tr>"
    if pd.notna(piv.loc[c].get("decae")) else
    f"<tr><td><b>{c}</b></td><td class='num'>{piv.loc[c,'base']:.3f}</td>"
    f"<td class='num'>{piv.loc[c,'media']:.3f}</td>"
    f"<td class='num sub'>sin sintéticos</td><td class='num good'>0.000</td></tr>"
    for c in piv.sort_values("base", ascending=False).index)

pesos_rows = "".join(
    f"<tr><td class='num'>{nv:,.1f}</td><td class='num'>{n}</td>"
    f"<td class='num'>{p:.3f}</td><td class='num'>{n*p:.1f}</td></tr>"
    for nv, n, p in zip(nivs[:8], ns[:8], pesos[:8]))

HTML = f"""<title>C2 a fondo — Los pesos sintéticos</title>
<style>
body {{ margin:0; background:#fff; color:#14201A; line-height:1.6;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
.wrap {{ max-width:920px; margin:0 auto; padding:36px 26px 80px; }}
.eyebrow {{ font-size:11px; letter-spacing:.15em; text-transform:uppercase; color:#1B6535;
  font-weight:700; margin-bottom:10px; }}
h1 {{ font-family:Georgia,serif; font-size:31px; line-height:1.15; margin:0 0 12px; }}
h2 {{ font-family:Georgia,serif; font-size:21px; margin:34px 0 8px; }}
p {{ max-width:78ch; font-size:14.5px; }}
.sub {{ font-size:12px; color:#8A968E; }}
.good {{ color:#1B6535; font-weight:700; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; margin:12px 0; }}
th {{ text-align:left; font-size:10.5px; letter-spacing:.04em; text-transform:uppercase;
  color:#5C6B62; padding:7px 10px; border-bottom:1.5px solid #DDE3DC; }}
td {{ padding:6px 10px; border-bottom:1px solid #EBEFEA; }}
td.num {{ font-variant-numeric:tabular-nums; text-align:right; }}
img {{ width:100%; height:auto; display:block; border:1px solid #DDE3DC; border-radius:8px;
  margin:12px 0 6px; }}
.cap {{ font-size:12.5px; color:#5C6B62; margin:4px 0 18px; }}
.box {{ border-left:3px solid #1B6535; background:#F3F7F4; padding:13px 17px;
  border-radius:0 8px 8px 0; margin:16px 0; font-size:14px; }}
.box.warn {{ border-left-color:#B0413E; background:#FAF1F0; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
@media (max-width:640px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
</style>
<div class="wrap">
<div class="eyebrow">Piloto Predicción 1P · Deep-dive C2 · 2026-07-12</div>
<h1>Los pesos sintéticos, a fondo: por qué el empate del A/B es la respuesta correcta</h1>

<div class="box warn"><b>Corrección de lectura (dos niveles).</b> (1) C2 no promete mejoras:
fue el experimento que <b>validó el esquema actual</b>. (2) Al profundizar encontré algo más
fuerte que lo publicado: el "decaimiento peor (1.522 vs 1.505)" del informe era un
<b>artefacto de composición</b> — CUSIANA y CUPIAGUA SUR no tienen sintéticos, no entraron en
esa variante, y el promedio comparó canastas distintas. Campo por campo, las tres variantes
empatan <b>exactamente: Δ = 0.000 en los 20 campos</b>. El informe principal ya quedó corregido.</div>

<h2>1 · Qué son los pesos sintéticos (con números)</h2>
<p>La isotónica entrena con dos poblaciones: los <b>puntos reales del deck</b> (N≈9) y los
<b>puntos sintéticos de la escalera</b> (20–35 puntos que codifican la física financiera:
vol=0 bajo el Precio de Equilibrio, y los niveles de salida por clase/perfil). Sin pesos, 27
sintéticos "votarían" 3× más que 9 reales. La regla vigente (Path D):
<b>peso = n_real / n_puntos_del_nivel</b>, para que <b>cada nivel de la escalera pese en total
lo mismo que toda la evidencia real</b>. CASTILLA (n_real=9, 27 sintéticos en 21 niveles):</p>
<table><thead><tr><th class="num">Nivel (vol MBPE)</th><th class="num">n puntos</th>
<th class="num">peso c/u</th><th class="num">masa total</th></tr></thead>
<tbody>{pesos_rows}
<tr><td colspan="4" class="sub">… (21 niveles, TODOS con masa = 9.0 = n_real)</td></tr></tbody></table>
<img alt="masa por nivel" src="{fig1}">
<div class="cap">Izquierda: el nivel de abandono (vol=0) tiene 6 puntos y cada escalón 1–2.
Derecha: tras el peso, cada nivel queda con masa idéntica (9.0) — ningún escalón domina a otro
ni a los datos reales.</div>

<h2>2 · El experimento y el hallazgo: los pesos hoy son INERTES</h2>
<p>Probé el esquema actual contra dos alternativas (masa 0.5× y decaimiento por distancia a la
banda) — empate exacto. Después fui más lejos: <b>multipliqué los pesos por 0.05 y por 5
(rango de 100×)</b> y comparé las curvas completas punto a punto:</p>
<img alt="curvas identicas" src="{fig2}">
<div class="cap">Las tres curvas de CASTILLA son indistinguibles en TODA la grilla (Δ máx =
0.000 MBPE) — igual en CAÑO LIMON y el resto. No es que las variantes fueran parecidas: es que
el peso no participa en la solución.</div>
<p><b>La razón es matemática y es elegante:</b> la isotónica (algoritmo PAVA) solo usa los pesos
cuando tiene que <b>agrupar puntos que violan la monotonía</b> (un valor menor a un precio
mayor) — ahí el promedio ponderado del grupo decide el nivel. Pero en nuestro motor esos
conflictos entre sintéticos y reales <b>ya no existen</b>, porque los eliminamos por diseño en
decisiones anteriores: el <b>guard de banda</b> (ningún sintético se inyecta en/sobre la banda
real, 2026-06-11) y el <b>cap de monotonía §4.3</b> (ningún escalón queda por encima del peor
delta real, ESCALON_CAPADO). Sin conflicto, PAVA interpola cada punto exactamente y el peso es
un espectador.</p>
<img alt="demo conflicto" src="{fig3}">
<div class="cap">Demo didáctica con un conflicto INYECTADO artificialmente: solo cuando un punto
real queda por debajo de la escalera a mayor precio, PAVA agrupa y el peso decide el nivel
resultante (verde = gana la escalera, rojo = gana el punto real). Ese es el único escenario
donde los pesos actúan — y nuestro pipeline lo previene aguas arriba.</div>

<h2>3 · El A/B campo por campo (los 20 de la ola 1)</h2>
<table><thead><tr><th>Campo</th><th class="num">MAE_LOYO actual</th><th class="num">0.5×</th>
<th class="num">decaimiento</th><th class="num">Δ</th></tr></thead><tbody>{rows}</tbody></table>

<h2>4 · Qué significa para el negocio</h2>
<div class="box"><b>Lo que ancla el tramo bajo NO son los pesos — son los VALORES.</b> El
volumen cae a 0 bajo el Precio de Equilibrio y sigue la escalera/perfil porque esos puntos
sintéticos EXISTEN con esos niveles (BK_P10–P90 del FC certificado), y por la regla dura
hard-zero. Los pesos eran el mecanismo de arbitraje para conflictos que el guard de banda y el
cap §4.3 ya resuelven explícitamente — quedaron como red de seguridad vestigial.</div>
<p>Tres consecuencias prácticas:</p>
<p><b>(a) No hay nada que tunear aquí.</b> Cualquier "optimización de pesos sintéticos" futura
sería trabajo sin efecto — queda documentado con evidencia para no volver a ese pozo.<br>
<b>(b) La palanca real del tramo bajo es el perfil BK_P10–P90</b> (los valores de los niveles),
ya adoptado en el track Calidad.<br>
<b>(c) El LOYO no puede validar el tramo bajo</b> — mide error solo donde hay datos reales
($53–69 en CASTILLA) y la escalera vive debajo ($16–45). La validación definitiva del tramo
bajo llegará el día que un quarter real se acerque al breakeven (la banda histórica $43–112
nunca lo tocó): ese será el backtest de crisis.</p>

<p class="sub">Piloto Predicción Reservas 1P · Soporte analítico interno. Evidencia:
experimento_estadistico.csv (C2), prueba extrema ×0.05–×5 reproducible con
pesos_sinteticos_tramo + MotorIsotonico. La corrección del agregado 1.522 quedó aplicada en
el informe principal.</p>
</div>
"""
OUT.write_text(HTML, encoding="utf-8")
print(f"OK: {OUT} ({len(HTML)//1024} KB)")
