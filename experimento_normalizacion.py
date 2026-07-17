"""
experimento_normalizacion.py — Experimento offline v2: ¿normalizar el target contra el
cierre del AÑO ANTERIOR (Baseline) en vez del cierre del mismo año (Checkpoint) reduce el
confound de vigencia? (Modelo 1)

Contexto (usuario 2026-07-09):
  El target de producción restaba el cierre OFICIAL del MISMO año A (hoy CHECKPOINT_1P_MBPE).
  Ese salto de certificado A−1→A es el confound de vigencia: mezcla el efecto precio con el
  re-basado anual del deck. El análisis del usuario en Hoja1 del Consolidado normaliza contra
  el cierre del año ANTERIOR (A−1), y además normaliza el eje de precio (%Δ vs ancla A−1).
  Este experimento validó esa idea → ADOPTADA EN PRODUCCIÓN el 2026-07-09 (s2): el pipeline
  entrena DELTA_SENS = VOL_SENS − BASELINE_1P_MBPE (variante DBASE). Se conserva como
  herramienta de diagnóstico de variantes.

Terminología (decisión 2026-07-09):
  BASELINE   = cierre OFICIAL del año pasado (A−1).            ← BASELINE_1P_MBPE (producción)
  CHECKPOINT = cierre OFICIAL del presente año (A).            ← CHECKPOINT_1P_MBPE

Este script NO altera el pipeline (01–04 conservan sus nombres hasta directriz de finanzas;
memoria: cambios-modelo-requieren-directriz). Solo experimento + CSV de diagnóstico.

Se comparan SEIS variantes para la isotónica de M1, sobre los mismos puntos reales. El diseño
es 2×2 (ancla CHK vs BASE) × (escala Δ vs %) + réplica Hoja1 (normaliza también el precio) +
volumen absoluto de referencia:

  MΔ_CHK   VOL − CHECKPOINT(A)            feat precio $   producción actual (referencia)
  Mpct_CHK (VOL − CHK)/CHK               feat precio $   %delta v1 (referencia)
  MΔ_BASE  VOL − BASELINE(A−1)           feat precio $   aísla efecto ANCLA
  Mpct_BASE(VOL − BASE)/BASE            feat precio $   corrección pedida (fila 13 Hoja1)
  M_HOJA1  (VOL − BASE)/BASE             feat %Δprecio   réplica Hoja1 (normaliza ambos ejes)
  Mvol     VOL                           feat precio $   volumen absoluto (baseline común)

Diagnóstico por campo:
  - ETA2 del AÑO sobre cada target (magnitud del confound; 1 = el año lo explica todo).
  - SOLAPA de precio entre años en $ (invariante a normalización) y en % (clave: ver si
    normalizar el eje precio crea solape entre años donde en $ no lo hay).
  - SKILL_LOO medido SIEMPRE en volumen absoluto (comparable entre variantes). Dos lecturas:
      SKILL_*      : LOO sobre el máximo set usable de cada variante (los _BASE ganan 2026).
      SKILL_COMUN_*: LOO sobre el subconjunto COMÚN (quarters con CHK y BASE presentes) →
                     comparación apples-to-apples cuando el N difiere.
  - RANGO de la curva de volumen en la banda (MBPE): sensibilidad real al precio.

CRITERIO: una variante solo es mejor si sube SKILL/RANGO SIN fuga de etiqueta de año y SIN
degradar el gate dorado.

Salida: resultados/experimento_normalizacion.csv
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from homologacion import Homologador
from motores_modelo1 import MotorIsotonico

BASE_DIR   = Path(__file__).parent
STAGING    = BASE_DIR / "datos" / "staging"
RESULTADOS = BASE_DIR / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

modelo = importlib.import_module("03_modelo")

FEATURE   = "PRECIO_NETO_USD_BBL"
DELTA     = "DELTA_SENS_MBPE"
VOL       = "VOLUMEN_1P_SENSIBILIDAD_MBPE"
CHECKPOINT = "CHECKPOINT_1P_MBPE"   # cierre A (renombrado en el pipeline 2026-07-09)
OFICIAL   = "VOLUMEN_1P_OFICIAL_MBPE"

# Gate Dorado = pareto-10 (directriz 2026-07-09; ver docs/NORTE.md)
GATE_DORADO = ["RUBIALES", "CASTILLA", "CAÑO SUR ESTE", "CASTILLA NORTE", "AKACIAS",
               "CHICHIMENE", "CHICHIMENE SW", "LA CIRA", "CUPIAGUA", "YARIGUI-CANTAGALLO"]
FOCO        = ["AKACIAS", "CUPIAGUA", "RUBIALES"]

# Exclusiones permanentes del piloto (memoria: exclusiones-permanentes-analisis).
# Filiales se detectan por NEW GERENCIA vía Homologador; PAUTO SUR y el gas son explícitos.
EXCLUIR_EXPLICITO = {"PAUTO SUR", "BALLENA", "CHUCHUPA"}

# Registro de variantes: (kind, ancla, feature). ancla None = volumen absoluto.
#   kind   : 'delta' -> vol_pred = ancla + f(x);  'pct' -> vol_pred = ancla*(1+f(x));  'vol' -> f(x)
#   ancla  : 'chk' (cierre A) | 'base' (cierre A−1) | None
#   feat   : 'precio' (precio neto $) | 'pctprice' ((p − p_base)/p_base)
VARIANTES = {
    "DCHK":  ("delta", "chk",  "precio"),
    "PCHK":  ("pct",   "chk",  "precio"),
    "DBASE": ("delta", "base", "precio"),
    "PBASE": ("pct",   "base", "precio"),
    "HOJA1": ("pct",   "base", "pctprice"),
    "VOL":   ("vol",   None,   "precio"),
}


def eta2(grupos: np.ndarray, y: np.ndarray) -> float:
    """Fracción de la varianza de y explicada por el grupo (año). 1 = el año lo explica todo."""
    if len(y) < 2:
        return np.nan
    grand = float(np.mean(y))
    ss_tot = float(np.sum((y - grand) ** 2))
    if ss_tot < 1e-9:
        return np.nan
    ss_bet = sum(len(y[grupos == g]) * (float(np.mean(y[grupos == g])) - grand) ** 2
                 for g in np.unique(grupos))
    return ss_bet / ss_tot


def solapa(valor: np.ndarray, anios: np.ndarray) -> float:
    """Mayor solape de banda entre pares de años. 0 = bandas separadas."""
    us = np.unique(anios)
    if len(us) < 2:
        return np.nan
    bandas = {a: (float(valor[anios == a].min()), float(valor[anios == a].max())) for a in us}
    mejor = 0.0
    for i, a in enumerate(us):
        for b in us[i + 1:]:
            lo = max(bandas[a][0], bandas[b][0]); hi = min(bandas[a][1], bandas[b][1])
            mejor = max(mejor, hi - lo)
    return mejor


def _skill(vol_true, vol_pred, vol_naive):
    mae = mean_absolute_error(vol_true, vol_pred)
    mae_n = mean_absolute_error(vol_true, vol_naive)
    return (1 - mae / mae_n) if mae_n > 0.01 else np.nan


def loo_vol(pts: pd.DataFrame, kind: str, ancla_col: str, feat: str,
            base_latest: float, p_latest: float, score_mask: np.ndarray = None) -> tuple:
    """LOO sobre puntos reales; predice el VOLUMEN ABSOLUTO por variante y lo puntúa contra el
    volumen real (comparable entre variantes). El ajuste usa SIEMPRE todos los puntos de `pts`;
    `score_mask` (opcional) restringe QUÉ puntos se puntúan (para SKILL_COMUN sobre el
    subconjunto común CHK∩BASE). Devuelve (skill, rango_curva_vol).
    """
    r = pts.reset_index(drop=True)
    n = len(r)
    if n < 2:
        return np.nan, np.nan
    vol_true = r[VOL].values
    ancla = r[ancla_col].values if ancla_col else None

    # Feature según la variante
    if feat == "pctprice":
        pb = r["P_BASE"].values
        x = (r[FEATURE].values - pb) / pb
    else:
        x = r[FEATURE].values

    vol_pred = np.zeros(n); vol_naive = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool); m[i] = False
        if kind == "delta":
            y = vol_true[m] - ancla[m]
        elif kind == "pct":
            y = (vol_true[m] - ancla[m]) / ancla[m]
        else:
            y = vol_true[m]
        f = MotorIsotonico().fit(x[m], y)
        yi = float(f.predict([x[i]])[0])
        if kind == "delta":
            vol_pred[i] = ancla[i] + yi
        elif kind == "pct":
            vol_pred[i] = ancla[i] * (1.0 + yi)
        else:
            vol_pred[i] = yi
        vol_naive[i] = float(np.mean(vol_true[m]))

    sel = score_mask if score_mask is not None else np.ones(n, dtype=bool)
    if sel.sum() < 1:
        skill = np.nan
    else:
        skill = _skill(vol_true[sel], vol_pred[sel], vol_naive[sel])

    # Curva de volumen sobre la banda observada (sensibilidad real al precio), re-anclada al
    # certificado más reciente (base_latest) y su precio (p_latest) — como producción.
    if kind == "delta":
        y_all = vol_true - ancla
    elif kind == "pct":
        y_all = (vol_true - ancla) / ancla
    else:
        y_all = vol_true
    f_all = MotorIsotonico().fit(x, y_all)
    grid_p = np.linspace(float(r[FEATURE].min()), float(r[FEATURE].max()), 50)
    grid_x = (grid_p - p_latest) / p_latest if feat == "pctprice" else grid_p
    yg = f_all.predict(grid_x)
    if kind == "delta":
        vol_g = base_latest + yg
    elif kind == "pct":
        vol_g = base_latest * (1.0 + yg)
    else:
        vol_g = yg
    return skill, float(vol_g.max() - vol_g.min())


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # consola cp1252 no imprime −/Δ/∩
    except Exception:
        pass
    print("=== experimento_normalizacion.py v2 — Baseline(A−1) vs Checkpoint(A) ===\n")
    ruta = STAGING / "tablon_unico.parquet"
    if not ruta.exists():
        raise FileNotFoundError("Ejecutar 01_etl.py y 02_synthetic.py primero.")
    df = pd.read_parquet(ruta)

    # ── Lookups OFICIALES por (campo, año) desde filas ES_BASELINE (serie HIST) ────────────
    base_rows = df[df["ES_BASELINE"] & df[OFICIAL].notna()]
    oficial   = {(c, int(a)): float(v) for c, a, v in
                 zip(base_rows["CAMPO"], base_rows["AÑO"], base_rows[OFICIAL])}
    pneto_of  = {(c, int(a)): float(v) for c, a, v in
                 zip(base_rows["CAMPO"], base_rows["AÑO"], base_rows[FEATURE])}
    # cierre del AÑO MÁXIMO por campo (fix: antes se usaba .max() sobre valores = máx histórico)
    max_anio = base_rows.groupby("CAMPO")["AÑO"].max().astype(int).to_dict()

    # ── Selección de campos: gate + foco + top-20 materiales, con exclusiones ──────────────
    h = Homologador()

    def es_filial(campo: str) -> bool:
        return str(h.atributos(campo).get("NEW GERENCIA", "")).strip().upper() == "FILIAL"

    insens = (df.groupby("CAMPO")["BRENT_INSENSITIVE"]
                .apply(lambda s: bool(s.dropna().astype(bool).any())))

    def excluido(campo: str) -> bool:
        return (campo in EXCLUIR_EXPLICITO or es_filial(campo)
                or bool(insens.get(campo, False)))

    base_latest_all = {c: oficial.get((c, a)) for c, a in max_anio.items()}
    materiales = [c for c in sorted(base_latest_all, key=lambda k: base_latest_all[k] or 0,
                                    reverse=True) if not excluido(c)][:20]
    campos = [c for c in dict.fromkeys(GATE_DORADO + FOCO + materiales) if not excluido(c)]

    registros = []
    for campo in campos:
        sub = df[(df["CAMPO"] == campo) & (~df["ES_SINTETICO"]) & (~df["ES_BASELINE"])
                 & df[VOL].notna() & df[FEATURE].notna()].copy()
        if sub.empty:
            continue
        sub["AÑO_V"] = sub["VIGENCIA"].astype(str).str[:4].astype(int)
        sub["CHK"]    = sub[CHECKPOINT]
        sub["BASE"]   = sub["AÑO_V"].apply(lambda A: oficial.get((campo, A - 1), np.nan))
        sub["P_BASE"] = sub["AÑO_V"].apply(lambda A: pneto_of.get((campo, A - 1), np.nan))

        a_max = max_anio.get(campo)
        base_latest = oficial.get((campo, a_max), np.nan)
        p_latest    = pneto_of.get((campo, a_max), np.nan)
        if not (np.isfinite(base_latest) and base_latest > 0 and np.isfinite(p_latest)):
            continue

        n_total = len(sub)
        if n_total < 4 or sub["AÑO_V"].nunique() < 2:
            continue

        m_chk  = sub["CHK"].notna().values
        m_base = sub["BASE"].notna().values & sub["P_BASE"].notna().values
        m_comun = m_chk & m_base
        n_comun = int(m_comun.sum())
        n_2026  = int((sub["AÑO_V"].values == 2026).sum())
        n_sin_base = int((~m_base).sum())

        rec = {
            "CAMPO": campo, "N_REAL": n_total, "N_COMUN": n_comun, "N_2026": n_2026,
            "N_SIN_BASE_A1": n_sin_base, "BASELINE_MBPE": round(base_latest, 1),
            "SOLAPA_PRECIO_USD": round(solapa(sub[FEATURE].values, sub["AÑO_V"].values), 2),
        }
        # Solapa en % (sobre puntos con BASE): (p − p_base)/p_base por año
        if m_base.sum() >= 2:
            sb = sub[m_base]
            pctp = ((sb[FEATURE].values - sb["P_BASE"].values) / sb["P_BASE"].values)
            rec["SOLAPA_PRECIO_PCT"] = round(solapa(pctp, sb["AÑO_V"].values), 4)
        else:
            rec["SOLAPA_PRECIO_PCT"] = np.nan

        comun_sub = sub[m_comun]
        for tag, (kind, ancla, feat) in VARIANTES.items():
            ancla_col = {"chk": "CHK", "base": "BASE", None: None}[ancla]
            usable = np.ones(n_total, bool) if ancla is None else (m_chk if ancla == "chk" else m_base)
            u = sub[usable]

            # ETA2 del año sobre el target de la variante (en su set usable)
            if len(u) >= 2:
                if kind == "delta":
                    yt = u[VOL].values - u[ancla_col].values
                elif kind == "pct":
                    yt = (u[VOL].values - u[ancla_col].values) / u[ancla_col].values
                else:
                    yt = u[VOL].values
                rec[f"ETA2_{tag}"] = round(eta2(u["AÑO_V"].values, yt), 3)
            else:
                rec[f"ETA2_{tag}"] = np.nan

            # SKILL/RANGO sobre el máximo set usable
            if len(u) >= 2 and u["AÑO_V"].nunique() >= 1:
                sk, rg = loo_vol(u, kind, ancla_col, feat, base_latest, p_latest)
            else:
                sk, rg = np.nan, np.nan
            rec[f"SKILL_{tag}"] = round(sk, 3) if pd.notna(sk) else None
            rec[f"RANGO_{tag}"] = round(rg, 1) if pd.notna(rg) else None

            # SKILL_COMUN: LOO sobre el subconjunto común (apples-to-apples)
            if n_comun >= 2:
                skc, _ = loo_vol(comun_sub, kind, ancla_col, feat, base_latest, p_latest)
                rec[f"SKILL_COMUN_{tag}"] = round(skc, 3) if pd.notna(skc) else None
            else:
                rec[f"SKILL_COMUN_{tag}"] = None

        registros.append(rec)

    res = pd.DataFrame(registros)
    res.to_csv(RESULTADOS / "experimento_normalizacion.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 60)
    print(res.to_string(index=False))
    print(f"\n  Guardado: {RESULTADOS / 'experimento_normalizacion.csv'}")

    # ── Lectura del criterio ──────────────────────────────────────────────────────────────
    print(f"\n{'-'*72}\n  Lectura del criterio (sobre subconjunto común CHK∩BASE)\n{'-'*72}")
    val = res[res["SKILL_COMUN_DCHK"].notna()]
    if not val.empty:
        for tag, etiqueta in [("PBASE", "Mpct_BASE"), ("HOJA1", "M_HOJA1"),
                              ("DBASE", "MΔ_BASE"), ("VOL", "Mvol")]:
            col = f"SKILL_COMUN_{tag}"
            comp = val[val[col].notna()]
            mej = (comp[col] > comp["SKILL_COMUN_DCHK"] + 0.02).sum()
            print(f"  {etiqueta:10s} mejora skill vs MΔ_CHK (>+0.02): {mej}/{len(comp)}")
        sep = res[res["SOLAPA_PRECIO_USD"] < 2.0]
        sep_pct = res[res["SOLAPA_PRECIO_PCT"].fillna(9) < 0.02]
        print(f"\n  Bandas de precio separadas en $ (<$2): {len(sep)}/{len(res)}")
        print(f"  Bandas de precio separadas en % (<2%): {len(sep_pct)}/{len(res)}")
    print("\n  NOTA: SOLAPA~0 es el límite de identificación; ninguna normalización crea")
    print("  variación de precio donde los años no se solapan. Cambiar el ancla (CHK→BASE)")
    print("  reubica el nivel; cambiar la escala (Δ→%) reescala; normalizar el eje precio")
    print("  (M_HOJA1) puede crear solape en % que en $ no existe — ver SOLAPA_PRECIO_PCT.")
    print("\n=== experimento_normalizacion.py v2 — Completado ===")
