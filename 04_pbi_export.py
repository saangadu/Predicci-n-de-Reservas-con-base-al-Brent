"""
04_pbi_export.py — Matrices de prediccion para Power BI (Modelo 2 + Modelo 1 + cadena)

Pre-calcula las combinaciones (Brent x Campo x Motor) como CSV estatico.
PBI consume el CSV con DAX + What-If slider de Brent.

ARQUITECTURA (2026-06-11; reformada 2026-06-12 — sin escenarios, con re-anclaje):
  Para cada Brent de la grilla:
    1. Modelo 2 (03b_correlacion_brent.py): Brent -> PRECIO ACEITE por campo
       (recta unica BASE; escenarios BAJO/ALTO retirados).
    2. Modelo 1 (03_modelo.py): PRECIO ACEITE -> DELTA Reservas. Motores 1D:
       - Isotonica (PRIMARIO), Suave/PCHIP (VALIDACION). XGBoost retirado.
    3. RE-ANCLAJE: VOLUMEN = max(BASELINE + [f(p) − f(p_ref)], 0), piso duro
       p < BK_ANCLA_PDP -> 0. p_ref/delta_ref vienen de metricas.csv (03).
       Garantia: en Brent=BRENT_REF (ultimo quarter conocido) Vol = baseline exacto.

  ES_VIABLE        = precio_aceite >= BREAKEVEN_OPERACIONAL (piso inferior/abandono).
  ES_FULL_RESERVAS = precio_aceite >= BREAKEVEN_FINANCIERO  (piso superior).
  ES_EXTRAPOLADO   = Brent fuera de la banda observada del Consolidado por campo ± margen.
  M2_ES_FALLBACK   = el campo no tiene relacion Aceite~Brent propia (k de portafolio).

TRES MATRICES (aislar el origen de errores, directriz 2026-06-12):
  output_matriz_modelo1.csv   grilla Precio Aceite -> delta anclado/volumen (M1 puro)
  output_matriz_modelo2.csv   grilla Brent -> Precio Aceite + metricas (M2 puro)
  output_matriz_prediccion.csv  cadena completa Brent -> Aceite -> Volumen

Confianza por campo: usa MAE_LOO del PRIMARIO (Isotonica) y la divergencia Isotonica↔Suave.
"""

import importlib
import shutil
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from motores_modelo1 import volumen_anclado

BASE_DIR    = Path(__file__).parent
STAGING     = BASE_DIR / "datos" / "staging"
MODELOS_DIR = STAGING / "modelos"
RESULTADOS  = BASE_DIR / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

# Modelo 2: predictor Brent -> Neto (import dinamico: el nombre empieza por digito)
m2 = importlib.import_module("03b_correlacion_brent")

BRENT_PASO = 1
MARGEN_BRENT_USD  = 10    # margen sobre la banda del Consolidado para el meshgrid
MARGEN_EXTRAP_USD = 5.0   # margen para marcar ES_EXTRAPOLADO por campo

# ── Umbrales de confianza (ajustables) ───────────────────────────────────────
CONFIANZA_N_REAL_MIN    = 6
CONFIANZA_BASELINE_MIN  = 0.5
CONFIANZA_MAE_REL_ALTA  = 0.20
CONFIANZA_MAE_REL_MEDIA = 0.40
CONFIANZA_DIV_ALTA      = 0.30
CONFIANZA_DIV_MEDIA     = 0.50
CONFIANZA_MAE_ABS_ALTA  = 2.0
CONFIANZA_MAE_ABS_MEDIA = 5.0
CONFIANZA_OUTLIER_RATIO = 2.0
# Cap de outlier condicionado a materialidad (2026-06-11): el flag OUTLIER_LOO
# (RMSE/MAE>2) solo degrada si el error relativo es material. Caso RUBIALES: el
# ratio lo dispara un unico fold en la frontera de vigencias (revision +21.8 MBPE
# del baseline 2025 → a mismo neto conviven deltas de dos vigencias) con MAE_REL
# 0.9% sobre 323 MBPE — inmaterial para CAPEX.
CONFIANZA_MAE_REL_OUTLIER = 0.05
# Gate de skill (2026-06-11): ALTA exige ademas que el modelo SUPERE al predictor
# ingenuo (SKILL_ISO > 0) — error bajo + motores de acuerdo no implican que el
# modelo extraiga la señal de precio (caso AKACIAS: MAE_rel 12% pero SKILL=0).
# Exencion: campos "planos" (MAE_NAIVE < CONFIANZA_MAE_ABS_ALTA) — alli la media
# ingenua es imbatible por construccion y SKILL<=0 no indica un mal modelo.
CONFIANZA_SKILL_MIN = 0.0

HISTORICO_DIR = RESULTADOS / "historico_predicciones"

# Motores 1D: PRIMARIO (Isotonica) y VALIDACION (Suave). Sufijo de archivo joblib y label.
MOTORES = [("Isotonica", "iso"), ("Suave", "suave")]

# Grilla de Precio Aceite para la matriz M1 pura (independiente del Brent)
PNETO_GRID_MIN, PNETO_GRID_MAX, PNETO_GRID_PASO = 20.0, 110.0, 1.0


def siguiente_quarter(q: str) -> str:
    """Avanza un quarter: '2026_Q1' -> '2026_Q2', '2025_Q4' -> '2026_Q1'."""
    anio, qn = q.split("_Q")
    anio, qn = int(anio), int(qn)
    if qn == 4:
        return f"{anio + 1}_Q1"
    return f"{anio}_Q{qn + 1}"


def derivar_vigencias(df: pd.DataFrame) -> tuple[str, str]:
    """Maximo quarter de los escenarios CONSOLIDADO_* del tablon -> (base, objetivo)."""
    qs = [s.replace("CONSOLIDADO_", "")
          for s in df["ESCENARIO"].dropna().unique()
          if str(s).startswith("CONSOLIDADO_")]
    if not qs:
        return "DESCONOCIDA", "DESCONOCIDA"
    vigencia_base = sorted(qs)[-1]
    return vigencia_base, siguiente_quarter(vigencia_base)


def cargar_correlacion() -> dict:
    """Coeficientes del Modelo 2 por campo (correlacion_brent.csv): {campo: dict_coef}."""
    ruta = STAGING / "correlacion_brent.csv"
    if not ruta.exists():
        raise FileNotFoundError("correlacion_brent.csv no existe: correr 03b_correlacion_brent.py")
    df = pd.read_csv(ruta)
    return {r["CAMPO"]: r.to_dict() for _, r in df.iterrows()}


def cargar_anclas(df: pd.DataFrame) -> dict:
    """Anclas (BK_ANCLA_FIN superior, BK_ANCLA_PDP inferior) por campo, vigencia reciente."""
    sub = df.dropna(subset=["BK_ANCLA_FIN_USD_BBL", "BK_ANCLA_PDP_USD_BBL"], how="all")
    out = {}
    for campo, g in sub.groupby("CAMPO"):
        fila = g.sort_values("VIGENCIA_BREAKEVEN", ascending=False).iloc[0]
        bk_fin = float(fila["BK_ANCLA_FIN_USD_BBL"]) if pd.notna(fila["BK_ANCLA_FIN_USD_BBL"]) else np.nan
        bk_pdp = float(fila["BK_ANCLA_PDP_USD_BBL"]) if pd.notna(fila["BK_ANCLA_PDP_USD_BBL"]) else np.nan
        if np.isnan(bk_fin):
            bk_fin = bk_pdp
        if np.isnan(bk_pdp):
            bk_pdp = bk_fin
        out[campo] = (bk_fin, bk_pdp)
    return out


def cargar_ponderados(df: pd.DataFrame) -> dict:
    """Breakevens PONDERADOS 1P globales (D6, referencia de reporte) por campo."""
    sub = df.dropna(subset=["BREAKEVEN_FINANCIERO_USD_BBL",
                            "BREAKEVEN_OPERACIONAL_USD_BBL"], how="all")
    out = {}
    for campo, g in sub.groupby("CAMPO"):
        fila = g.sort_values("VIGENCIA_BREAKEVEN", ascending=False).iloc[0]
        fin = float(fila["BREAKEVEN_FINANCIERO_USD_BBL"]) \
            if pd.notna(fila["BREAKEVEN_FINANCIERO_USD_BBL"]) else np.nan
        ope = float(fila["BREAKEVEN_OPERACIONAL_USD_BBL"]) \
            if pd.notna(fila["BREAKEVEN_OPERACIONAL_USD_BBL"]) else np.nan
        out[campo] = (fin, ope)
    return out


def cargar_baselines(df: pd.DataFrame) -> dict:
    """Ultimo VOLUMEN_1P_OFICIAL_MBPE certificado por campo."""
    df_b = df[(df["ESCENARIO"] == "BASE") & df["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
    return (df_b.sort_values("AÑO")
            .groupby("CAMPO")["VOLUMEN_1P_OFICIAL_MBPE"].last().to_dict())


def banda_historica_brent(df: pd.DataFrame) -> dict:
    """Rango de Brent observado en CONSOLIDADO real por campo (para ES_EXTRAPOLADO)."""
    df_real = df[(~df["ES_SINTETICO"]) & (~df["ES_BASELINE"]) & df["BRENT_FLAT_USD_BBL"].notna()]
    out = {}
    for campo, sub in df_real.groupby("CAMPO"):
        out[campo] = (float(sub["BRENT_FLAT_USD_BBL"].min()),
                      float(sub["BRENT_FLAT_USD_BBL"].max()))
    return out


def clasificar_confianza(n_real, mae_rel, divergencia, baseline,
                         mae_abs=999.0, outlier_lloo=False,
                         skill=np.nan, mae_naive=np.nan) -> tuple[str, str]:
    """Clasifica la confianza usando criterios funcionales del piloto (MAESTRO §7.4).

    Gate de skill (2026-06-11): ALTA exige SKILL_ISO > 0 o campo "plano"
    (MAE_NAIVE pequeño — la media ingenua es imbatible por construccion alli).
    SKILL NaN (sin reales suficientes o naive≈0) no penaliza."""
    if mae_abs < CONFIANZA_MAE_ABS_ALTA:
        mae_rel_eff = min(mae_rel, CONFIANZA_MAE_REL_ALTA * 0.99)
    elif mae_abs < CONFIANZA_MAE_ABS_MEDIA:
        mae_rel_eff = min(mae_rel, CONFIANZA_MAE_REL_MEDIA * 0.99)
    else:
        mae_rel_eff = mae_rel

    # Outlier inmaterial: el ratio RMSE/MAE dispara pero el error relativo es
    # despreciable para CAPEX → no degrada, solo queda en el motivo (auditable).
    outlier_material = outlier_lloo and mae_rel >= CONFIANZA_MAE_REL_OUTLIER

    partes = [f"N={n_real}", f"MAE_rel={mae_rel:.2f}", f"MAE_abs={mae_abs:.2f}MBPE",
              f"div={divergencia:.2f}", f"base={baseline:.1f}MBPE"]
    if pd.notna(skill):
        partes.append(f"skill={skill:.2f}")
    if outlier_lloo:
        partes.append("OUTLIER_LOO" if outlier_material else "OUTLIER_LOO_INMATERIAL")
    # Separador " | " (no ";"): Excel es-CO interpreta ";" como delimitador de
    # columnas y parte la celda MOTIVO_CONFIANZA al abrir el CSV (ver MAESTRO §10).
    motivo = " | ".join(partes)

    if n_real == 0:
        return "SOLO_SINTETICO", f"sin datos reales | {motivo}"
    if baseline < CONFIANZA_BASELINE_MIN:
        return "BAJA", f"micro-campo | {motivo}"
    if outlier_material:
        if (mae_rel_eff < CONFIANZA_MAE_REL_MEDIA and divergencia < CONFIANZA_DIV_MEDIA):
            return "MEDIA", motivo
        return "BAJA", motivo

    # Gate de skill: falla solo si el modelo NO supera al ingenuo en un campo
    # con variacion material de deltas (naive grande). Campos planos exentos.
    sin_skill = (pd.notna(skill) and skill <= CONFIANZA_SKILL_MIN
                 and pd.notna(mae_naive) and mae_naive >= CONFIANZA_MAE_ABS_ALTA)

    if (n_real >= CONFIANZA_N_REAL_MIN and mae_rel_eff < CONFIANZA_MAE_REL_ALTA
            and divergencia < CONFIANZA_DIV_ALTA and not sin_skill):
        return "ALTA", motivo
    if (mae_rel_eff < CONFIANZA_MAE_REL_MEDIA and divergencia < CONFIANZA_DIV_MEDIA):
        if sin_skill:
            return "MEDIA", f"sin skill vs ingenuo | {motivo}"
        return "MEDIA", motivo
    return "BAJA", motivo


def calcular_divergencia_motores(df_out: pd.DataFrame) -> pd.Series:
    """Divergencia media |Suave - Isotonica| / Isotonica por campo en banda Brent $40-$80.
    Retorna Serie indexada por CAMPO."""
    banda = df_out[(df_out["BRENT_USD_BBL"] >= 40) & (df_out["BRENT_USD_BBL"] <= 80)]
    piv = (banda.pivot_table(index=["CAMPO", "BRENT_USD_BBL"], columns="MOTOR",
                             values="VOLUMEN_1P_PREDICHO_MBPE")
           .reset_index().dropna())
    if piv.empty or "Isotonica" not in piv.columns or "Suave" not in piv.columns:
        return pd.Series(dtype=float)
    piv["div"] = (piv["Suave"] - piv["Isotonica"]).abs() / piv["Isotonica"].replace(0, np.nan)
    return piv.groupby("CAMPO")["div"].mean()


def _extraer_en_brent_ref(df: pd.DataFrame, brent_ref: float) -> pd.DataFrame:
    """Para cada CAMPO x MOTOR, fila con BRENT mas cercano a brent_ref (escenario BASE)."""
    sub = df
    if "ESCENARIO_DESCUENTO" in sub.columns:
        sub = sub[sub["ESCENARIO_DESCUENTO"] == "BASE"]
    sub = sub.copy()
    sub["_dist"] = (sub["BRENT_USD_BBL"] - brent_ref).abs()
    idx = sub.groupby(["CAMPO", "MOTOR"])["_dist"].idxmin()
    out = sub.loc[idx].reset_index(drop=True)
    for c in ["NIVEL_CONFIANZA", "Q_OBJETIVO"]:
        if c not in out.columns:
            out[c] = np.nan
    return out[["CAMPO", "MOTOR", "VOLUMEN_1P_PREDICHO_MBPE", "Q_OBJETIVO", "NIVEL_CONFIANZA"]]


def generar_comparacion_vs_anterior(df_out, q_objetivo, brent_ref, ruta_actual):
    """Compara la prediccion actual (BASE, Brent~ref) vs el snapshot previo mas reciente."""
    cols_out = ["CAMPO", "MOTOR", "Q_OBJETIVO_ANTERIOR", "Q_OBJETIVO_NUEVO",
                "VOL_ANTERIOR_MBPE", "VOL_NUEVO_MBPE", "DIF_ABS_MBPE", "DIF_PCT",
                "NIVEL_CONFIANZA_ANTERIOR", "NIVEL_CONFIANZA_NUEVO"]
    ruta_comp = RESULTADOS / "comparacion_vs_anterior.csv"
    previos = sorted((p for p in HISTORICO_DIR.glob("prediccion_*.csv")
                      if p.resolve() != ruta_actual.resolve()),
                     key=lambda p: p.stat().st_mtime)
    if not previos:
        pd.DataFrame(columns=cols_out).to_csv(ruta_comp, index=False, encoding="utf-8-sig")
        print(f"\n  [INFO] Sin snapshot previo: {ruta_comp} (solo encabezados)")
        return None

    ruta_prev = previos[-1]
    df_prev = pd.read_csv(ruta_prev)
    prev_ref  = _extraer_en_brent_ref(df_prev, brent_ref)
    nuevo_ref = _extraer_en_brent_ref(df_out, brent_ref)
    merged = nuevo_ref.merge(prev_ref, on=["CAMPO", "MOTOR"], how="outer",
                             suffixes=("_NUEVO", "_ANTERIOR"))
    merged = merged.rename(columns={
        "VOLUMEN_1P_PREDICHO_MBPE_NUEVO":    "VOL_NUEVO_MBPE",
        "VOLUMEN_1P_PREDICHO_MBPE_ANTERIOR": "VOL_ANTERIOR_MBPE"})
    merged["DIF_ABS_MBPE"] = merged["VOL_NUEVO_MBPE"] - merged["VOL_ANTERIOR_MBPE"]
    merged["DIF_PCT"] = (merged["DIF_ABS_MBPE"] /
                         merged["VOL_ANTERIOR_MBPE"].replace(0, np.nan) * 100)
    merged = merged[cols_out].sort_values("DIF_ABS_MBPE",
                                          key=lambda s: s.abs(), ascending=False,
                                          na_position="last")
    merged.to_csv(ruta_comp, index=False, encoding="utf-8-sig")
    print(f"\n  Comparacion vs corrida anterior ({ruta_prev.name}): {ruta_comp}")
    print(f"    @Brent~${brent_ref}: {len(merged)} filas comparadas")
    return merged


def actualizar_changelog(q_objetivo, fecha, n_campos, comp) -> None:
    """Agrega entrada fechada a docs/CHANGELOG_PREDICCIONES.md."""
    ruta = BASE_DIR / "docs" / "CHANGELOG_PREDICCIONES.md"
    if not ruta.exists():
        ruta.write_text("# Changelog de Predicciones\n\n"
                        "Historial de corridas del pipeline 01-04.\n", encoding="utf-8")
    bloque = [f"\n## {fecha} — {q_objetivo} ({n_campos} campos)\n"]
    if comp is None or comp.empty:
        bloque.append("\nSin snapshot previo para comparar.\n")
    else:
        top = comp.head(5)
        bloque.append("\nTop 5 movimientos vs corrida anterior (|DIF_ABS_MBPE|):\n\n")
        bloque.append("| CAMPO | MOTOR | VOL_ANTERIOR | VOL_NUEVO | DIF_ABS | DIF_% |\n")
        bloque.append("|---|---|---:|---:|---:|---:|\n")
        for _, r in top.iterrows():
            bloque.append(f"| {r['CAMPO']} | {r['MOTOR']} | {r['VOL_ANTERIOR_MBPE']:.1f} | "
                          f"{r['VOL_NUEVO_MBPE']:.1f} | {r['DIF_ABS_MBPE']:.1f} | "
                          f"{r['DIF_PCT']:.1f}% |\n")
    with open(ruta, "a", encoding="utf-8") as f:
        f.writelines(bloque)
    print(f"  Changelog actualizado: {ruta}")


if __name__ == "__main__":
    print("=== 04_pbi_export.py — Meshgrid Brent->Neto->Delta (Modelo 2 + Modelo 1) ===\n")

    ruta = STAGING / "tablon_unico.parquet"
    if not ruta.exists():
        raise FileNotFoundError("Ejecutar scripts 01-03b primero.")
    df = pd.read_parquet(ruta)

    correlacion = cargar_correlacion()
    anclas      = cargar_anclas(df)
    ponderados  = cargar_ponderados(df)
    baselines   = cargar_baselines(df)
    bandas      = banda_historica_brent(df)

    # Re-anclaje (2026-06-12): p_ref/delta_ref por campo x motor desde metricas.csv (03)
    ruta_met = STAGING / "metricas.csv"
    if not ruta_met.exists():
        raise FileNotFoundError("metricas.csv no existe: correr 03_modelo.py primero.")
    _met_full = pd.read_csv(ruta_met)
    anclaje = _met_full.set_index("CAMPO")[
        ["BRENT_REF_USD_BBL", "P_REF_USD_BBL", "DELTA_REF_ISO", "DELTA_REF_SUAVE"]
    ].to_dict("index")
    BRENT_REF = float(_met_full["BRENT_REF_USD_BBL"].dropna().iloc[0]) \
        if _met_full["BRENT_REF_USD_BBL"].notna().any() else np.nan

    vigencia_base, q_objetivo = derivar_vigencias(df)
    fecha_prediccion = str(date.today())
    print(f"\n{'='*55}")
    print(f"  PREDICCION para {q_objetivo}  (datos base: {vigencia_base}, "
          f"generada: {fecha_prediccion})")
    print(f"{'='*55}\n")

    # Meshgrid dinamico: banda Consolidado real ± MARGEN_BRENT_USD
    df_cons_real = df[(~df["ES_SINTETICO"]) & (~df["ES_BASELINE"]) & df["BRENT_FLAT_USD_BBL"].notna()]
    if not df_cons_real.empty:
        brent_obs_min = float(df_cons_real["BRENT_FLAT_USD_BBL"].min())
        brent_obs_max = float(df_cons_real["BRENT_FLAT_USD_BBL"].max())
    else:
        brent_obs_min, brent_obs_max = 58.0, 92.0
    brent_min_din = int(np.floor(brent_obs_min)) - MARGEN_BRENT_USD
    brent_max_din = int(np.ceil(brent_obs_max)) + MARGEN_BRENT_USD
    brent_range = np.arange(brent_min_din, brent_max_din + BRENT_PASO, BRENT_PASO,
                            dtype=float)
    # Incluir el punto EXACTO de re-anclaje: la fila Brent=BRENT_REF debe existir
    # en la matriz para que Vol=baseline sea verificable sin interpolar
    if pd.notna(BRENT_REF):
        brent_range = np.unique(np.append(brent_range, round(BRENT_REF, 2)))
    campos = sorted(df["CAMPO"].unique())

    print(f"Banda Consolidado observada: [${brent_obs_min:.1f}, ${brent_obs_max:.1f}]")
    print(f"Grilla Brent (+-{MARGEN_BRENT_USD}): [${brent_min_din}, ${brent_max_din}] "
          f"paso ${BRENT_PASO} = {len(brent_range)} pts")
    print(f"Campos: {len(campos)} | Motores: Isotonica (primario), Suave (validacion)")
    print(f"Re-anclaje: BRENT_REF={BRENT_REF:.2f} -> Vol(BRENT_REF) = baseline exacto\n")

    pneto_grid = np.arange(PNETO_GRID_MIN, PNETO_GRID_MAX + PNETO_GRID_PASO,
                           PNETO_GRID_PASO)
    filas    = []   # cadena completa Brent -> Aceite -> Volumen
    filas_m1 = []   # Modelo 1 puro (grilla en Precio Aceite)
    filas_m2 = []   # Modelo 2 puro (grilla en Brent)
    for campo in campos:
        modelos = {}
        for label, suf in MOTORES:
            rmodel = MODELOS_DIR / f"{campo.replace(' ', '_')}_{suf}.joblib"
            if rmodel.exists():
                modelos[label] = joblib.load(rmodel)
        if not modelos:
            print(f"  [WARN] {campo}: sin modelos, omitiendo")
            continue
        coef = correlacion.get(campo)
        if coef is None:
            print(f"  [WARN] {campo}: sin correlacion Brent->Aceite, omitiendo")
            continue

        bk_fin, bk_pdp = anclas.get(campo, (np.nan, np.nan))
        bk_ref_fin, bk_ref_ope = ponderados.get(campo, (np.nan, np.nan))
        baseline = baselines.get(campo, np.nan)
        bk_min_hist, bk_max_hist = bandas.get(campo, (40.0, 80.0))
        anc = anclaje.get(campo, {})
        p_ref = anc.get("P_REF_USD_BBL", np.nan)
        delta_refs = {"Isotonica": anc.get("DELTA_REF_ISO", 0.0) or 0.0,
                      "Suave":     anc.get("DELTA_REF_SUAVE", 0.0) or 0.0}
        m2_metodo = coef.get("METODO", "")
        m2_fallback = bool(coef.get("ES_FALLBACK", False))
        _bk_dura = bk_pdp if pd.notna(bk_pdp) else None

        # ── Matriz M2 pura: Brent -> Precio Aceite ────────────────────────────
        aceite_grid = m2.neto_desde_brent(coef, brent_range)
        for i, brent in enumerate(brent_range):
            filas_m2.append({
                "CAMPO":                  campo,
                "BRENT_USD_BBL":          float(brent),
                "PRECIO_ACEITE_USD_BBL":  round(float(aceite_grid[i]), 2),
                "ALPHA":                  coef.get("ALPHA"),
                "BETA":                   coef.get("BETA"),
                "M2_METODO":              m2_metodo,
                "M2_ES_FALLBACK":         m2_fallback,
                "N_PUNTOS":               coef.get("N_PUNTOS"),
                "R2":                     coef.get("R2"),
                "R2_LOO":                 coef.get("R2_LOO"),
                "MAE_LOO":                coef.get("MAE_LOO"),
                "ALERTA":                 coef.get("ALERTA", ""),
            })

        # ── Matriz M1 pura: Precio Aceite -> Delta anclado / Volumen ─────────
        for label, modelo in modelos.items():
            d_ref = float(delta_refs[label])
            delta_anc = modelo.predict(pneto_grid) - d_ref
            vol_anc = volumen_anclado(modelo, pneto_grid, baseline, d_ref, _bk_dura) \
                if pd.notna(baseline) else np.full(len(pneto_grid), np.nan)
            for i, pn in enumerate(pneto_grid):
                filas_m1.append({
                    "CAMPO":                       campo,
                    "MOTOR":                       label,
                    "PRECIO_ACEITE_USD_BBL":       float(pn),
                    "DELTA_ANCLADO_MBPE":          round(float(delta_anc[i]), 2),
                    "VOLUMEN_1P_PREDICHO_MBPE":    round(float(vol_anc[i]), 2)
                                                   if pd.notna(vol_anc[i]) else None,
                    "VOLUMEN_1P_BASELINE_MBPE":    round(float(baseline), 2)
                                                   if pd.notna(baseline) else None,
                    "P_REF_USD_BBL":               round(float(p_ref), 2)
                                                   if pd.notna(p_ref) else None,
                    "BREAKEVEN_FINANCIERO_USD_BBL":  round(bk_fin, 2) if pd.notna(bk_fin) else None,
                    "BREAKEVEN_OPERACIONAL_USD_BBL": round(bk_pdp, 2) if pd.notna(bk_pdp) else None,
                })

        # ── Cadena completa: Brent -> Aceite -> Volumen anclado ──────────────
        for label, modelo in modelos.items():
            d_ref = float(delta_refs[label])
            delta_anc = modelo.predict(aceite_grid) - d_ref
            vol = volumen_anclado(modelo, aceite_grid, baseline, d_ref, _bk_dura) \
                if pd.notna(baseline) else modelo.predict(aceite_grid) - d_ref

            for i, brent in enumerate(brent_range):
                pn   = float(aceite_grid[i])
                pn_r = round(pn, 2)
                es_viable = (pn_r >= round(bk_pdp, 2)) if pd.notna(bk_pdp) else True
                es_full   = (pn_r >= round(bk_fin, 2)) if pd.notna(bk_fin) else True
                es_extrap = (float(brent) < bk_min_hist - MARGEN_EXTRAP_USD or
                             float(brent) > bk_max_hist + MARGEN_EXTRAP_USD)

                vol_pred = float(vol[i])
                vol_base = float(baseline) if pd.notna(baseline) else np.nan
                delta_vs = round(vol_pred - vol_base, 2) if pd.notna(baseline) else np.nan
                pct_vs   = round((vol_pred - vol_base) / vol_base * 100, 2) \
                    if pd.notna(baseline) and vol_base > 0 else np.nan

                filas.append({
                    "CAMPO":                          campo,
                    "MOTOR":                          label,
                    "BRENT_USD_BBL":                  float(brent),
                    "PRECIO_NETO_EFECTIVO_USD_BBL":   round(pn, 2),
                    "DELTA_PRED_MBPE":                round(float(delta_anc[i]), 2),
                    "VOLUMEN_1P_BASELINE_MBPE":        round(vol_base, 2) if pd.notna(vol_base) else None,
                    "VOLUMEN_1P_PREDICHO_MBPE":        round(vol_pred, 2),
                    "DELTA_VS_BASE_MBPE":              delta_vs,
                    "DELTA_VS_BASE_PCT":               pct_vs,
                    # Re-anclaje: punto actual donde Vol=baseline por construccion
                    "BRENT_REF_USD_BBL":               round(BRENT_REF, 2) if pd.notna(BRENT_REF) else None,
                    "P_REF_USD_BBL":                   round(float(p_ref), 2) if pd.notna(p_ref) else None,
                    # Tag M2: campos sin relacion propia Aceite~Brent (k de portafolio)
                    "M2_METODO":                       m2_metodo,
                    "M2_ES_FALLBACK":                  m2_fallback,
                    "BREAKEVEN_FINANCIERO_USD_BBL":    round(bk_fin, 2) if pd.notna(bk_fin) else None,
                    "BREAKEVEN_OPERACIONAL_USD_BBL":   round(bk_pdp, 2) if pd.notna(bk_pdp) else None,
                    "BK_REFERENCIA_FIN_USD_BBL":       round(bk_ref_fin, 2) if pd.notna(bk_ref_fin) else None,
                    "BK_REFERENCIA_OPE_USD_BBL":       round(bk_ref_ope, 2) if pd.notna(bk_ref_ope) else None,
                    "ES_VIABLE":                       es_viable,
                    "ES_FULL_RESERVAS":                es_full,
                    "ES_EXTRAPOLADO":                  es_extrap,
                    "TIPO_DATO":                       "PREDICCIÓN",
                    "Q_OBJETIVO":                      q_objetivo,
                    "VIGENCIA_BASE":                   vigencia_base,
                    "FECHA_PREDICCION":                fecha_prediccion,
                })

        # Resumen impreso (motor primario, prediccion anclada)
        if "Isotonica" in modelos and pd.notna(baseline):
            d_ref = float(delta_refs["Isotonica"])
            nb = m2.neto_desde_brent(coef, np.array([brent_min_din, brent_max_din]))
            v = volumen_anclado(modelos["Isotonica"], nb, baseline, d_ref, _bk_dura)
            print(f"  {campo:<20} | baseline={baseline:.1f} MBPE | "
                  f"Iso@${brent_min_din}={v[0]:.0f} -> @${brent_max_din}={v[1]:.0f}")

    df_out = pd.DataFrame(filas)

    # ── Clasificacion de confianza por campo (primario = Isotonica) ───────────
    if True:
        _met_raw = _met_full.copy()
        for _c in ["ALERTA_LOO_OUTLIER_ISO"]:
            if _c not in _met_raw.columns:
                _met_raw[_c] = False
        for _c in ["SKILL_ISO", "MAE_NAIVE"]:
            if _c not in _met_raw.columns:
                _met_raw[_c] = np.nan
        met = _met_raw[["CAMPO", "N_REAL_DELTA", "MAE_LOO_ISO", "BASELINE_LATEST",
                        "ALERTA_LOO_OUTLIER_ISO", "SKILL_ISO", "MAE_NAIVE"]].copy()
        met["MAE_REL_LOO"] = (met["MAE_LOO_ISO"] /
                              met["BASELINE_LATEST"].replace(0, np.nan)).fillna(999.0)
        met["N_REAL_DELTA"] = met["N_REAL_DELTA"].fillna(0).astype(int)
        met["ALERTA_LOO_OUTLIER_ISO"] = met["ALERTA_LOO_OUTLIER_ISO"].fillna(False).astype(bool)

        div_serie = calcular_divergencia_motores(df_out).rename("DIVERGENCIA_MOTORES_PCT")
        met = met.merge(div_serie, on="CAMPO", how="left")
        met["DIVERGENCIA_MOTORES_PCT"] = met["DIVERGENCIA_MOTORES_PCT"].fillna(999.0)

        rows_conf = []
        for _, r in met.iterrows():
            nivel, motivo = clasificar_confianza(
                n_real=int(r["N_REAL_DELTA"]),
                mae_rel=float(r["MAE_REL_LOO"]),
                divergencia=float(r["DIVERGENCIA_MOTORES_PCT"]),
                baseline=float(r["BASELINE_LATEST"]) if pd.notna(r["BASELINE_LATEST"]) else 0.0,
                mae_abs=float(r["MAE_LOO_ISO"]) if pd.notna(r["MAE_LOO_ISO"]) else 999.0,
                outlier_lloo=bool(r["ALERTA_LOO_OUTLIER_ISO"]),
                skill=float(r["SKILL_ISO"]) if pd.notna(r["SKILL_ISO"]) else np.nan,
                mae_naive=float(r["MAE_NAIVE"]) if pd.notna(r["MAE_NAIVE"]) else np.nan)
            rows_conf.append({"CAMPO": r["CAMPO"], "N_REAL_DELTA": int(r["N_REAL_DELTA"]),
                              "MAE_REL_LOO": round(float(r["MAE_REL_LOO"]), 4),
                              "DIVERGENCIA_MOTORES_PCT": round(float(r["DIVERGENCIA_MOTORES_PCT"]), 4),
                              "ALERTA_LOO_OUTLIER_ISO": bool(r["ALERTA_LOO_OUTLIER_ISO"]),
                              "NIVEL_CONFIANZA": nivel, "MOTIVO_CONFIANZA": motivo})

        df_conf = pd.DataFrame(rows_conf)
        df_out = df_out.merge(
            df_conf[["CAMPO", "N_REAL_DELTA", "MAE_REL_LOO", "DIVERGENCIA_MOTORES_PCT",
                     "ALERTA_LOO_OUTLIER_ISO", "NIVEL_CONFIANZA", "MOTIVO_CONFIANZA"]],
            on="CAMPO", how="left")

        conteo = df_conf.groupby("NIVEL_CONFIANZA")["CAMPO"].nunique()
        print(f"\n{'='*55}\n  Clasificacion de confianza por campo\n{'='*55}")
        for nivel in ["ALTA", "MEDIA", "BAJA", "SOLO_SINTETICO"]:
            print(f"  {nivel:<15}: {conteo.get(nivel, 0):3d} campos")

        materiales = met[met["BASELINE_LATEST"].fillna(0) >= 50].merge(
            df_conf[["CAMPO", "NIVEL_CONFIANZA"]], on="CAMPO")
        if not materiales.empty:
            print("\n  Campos materiales (baseline >= 50 MBPE):")
            for _, r in materiales.sort_values("BASELINE_LATEST", ascending=False).iterrows():
                print(f"    {r['CAMPO']:<25} baseline={r['BASELINE_LATEST']:.0f} MBPE  "
                      f"MAE_rel={r['MAE_REL_LOO']:.0%}  -> {r['NIVEL_CONFIANZA']}")
        print(f"{'='*55}\n")

    ruta_csv = RESULTADOS / "output_matriz_prediccion.csv"
    df_out.to_csv(ruta_csv, index=False, encoding="utf-8-sig")

    # Matrices aisladas por modelo (directriz 2026-06-12)
    ruta_m1 = RESULTADOS / "output_matriz_modelo1.csv"
    ruta_m2_csv = RESULTADOS / "output_matriz_modelo2.csv"
    pd.DataFrame(filas_m1).to_csv(ruta_m1, index=False, encoding="utf-8-sig")
    pd.DataFrame(filas_m2).to_csv(ruta_m2_csv, index=False, encoding="utf-8-sig")
    print(f"  Matriz Modelo 1 (Aceite->Volumen, puro): {ruta_m1}")
    print(f"  Matriz Modelo 2 (Brent->Aceite, puro):   {ruta_m2_csv}")

    HISTORICO_DIR.mkdir(parents=True, exist_ok=True)
    nombre_snapshot = f"prediccion_{q_objetivo}_generada_{fecha_prediccion}.csv"
    ruta_snapshot = HISTORICO_DIR / nombre_snapshot
    shutil.copy2(ruta_csv, ruta_snapshot)
    print(f"  Snapshot fechado: {ruta_snapshot}")

    print(f"\nMatriz exportada: {ruta_csv}")
    print(f"  Filas: {len(df_out)}")
    print(f"  Extrapolados: {df_out['ES_EXTRAPOLADO'].sum()} "
          f"({df_out['ES_EXTRAPOLADO'].mean():.0%} del total)")

    # Sanity final: extremos del grid
    brent_lo = int(brent_range[0]); brent_hi = int(brent_range[-1])
    check_lo = df_out[df_out["BRENT_USD_BBL"] == brent_lo].groupby(
        ["CAMPO", "MOTOR"])["VOLUMEN_1P_PREDICHO_MBPE"].mean()
    check_hi = df_out[df_out["BRENT_USD_BBL"] == brent_hi].groupby(
        ["CAMPO", "MOTOR"])["VOLUMEN_1P_PREDICHO_MBPE"].mean()
    print(f"\n  Sanity Brent=${brent_lo} (extremo bajo): {len(check_lo)} series")
    print(f"  Sanity Brent=${brent_hi} (extremo alto): {len(check_hi)} series")

    # Sanity re-anclaje: en Brent=BRENT_REF el volumen debe ser el baseline exacto
    # (salvo p_ref < BK abandono: alli el piso duro manda y Vol=0)
    if pd.notna(BRENT_REF):
        ref = df_out.copy()
        ref["_dist"] = (ref["BRENT_USD_BBL"] - BRENT_REF).abs()
        idx = ref.groupby(["CAMPO", "MOTOR"])["_dist"].idxmin()
        ref = ref.loc[idx]
        ref = ref[ref["VOLUMEN_1P_BASELINE_MBPE"].notna()]
        # tolerancia = |curva(p@brent_cercano) − curva(p_ref)| ≈ paso de grilla
        dif = (ref["VOLUMEN_1P_PREDICHO_MBPE"] - ref["VOLUMEN_1P_BASELINE_MBPE"]).abs()
        piso = ref["PRECIO_NETO_EFECTIVO_USD_BBL"] < ref["BREAKEVEN_OPERACIONAL_USD_BBL"].fillna(-np.inf)
        ok = ((dif <= 1.0) | piso).sum()
        print(f"\n  Sanity re-anclaje @Brent~{BRENT_REF:.1f}: {ok}/{len(ref)} series "
              f"con Vol=baseline (tol grilla $1) o en piso de abandono")
        if ok < len(ref):
            peores = ref.loc[~((dif <= 1.0) | piso)].assign(_dif=dif)
            print(peores.nlargest(5, "_dif")[["CAMPO", "MOTOR", "VOLUMEN_1P_PREDICHO_MBPE",
                                              "VOLUMEN_1P_BASELINE_MBPE"]].to_string(index=False))

    # ── Ledger de cobertura (transparencia para decisiones CAPEX) ────────────────
    # Cada campo del baseline aparece con su estado y motivo de ausencia.
    # Motivos: sin_consolidado (sin tabla sensibilidad forward, mayoria activos US/gas),
    #          filial_migracion (padre perdio cert 2024/25 a entidad " FILIAL"),
    #          consolidado_sin_sens (tiene CONSOLIDADO pero sin vol sensibilidad),
    #          presente (incluido en la matriz de prediccion).
    presentes_en_matriz = set(df_out["CAMPO"].unique())
    # nombre de la columna AÑO (puede tener encoding raro en el parquet)
    _col_anio = next((c for c in df.columns if c in ("AÑO",) or
                      (c.startswith("A") and c.endswith("O") and len(c) <= 4)), None)

    rows_cob = []
    for campo, vol in baselines.items():
        if campo in presentes_en_matriz:
            motivo = "presente"
        else:
            sub = df[df["CAMPO"] == campo]
            cons = sub[sub["ESCENARIO"].str.startswith("CONSOLIDADO", na=False)]
            n_cons = len(cons)
            n_sens = int(cons["VOLUMEN_1P_SENSIBILIDAD_MBPE"].notna().sum())
            # Tiene certificado reciente 2024/2025 en el padre?
            b2425 = (sub[(sub["ESCENARIO"] == "BASE") &
                         sub[_col_anio].isin([2024, 2025])]
                     if _col_anio else sub.iloc[0:0])
            tiene_base_reciente = bool(b2425["VOLUMEN_1P_OFICIAL_MBPE"].notna().any())
            if n_cons == 0:
                motivo = "sin_consolidado"
            elif n_sens > 0 and not tiene_base_reciente:
                motivo = "filial_migracion"
            else:
                motivo = "consolidado_sin_sens"
        rows_cob.append({"CAMPO": campo,
                         "BASELINE_1P_MBPE": round(float(vol), 2),
                         "EN_PREDICCION": campo in presentes_en_matriz,
                         "MOTIVO_AUSENCIA": motivo})

    df_cob = pd.DataFrame(rows_cob).sort_values(
        ["EN_PREDICCION", "BASELINE_1P_MBPE"], ascending=[True, False])
    ruta_cob = RESULTADOS / "cobertura_portafolio.csv"
    df_cob.to_csv(ruta_cob, index=False, encoding="utf-8-sig")

    total_mbpe = df_cob["BASELINE_1P_MBPE"].sum()
    pres_mbpe  = df_cob[df_cob["EN_PREDICCION"]]["BASELINE_1P_MBPE"].sum()
    print(f"\n{'='*55}")
    print(f"  COBERTURA DEL PORTAFOLIO  ({ruta_cob.name})")
    print(f"{'='*55}")
    print(f"  Total baseline:  {len(df_cob):3d} campos | {total_mbpe:.1f} MBPE")
    print(f"  En prediccion:   {df_cob['EN_PREDICCION'].sum():3d} campos | "
          f"{pres_mbpe:.1f} MBPE ({100*pres_mbpe/total_mbpe:.1f}%)")
    for mot, desc in [("sin_consolidado",    "sin datos fuente    "),
                      ("filial_migracion",   "migr. FILIAL recup. "),
                      ("consolidado_sin_sens", "Consol. sin sensib. ")]:
        sub_m = df_cob[df_cob["MOTIVO_AUSENCIA"] == mot]
        if not sub_m.empty:
            print(f"  Ausente {desc}: {len(sub_m):3d} campos | "
                  f"{sub_m['BASELINE_1P_MBPE'].sum():.1f} MBPE")
    print(f"{'='*55}\n")

    brent_ref = round((brent_obs_min + brent_obs_max) / 2)
    comp = generar_comparacion_vs_anterior(df_out, q_objetivo, brent_ref, ruta_snapshot)
    actualizar_changelog(q_objetivo, fecha_prediccion, len(campos), comp)

    print("\n=== 04_pbi_export.py — Completado ===")
