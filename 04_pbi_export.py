"""
04_pbi_export.py — Meshgrid de predicciones para Power BI

Pre-calcula todas las combinaciones (Brent x Campo x Motor) como CSV estatico.
PBI consume el CSV con DAX + What-If slider de Brent.

Re-arquitectura 2026-06-04:
  - Prediccion = BASELINE_LATEST + DELTA_PRED (reconstruccion desde espacio delta).
  - BASELINE_LATEST = ultimo VOLUMEN_1P_OFICIAL certificado por campo.

Correcciones:
  B3: ES_VIABLE ahora compara PRECIO_NETO_EFECTIVO >= BREAKEVEN_FINANCIERO
      (mismo marco de precio). Antes comparaba BRENT >= BREAKEVEN_NETO → ~10 USD optimista.
  ES_EXTRAPOLADO: True si el Brent esta fuera de la banda del Consolidado por campo ± margen.

Meshgrid dinamico (2026-06-09):
  La grilla ya NO es fija $20-$120. Se deriva del Brent observado en los datos CONSOLIDADO
  reales + MARGEN_BRENT_USD. Los cambios inter-Q son pequeños; ceñirse a la banda observada
  permite INTERPOLACION en lugar de extrapolacion lejos del rango con datos. Se recalcula
  automaticamente cada Q cuando entran datos nuevos.
  Ejemplo hoy: Consolidado $68-$82 → grilla $58-$92 (~35 pts paso $1).

Re-arquitectura 2026-06-09 (escalera financiero/operacional + 3D descuentos):
  - Convencion finanzas: BREAKEVEN_FINANCIERO_USD_BBL = piso SUPERIOR (delta=0, mata
    PNP+PND), BREAKEVEN_OPERACIONAL_USD_BBL = piso INFERIOR (abandono, mata todo).
  - ES_VIABLE = precio_neto >= BREAKEVEN_OPERACIONAL (el campo conserva alguna reserva).
  - ES_FULL_RESERVAS = precio_neto >= BREAKEVEN_FINANCIERO (escalera completa, sin
    castigo de PNP+PND).
  - Eje nuevo ESCENARIO_DESCUENTO ∈ {BAJO, BASE, ALTO}: P10/mediana/P90 historico por
    campo de DESCUENTO_CALIDAD/TRANSPORTE. El meshgrid pasa a Brent x Campo x Motor x
    3 escenarios.
  - Se retira la correccion de bias post-hoc (Opp-1, VOLUMEN_1P_CORREGIDO_MBPE /
    BIAS_CORRECCION_MBPE): el sesgo se ataca en la raiz vía peso dinamico de
    sinteticos en 03_modelo.py.
"""

import shutil
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR    = Path(__file__).parent
STAGING     = BASE_DIR / "datos" / "staging"
MODELOS_DIR = STAGING / "modelos"
RESULTADOS  = BASE_DIR / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

BRENT_PASO = 1
# Margen sobre la banda del Consolidado para el meshgrid (cubre escenarios up/down)
MARGEN_BRENT_USD  = 10
# Margen para marcar ES_EXTRAPOLADO por campo (diferente al margen del grid)
MARGEN_EXTRAP_USD = 5.0

# ── Umbrales de confianza (ajustables) ───────────────────────────────────────
# ALTA:  dato real suficiente + baseline material + error bajo + motores concordantes
# MEDIA: dato real presente + baseline material + error/divergencia moderados
# BAJA:  dato real pero error alto, motores divergen, o campo inmaterial (baseline<0.5)
# SOLO_SINTETICO: sin ningun punto real de sensibilidad
CONFIANZA_N_REAL_MIN    = 6     # N_REAL_DELTA mínimo para ALTA/MEDIA
CONFIANZA_BASELINE_MIN  = 0.5   # MBPE — por debajo es micro-campo (BAJA)
CONFIANZA_MAE_REL_ALTA  = 0.20  # MAE/baseline < 20% para ALTA
CONFIANZA_MAE_REL_MEDIA = 0.40  # MAE/baseline < 40% para MEDIA
CONFIANZA_DIV_ALTA      = 0.30  # divergencia XGB↔ISO < 30% para ALTA
CONFIANZA_DIV_MEDIA     = 0.50  # divergencia XGB↔ISO < 50% para MEDIA
# Opp-5: umbrales absolutos para micro-campos (evitar penalizar por ratio MAE/baseline alto)
CONFIANZA_MAE_ABS_ALTA  = 2.0   # MBPE: error < 2 MBPE → no penalizar ratio alto en ALTA
CONFIANZA_MAE_ABS_MEDIA = 5.0   # MBPE: error < 5 MBPE → no penalizar ratio alto en MEDIA
# Opp-3: ratio RMSE/MAE > umbral indica fold LOO dominado por outlier → cap confianza MEDIA
CONFIANZA_OUTLIER_RATIO = 2.0

HISTORICO_DIR = BASE_DIR / "resultados" / "historico_predicciones"


def siguiente_quarter(q: str) -> str:
    """Avanza un quarter: '2026_Q1' -> '2026_Q2', '2025_Q4' -> '2026_Q1'."""
    anio, qn = q.split("_Q")
    anio, qn = int(anio), int(qn)
    if qn == 4:
        return f"{anio + 1}_Q1"
    return f"{anio}_Q{qn + 1}"


def derivar_vigencias(df: pd.DataFrame) -> tuple[str, str]:
    """
    Extrae el maximo quarter de los escenarios CONSOLIDADO_* del tablon.
    Retorna (vigencia_base, q_objetivo).
    """
    qs = [
        s.replace("CONSOLIDADO_", "")
        for s in df["ESCENARIO"].dropna().unique()
        if str(s).startswith("CONSOLIDADO_")
    ]
    if not qs:
        return "DESCONOCIDA", "DESCONOCIDA"
    vigencia_base = sorted(qs)[-1]
    return vigencia_base, siguiente_quarter(vigencia_base)


def cargar_medianas(df: pd.DataFrame) -> dict:
    """Medianas historicas de Cal y Tra por campo (filas reales con precio)."""
    df_real = df[
        (~df["ES_SINTETICO"]) &
        df["BRENT_FLAT_USD_BBL"].notna() &
        df["DESCUENTO_CALIDAD_USD_BBL"].notna() &
        df["DESCUENTO_TRANSPORTE_USD_BBL"].notna()
    ]
    return (df_real
            .groupby("CAMPO")[["DESCUENTO_CALIDAD_USD_BBL",
                                "DESCUENTO_TRANSPORTE_USD_BBL"]]
            .median()
            .to_dict(orient="index"))


def cargar_anclas(df: pd.DataFrame) -> dict:
    """
    Anclas de la escalera financiero/operacional por campo (constantes en el tablon).
    Retorna {campo: (bk_fin_superior, bk_pdp_inferior)}.
    """
    # FIN y PDP deben venir de la MISMA vigencia (calcular_breakeven_ponderado los
    # calcula en pareja por CAMPO×VIGENCIA): mezclar vigencias puede invertir los
    # pisos. Se usa la vigencia mas reciente (igual que 02_synthetic.py).
    sub = df.dropna(subset=["BK_ANCLA_FIN_USD_BBL", "BK_ANCLA_PDP_USD_BBL"], how="all")
    out = {}
    for campo, g in sub.groupby("CAMPO"):
        fila_ancla = g.sort_values("VIGENCIA_BREAKEVEN", ascending=False).iloc[0]
        bk_fin = float(fila_ancla["BK_ANCLA_FIN_USD_BBL"]) if pd.notna(fila_ancla["BK_ANCLA_FIN_USD_BBL"]) else np.nan
        bk_pdp = float(fila_ancla["BK_ANCLA_PDP_USD_BBL"]) if pd.notna(fila_ancla["BK_ANCLA_PDP_USD_BBL"]) else np.nan
        # Si solo una ancla esta disponible, usar la misma para ambas (tramo unico,
        # igual que 02_synthetic.py): evita ES_FULL_RESERVAS=True con ES_VIABLE=False.
        if np.isnan(bk_fin):
            bk_fin = bk_pdp
        if np.isnan(bk_pdp):
            bk_pdp = bk_fin
        out[campo] = (bk_fin, bk_pdp)
    return out


def cargar_escenarios_descuento(df: pd.DataFrame) -> dict:
    """
    Por campo, deriva 3 escenarios de descuento desde filas reales historicas:
    BAJO=P10, BASE=mediana, ALTO=P90 de DESCUENTO_CALIDAD/TRANSPORTE.
    Retorna {campo: {"BAJO": (cal,tra), "BASE": (cal,tra), "ALTO": (cal,tra)}}.
    """
    df_real = df[
        (~df["ES_SINTETICO"]) &
        df["DESCUENTO_CALIDAD_USD_BBL"].notna() &
        df["DESCUENTO_TRANSPORTE_USD_BBL"].notna()
    ]
    out = {}
    for campo, g in df_real.groupby("CAMPO"):
        cal = g["DESCUENTO_CALIDAD_USD_BBL"]
        tra = g["DESCUENTO_TRANSPORTE_USD_BBL"]
        out[campo] = {
            "BAJO": (float(cal.quantile(0.10)), float(tra.quantile(0.10))),
            "BASE": (float(cal.median()),       float(tra.median())),
            "ALTO": (float(cal.quantile(0.90)), float(tra.quantile(0.90))),
        }
    return out


def cargar_baselines(df: pd.DataFrame) -> dict:
    """Ultimo VOLUMEN_1P_OFICIAL_MBPE certificado por campo."""
    df_b = df[(df["ESCENARIO"] == "BASE") & df["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
    return (df_b.sort_values("AÑO")
            .groupby("CAMPO")["VOLUMEN_1P_OFICIAL_MBPE"]
            .last().to_dict())


def clasificar_confianza(
    n_real: int,
    mae_rel: float,      # MAE_LOO_XGB / BASELINE_LATEST
    divergencia: float,  # media |XGB-ISO|/ISO en banda $40-$80
    baseline: float,
    mae_abs: float = 999.0,      # MAE_LOO_XGB en MBPE absolutas (opp-5: umbral dinámico)
    outlier_lloo: bool = False,  # opp-3: fold LOO dominado por outlier → cap MEDIA
) -> tuple[str, str]:
    """
    Clasifica la confianza de la prediccion de un campo.

    La clasificacion usa los criterios funcionales del piloto (MAESTRO §7.4) en lugar
    del R²_LOO, que es estadisticamente inestable con N~8 puntos por campo.

    Opp-5: para micro-campos con MAE absoluto pequeño (<2 MBPE para ALTA, <5 para MEDIA),
    el ratio mae_rel puede ser alto pero el error real es irrelevante para CAPEX. Se aplica
    un umbral efectivo que no penaliza por tamaño de campo.

    Opp-3: si un fold LOO concentra el error (RMSE/MAE > 2), la confianza se limita a MEDIA
    máximo para advertir al analista sobre inestabilidad de validación.

    Retorna (nivel, motivo_auditable).
    """
    # Umbral efectivo de MAE_rel: micro-campos con error absoluto pequeño no se penalizan
    if mae_abs < CONFIANZA_MAE_ABS_ALTA:
        mae_rel_eff = min(mae_rel, CONFIANZA_MAE_REL_ALTA * 0.99)
    elif mae_abs < CONFIANZA_MAE_ABS_MEDIA:
        mae_rel_eff = min(mae_rel, CONFIANZA_MAE_REL_MEDIA * 0.99)
    else:
        mae_rel_eff = mae_rel

    partes = [f"N={n_real}", f"MAE_rel={mae_rel:.2f}", f"MAE_abs={mae_abs:.2f}MBPE",
              f"div={divergencia:.2f}", f"base={baseline:.1f}MBPE"]
    if outlier_lloo:
        partes.append("OUTLIER_LOO")
    motivo = "; ".join(partes)

    if n_real == 0:
        return "SOLO_SINTETICO", f"sin datos reales; {motivo}"

    if baseline < CONFIANZA_BASELINE_MIN:
        return "BAJA", f"micro-campo; {motivo}"

    # Opp-3: campo con outlier severo en LOO → cap MEDIA (no puede ser ALTA)
    if outlier_lloo:
        if (mae_rel_eff < CONFIANZA_MAE_REL_MEDIA and divergencia < CONFIANZA_DIV_MEDIA):
            return "MEDIA", motivo
        return "BAJA", motivo

    if (n_real >= CONFIANZA_N_REAL_MIN
            and mae_rel_eff < CONFIANZA_MAE_REL_ALTA
            and divergencia < CONFIANZA_DIV_ALTA):
        return "ALTA", motivo

    if (mae_rel_eff < CONFIANZA_MAE_REL_MEDIA
            and divergencia < CONFIANZA_DIV_MEDIA):
        return "MEDIA", motivo

    return "BAJA", motivo


def calcular_divergencia_motores(df_out: pd.DataFrame) -> pd.Series:
    """
    Divergencia media |XGB - ISO| / ISO por campo en banda Brent $40-$80,
    escenario de descuento BASE (evita triplicar la comparacion por escenario).
    Retorna Serie indexada por CAMPO.
    """
    banda = df_out[
        (df_out["BRENT_USD_BBL"] >= 40) & (df_out["BRENT_USD_BBL"] <= 80) &
        (df_out["ESCENARIO_DESCUENTO"] == "BASE")
    ]
    piv = (banda.pivot_table(
                index=["CAMPO", "BRENT_USD_BBL"],
                columns="MOTOR",
                values="VOLUMEN_1P_PREDICHO_MBPE")
           .reset_index()
           .dropna())

    if piv.empty or "XGBoost" not in piv.columns or "Isotonica" not in piv.columns:
        return pd.Series(dtype=float)

    piv["div"] = (
        (piv["XGBoost"] - piv["Isotonica"]).abs()
        / piv["Isotonica"].replace(0, np.nan)
    )
    return piv.groupby("CAMPO")["div"].mean()


def banda_historica_brent(df: pd.DataFrame) -> dict:
    """
    Rango de Brent observado en datos CONSOLIDADO reales por campo (no sinteticos, no BASE).
    Usado para marcar ES_EXTRAPOLADO fuera de la banda donde el modelo tiene sensibilidades.
    """
    df_real = df[
        (~df["ES_SINTETICO"]) &
        (~df["ES_BASELINE"]) &
        df["BRENT_FLAT_USD_BBL"].notna()
    ]
    out = {}
    for campo, sub in df_real.groupby("CAMPO"):
        out[campo] = (float(sub["BRENT_FLAT_USD_BBL"].min()),
                      float(sub["BRENT_FLAT_USD_BBL"].max()))
    return out


def _extraer_en_brent_ref(df: pd.DataFrame, brent_ref: float) -> pd.DataFrame:
    """
    Para cada CAMPO x MOTOR, extrae la fila con BRENT_USD_BBL mas cercano a brent_ref,
    en escenario de descuento BASE (si la columna existe; snapshots viejos no la tienen
    y se usan tal cual, sin tripletas de escenario).
    """
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


def generar_comparacion_vs_anterior(df_out: pd.DataFrame, q_objetivo: str,
                                     brent_ref: float, ruta_actual: Path):
    """
    Compara la prediccion actual (escenario BASE, Brent~brent_ref) contra el snapshot
    previo mas reciente en resultados/historico_predicciones/. Escribe
    resultados/comparacion_vs_anterior.csv. Si no hay snapshot previo, escribe solo
    encabezados y avisa. Retorna el DataFrame de comparacion (o None si no hay previo).
    """
    cols_out = ["CAMPO", "MOTOR", "Q_OBJETIVO_ANTERIOR", "Q_OBJETIVO_NUEVO",
                "VOL_ANTERIOR_MBPE", "VOL_NUEVO_MBPE", "DIF_ABS_MBPE", "DIF_PCT",
                "NIVEL_CONFIANZA_ANTERIOR", "NIVEL_CONFIANZA_NUEVO"]
    ruta_comp = RESULTADOS / "comparacion_vs_anterior.csv"

    previos = sorted(
        (p for p in HISTORICO_DIR.glob("prediccion_*.csv") if p.resolve() != ruta_actual.resolve()),
        key=lambda p: p.stat().st_mtime,
    )
    if not previos:
        pd.DataFrame(columns=cols_out).to_csv(ruta_comp, index=False, encoding="utf-8-sig")
        print(f"\n  [INFO] Sin snapshot previo: {ruta_comp} (solo encabezados)")
        return None

    ruta_prev = previos[-1]
    df_prev = pd.read_csv(ruta_prev)

    prev_ref  = _extraer_en_brent_ref(df_prev, brent_ref)
    nuevo_ref = _extraer_en_brent_ref(df_out, brent_ref)

    merged = nuevo_ref.merge(
        prev_ref, on=["CAMPO", "MOTOR"], how="outer", suffixes=("_NUEVO", "_ANTERIOR"),
    )
    merged = merged.rename(columns={
        "VOLUMEN_1P_PREDICHO_MBPE_NUEVO":    "VOL_NUEVO_MBPE",
        "VOLUMEN_1P_PREDICHO_MBPE_ANTERIOR": "VOL_ANTERIOR_MBPE",
    })
    merged["DIF_ABS_MBPE"] = merged["VOL_NUEVO_MBPE"] - merged["VOL_ANTERIOR_MBPE"]
    merged["DIF_PCT"] = (
        merged["DIF_ABS_MBPE"] / merged["VOL_ANTERIOR_MBPE"].replace(0, np.nan) * 100
    )
    merged = merged[cols_out].sort_values(
        "DIF_ABS_MBPE", key=lambda s: s.abs(), ascending=False, na_position="last")
    merged.to_csv(ruta_comp, index=False, encoding="utf-8-sig")
    print(f"\n  Comparacion vs corrida anterior ({ruta_prev.name}): {ruta_comp}")
    print(f"    @Brent~${brent_ref}: {len(merged)} filas comparadas")
    return merged


def actualizar_changelog(q_objetivo: str, fecha: str, n_campos: int, comp) -> None:
    """Agrega entrada fechada a docs/CHANGELOG_PREDICCIONES.md con resumen de la corrida."""
    ruta = BASE_DIR / "docs" / "CHANGELOG_PREDICCIONES.md"
    if not ruta.exists():
        ruta.write_text(
            "# Changelog de Predicciones\n\n"
            "Historial de corridas del pipeline 01-04 (loop rodante Brent vs Reservas 1P).\n\n"
            "## 2026-06-09 — Cambios estructurales\n\n"
            "- Swap de etiquetas de breakeven (financiero=piso superior/delta 0, "
            "operacional=piso inferior/abandono), validado con equipo financiero.\n"
            "- Escalera de sinteticos corregida: tramo BAJO (vol=0) + tramo ESCALERA "
            "(vol=PDP), RANGO_USD 20->5.\n"
            "- Peso dinamico de sinteticos (anti-sesgo): w_sint = n_real/n_sint, "
            "minimo 0.05.\n"
            "- Eje ESCENARIO_DESCUENTO (BAJO/BASE/ALTO) en el meshgrid de Power BI.\n"
            "- ES_VIABLE ahora usa el piso operacional; nuevo ES_FULL_RESERVAS usa "
            "el piso financiero.\n"
            "- Retiro de la correccion de bias post-hoc (Opp-1).\n"
            "- Versionamiento: comparacion_vs_anterior.csv + este changelog.\n",
            encoding="utf-8",
        )

    bloque = [f"\n## {fecha} — {q_objetivo} ({n_campos} campos)\n"]
    if comp is None or comp.empty:
        bloque.append("\nSin snapshot previo para comparar.\n")
    else:
        top = comp.head(5)
        bloque.append("\nTop 5 movimientos vs corrida anterior (|DIF_ABS_MBPE|):\n\n")
        bloque.append("| CAMPO | MOTOR | VOL_ANTERIOR | VOL_NUEVO | DIF_ABS | DIF_% |\n")
        bloque.append("|---|---|---:|---:|---:|---:|\n")
        for _, r in top.iterrows():
            bloque.append(
                f"| {r['CAMPO']} | {r['MOTOR']} | {r['VOL_ANTERIOR_MBPE']:.1f} | "
                f"{r['VOL_NUEVO_MBPE']:.1f} | {r['DIF_ABS_MBPE']:.1f} | "
                f"{r['DIF_PCT']:.1f}% |\n"
            )

    with open(ruta, "a", encoding="utf-8") as f:
        f.writelines(bloque)
    print(f"  Changelog actualizado: {ruta}")


if __name__ == "__main__":
    print("=== 04_pbi_export.py — Meshgrid para Power BI ===\n")

    ruta = STAGING / "tablon_unico.parquet"
    if not ruta.exists():
        raise FileNotFoundError("Ejecutar scripts 01-03 primero.")

    df = pd.read_parquet(ruta)

    medianas       = cargar_medianas(df)
    anclas         = cargar_anclas(df)
    escenarios_desc = cargar_escenarios_descuento(df)
    baselines      = cargar_baselines(df)
    bandas         = banda_historica_brent(df)

    # Etiquetas de la corrida — loop rodante
    vigencia_base, q_objetivo = derivar_vigencias(df)
    fecha_prediccion = str(date.today())
    print(f"\n{'='*55}")
    print(f"  PREDICCION para {q_objetivo}  "
          f"(datos base: {vigencia_base}, generada: {fecha_prediccion})")
    print(f"{'='*55}\n")

    # Meshgrid dinamico: banda Consolidado real ± MARGEN_BRENT_USD (interpolar, no extrapolar)
    df_cons_real = df[
        (~df["ES_SINTETICO"]) & (~df["ES_BASELINE"]) & df["BRENT_FLAT_USD_BBL"].notna()
    ]
    if not df_cons_real.empty:
        brent_obs_min = float(df_cons_real["BRENT_FLAT_USD_BBL"].min())
        brent_obs_max = float(df_cons_real["BRENT_FLAT_USD_BBL"].max())
    else:
        brent_obs_min, brent_obs_max = 58.0, 92.0  # fallback conservador
    brent_min_din = int(np.floor(brent_obs_min)) - MARGEN_BRENT_USD
    brent_max_din = int(np.ceil(brent_obs_max)) + MARGEN_BRENT_USD

    brent_range = np.arange(brent_min_din, brent_max_din + BRENT_PASO, BRENT_PASO)
    campos = sorted(df["CAMPO"].unique())

    print(f"Banda Consolidado observada: [${brent_obs_min:.1f}, ${brent_obs_max:.1f}]")
    print(f"Grilla Brent (+-{MARGEN_BRENT_USD}): [${brent_min_din}, ${brent_max_din}] "
          f"paso ${BRENT_PASO} = {len(brent_range)} pts  (era $20-$120=101 pts)")
    print(f"Campos: {len(campos)}")
    print(f"Escenarios de descuento: BAJO/BASE/ALTO (P10/mediana/P90 historico por campo)")
    print(f"Total filas: {len(brent_range) * len(campos) * 2 * 3}\n")

    ESCENARIOS = ["BAJO", "BASE", "ALTO"]

    filas = []
    for campo in campos:
        ruta_xgb = MODELOS_DIR / f"{campo.replace(' ', '_')}_xgb.joblib"
        ruta_iso = MODELOS_DIR / f"{campo.replace(' ', '_')}_iso.joblib"
        if not ruta_xgb.exists() or not ruta_iso.exists():
            print(f"  [WARN] {campo}: modelos no encontrados, omitiendo")
            continue

        xgb_m  = joblib.load(ruta_xgb)
        iso_m  = joblib.load(ruta_iso)

        med_cal  = medianas.get(campo, {}).get("DESCUENTO_CALIDAD_USD_BBL",    -7.0)
        med_tra  = medianas.get(campo, {}).get("DESCUENTO_TRANSPORTE_USD_BBL", -3.3)
        bk_fin, bk_pdp = anclas.get(campo, (np.nan, np.nan))
        baseline = baselines.get(campo, np.nan)

        # Banda historica observada en datos BASE
        bk_min_hist, bk_max_hist = bandas.get(campo, (40.0, 80.0))

        # Escenarios de descuento del campo (fallback: BASE = mediana historica/default)
        esc_campo = escenarios_desc.get(campo, {
            "BAJO": (med_cal, med_tra), "BASE": (med_cal, med_tra), "ALTO": (med_cal, med_tra),
        })

        for esc in ESCENARIOS:
            cal, tra = esc_campo[esc]

            # ── XGBoost ──────────────────────────────────────────────────────
            X_xgb = np.column_stack([
                brent_range,
                np.full(len(brent_range), cal),
                np.full(len(brent_range), tra),
            ])
            delta_xgb    = xgb_m.predict(X_xgb)
            precio_neto  = brent_range + cal + tra
            vol_xgb      = np.maximum(baseline + delta_xgb, 0) if pd.notna(baseline) else delta_xgb

            for i, brent in enumerate(brent_range):
                pn = float(precio_neto[i])

                # ES_VIABLE/ES_FULL_RESERVAS sobre valores REDONDEADOS (consistente con las
                # columnas exportadas PRECIO_NETO_EFECTIVO/BREAKEVEN_*, evita falsos
                # desacuerdos por redondeo en el limite exacto).
                pn_r = round(pn, 2)
                # ES_VIABLE = el campo conserva alguna reserva (>= piso operacional/inferior)
                es_viable = (pn_r >= round(bk_pdp, 2)) if pd.notna(bk_pdp) else True
                # ES_FULL_RESERVAS = escalera completa, sin castigo PNP+PND (>= piso financiero/superior)
                es_full = (pn_r >= round(bk_fin, 2)) if pd.notna(bk_fin) else True

                # ES_EXTRAPOLADO: brent fuera de la banda historica (± margen)
                es_extrap = (float(brent) < bk_min_hist - MARGEN_EXTRAP_USD or
                             float(brent) > bk_max_hist + MARGEN_EXTRAP_USD)

                vol_pred = float(vol_xgb[i])
                vol_base = float(baseline) if pd.notna(baseline) else np.nan
                delta_vs = round(vol_pred - vol_base, 2) if pd.notna(baseline) else np.nan
                pct_vs   = round((vol_pred - vol_base) / vol_base * 100, 2) \
                    if pd.notna(baseline) and vol_base > 0 else np.nan

                filas.append({
                    "CAMPO":                          campo,
                    "MOTOR":                          "XGBoost",
                    "ESCENARIO_DESCUENTO":            esc,
                    "BRENT_USD_BBL":                  float(brent),
                    "PRECIO_NETO_EFECTIVO_USD_BBL":   round(pn, 2),
                    "DESCUENTO_CALIDAD_USD_BBL":       round(cal, 2),
                    "DESCUENTO_TRANSPORTE_USD_BBL":    round(tra, 2),
                    "DELTA_PRED_MBPE":                round(float(delta_xgb[i]), 2),
                    "VOLUMEN_1P_BASELINE_MBPE":        round(vol_base, 2) if pd.notna(vol_base) else None,
                    "VOLUMEN_1P_PREDICHO_MBPE":        round(vol_pred, 2),
                    "DELTA_VS_BASE_MBPE":              delta_vs,
                    "DELTA_VS_BASE_PCT":               pct_vs,
                    "BREAKEVEN_FINANCIERO_USD_BBL":    round(bk_fin, 2) if pd.notna(bk_fin) else None,
                    "BREAKEVEN_OPERACIONAL_USD_BBL":   round(bk_pdp, 2) if pd.notna(bk_pdp) else None,
                    "ES_VIABLE":                       es_viable,
                    "ES_FULL_RESERVAS":                es_full,
                    "ES_EXTRAPOLADO":                  es_extrap,
                    "TIPO_DATO":                       "PREDICCIÓN",
                    "Q_OBJETIVO":                      q_objetivo,
                    "VIGENCIA_BASE":                   vigencia_base,
                    "FECHA_PREDICCION":                fecha_prediccion,
                })

            # ── Isotonica ─────────────────────────────────────────────────────
            delta_iso = iso_m.predict(precio_neto)
            vol_iso   = np.maximum(baseline + delta_iso, 0) if pd.notna(baseline) else delta_iso

            for i, brent in enumerate(brent_range):
                pn = float(precio_neto[i])
                pn_r = round(pn, 2)
                es_viable = (pn_r >= round(bk_pdp, 2)) if pd.notna(bk_pdp) else True
                es_full   = (pn_r >= round(bk_fin, 2)) if pd.notna(bk_fin) else True
                es_extrap = (float(brent) < bk_min_hist - MARGEN_EXTRAP_USD or
                             float(brent) > bk_max_hist + MARGEN_EXTRAP_USD)

                vol_pred = float(vol_iso[i])
                vol_base = float(baseline) if pd.notna(baseline) else np.nan
                delta_vs = round(vol_pred - vol_base, 2) if pd.notna(baseline) else np.nan
                pct_vs   = round((vol_pred - vol_base) / vol_base * 100, 2) \
                    if pd.notna(baseline) and vol_base > 0 else np.nan

                filas.append({
                    "CAMPO":                          campo,
                    "MOTOR":                          "Isotonica",
                    "ESCENARIO_DESCUENTO":            esc,
                    "BRENT_USD_BBL":                  float(brent),
                    "PRECIO_NETO_EFECTIVO_USD_BBL":   round(pn, 2),
                    "DESCUENTO_CALIDAD_USD_BBL":       round(cal, 2),
                    "DESCUENTO_TRANSPORTE_USD_BBL":    round(tra, 2),
                    "DELTA_PRED_MBPE":                round(float(delta_iso[i]), 2),
                    "VOLUMEN_1P_BASELINE_MBPE":        round(vol_base, 2) if pd.notna(vol_base) else None,
                    "VOLUMEN_1P_PREDICHO_MBPE":        round(vol_pred, 2),
                    "DELTA_VS_BASE_MBPE":              delta_vs,
                    "DELTA_VS_BASE_PCT":               pct_vs,
                    "BREAKEVEN_FINANCIERO_USD_BBL":    round(bk_fin, 2) if pd.notna(bk_fin) else None,
                    "BREAKEVEN_OPERACIONAL_USD_BBL":   round(bk_pdp, 2) if pd.notna(bk_pdp) else None,
                    "ES_VIABLE":                       es_viable,
                    "ES_FULL_RESERVAS":                es_full,
                    "ES_EXTRAPOLADO":                  es_extrap,
                    "TIPO_DATO":                       "PREDICCIÓN",
                    "Q_OBJETIVO":                      q_objetivo,
                    "VIGENCIA_BASE":                   vigencia_base,
                    "FECHA_PREDICCION":                fecha_prediccion,
                })

        # Resumen impreso con escenario BASE
        cal_base, tra_base = esc_campo["BASE"]
        vol_lo = np.maximum(baseline + xgb_m.predict(np.array([[brent_min_din + cal_base + tra_base, cal_base, tra_base]]))[0], 0) if pd.notna(baseline) else 0
        vol_hi = np.maximum(baseline + xgb_m.predict(np.array([[brent_max_din + cal_base + tra_base, cal_base, tra_base]]))[0], 0) if pd.notna(baseline) else 0
        print(f"  {campo:<20} | baseline={baseline:.1f} MBPE | "
              f"XGB@${brent_min_din}={vol_lo:.0f} -> @${brent_max_din}={vol_hi:.0f}")

    df_out = pd.DataFrame(filas)

    # ── Clasificacion de confianza por campo ─────────────────────────────────
    # Carga metricas.csv (generado por 03_modelo.py)
    ruta_met = STAGING / "metricas.csv"
    if ruta_met.exists():
        _cols_met = ["CAMPO", "N_REAL_DELTA", "MAE_LOO_XGB", "BASELINE_LATEST"]
        # columnas opcionales (pueden no existir en metricas.csv de versiones anteriores)
        _met_raw = pd.read_csv(ruta_met)
        for _c in ["ALERTA_LOO_OUTLIER_XGB"]:
            if _c not in _met_raw.columns:
                _met_raw[_c] = False
        met = _met_raw[_cols_met + ["ALERTA_LOO_OUTLIER_XGB"]]
        met["MAE_REL_LOO"] = (
            met["MAE_LOO_XGB"] / met["BASELINE_LATEST"].replace(0, np.nan)
        ).fillna(999.0)
        met["N_REAL_DELTA"] = met["N_REAL_DELTA"].fillna(0).astype(int)
        met["ALERTA_LOO_OUTLIER_XGB"] = met["ALERTA_LOO_OUTLIER_XGB"].fillna(False).astype(bool)

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
                mae_abs=float(r["MAE_LOO_XGB"]) if pd.notna(r["MAE_LOO_XGB"]) else 999.0,
                outlier_lloo=bool(r["ALERTA_LOO_OUTLIER_XGB"]),
            )
            rows_conf.append({
                "CAMPO": r["CAMPO"],
                "N_REAL_DELTA": int(r["N_REAL_DELTA"]),
                "MAE_REL_LOO": round(float(r["MAE_REL_LOO"]), 4),
                "DIVERGENCIA_MOTORES_PCT": round(float(r["DIVERGENCIA_MOTORES_PCT"]), 4),
                "ALERTA_LOO_OUTLIER_XGB": bool(r["ALERTA_LOO_OUTLIER_XGB"]),
                "NIVEL_CONFIANZA": nivel,
                "MOTIVO_CONFIANZA": motivo,
            })

        df_conf = pd.DataFrame(rows_conf)
        df_out = df_out.merge(
            df_conf[["CAMPO", "N_REAL_DELTA", "MAE_REL_LOO",
                     "DIVERGENCIA_MOTORES_PCT", "ALERTA_LOO_OUTLIER_XGB",
                     "NIVEL_CONFIANZA", "MOTIVO_CONFIANZA"]],
            on="CAMPO", how="left",
        )

        # Resumen por nivel
        conteo = df_conf.groupby("NIVEL_CONFIANZA")["CAMPO"].nunique()
        print(f"\n{'='*55}")
        print("  Clasificacion de confianza por campo")
        print(f"{'='*55}")
        for nivel in ["ALTA", "MEDIA", "BAJA", "SOLO_SINTETICO"]:
            n = conteo.get(nivel, 0)
            print(f"  {nivel:<15}: {n:3d} campos")

        # Detalle campos materiales (baseline >= 50 MBPE)
        materiales = met[met["BASELINE_LATEST"].fillna(0) >= 50].merge(
            df_conf[["CAMPO", "NIVEL_CONFIANZA"]], on="CAMPO")
        if not materiales.empty:
            print("\n  Campos materiales (baseline >= 50 MBPE):")
            for _, r in materiales.sort_values("BASELINE_LATEST", ascending=False).iterrows():
                print(f"    {r['CAMPO']:<25} baseline={r['BASELINE_LATEST']:.0f} MBPE  "
                      f"MAE_rel={r['MAE_REL_LOO']:.0%}  -> {r['NIVEL_CONFIANZA']}")
        print(f"{'='*55}\n")
    else:
        print("[WARN] metricas.csv no encontrado; NIVEL_CONFIANZA no calculado.")

    ruta_csv = RESULTADOS / "output_matriz_prediccion.csv"
    df_out.to_csv(ruta_csv, index=False, encoding="utf-8-sig")

    # Snapshot inmutable para backtesting predicho-vs-real
    # Nombre: prediccion_<Q_OBJETIVO>_generada_<FECHA>.csv — no sobreescribe corridas previas
    HISTORICO_DIR.mkdir(parents=True, exist_ok=True)
    nombre_snapshot = f"prediccion_{q_objetivo}_generada_{fecha_prediccion}.csv"
    ruta_snapshot = HISTORICO_DIR / nombre_snapshot
    shutil.copy2(ruta_csv, ruta_snapshot)
    print(f"  Snapshot fechado: {ruta_snapshot}")

    print(f"\nMatriz exportada: {ruta_csv}")
    print(f"  Filas: {len(df_out)}")
    print(f"  Extrapolados: {df_out['ES_EXTRAPOLADO'].sum()} "
          f"({df_out['ES_EXTRAPOLADO'].mean():.0%} del total)")

    # Sanity final: extremos del grid dinamico (escenario BASE)
    df_base = df_out[df_out["ESCENARIO_DESCUENTO"] == "BASE"]
    brent_lo = int(brent_range[0])
    brent_hi = int(brent_range[-1])
    check_lo = df_base[df_base["BRENT_USD_BBL"] == brent_lo].groupby(
        ["CAMPO", "MOTOR"])["VOLUMEN_1P_PREDICHO_MBPE"].mean()
    check_hi = df_base[df_base["BRENT_USD_BBL"] == brent_hi].groupby(
        ["CAMPO", "MOTOR"])["VOLUMEN_1P_PREDICHO_MBPE"].mean()

    print(f"\n  Sanity Brent=${brent_lo} (extremo bajo del grid, escenario BASE):")
    print(check_lo.to_string())
    print(f"\n  Sanity Brent=${brent_hi} (extremo alto del grid, escenario BASE):")
    print(check_hi.to_string())

    # ── Versionamiento: comparacion vs corrida anterior + changelog ──────────
    brent_ref = round((brent_obs_min + brent_obs_max) / 2)
    comp = generar_comparacion_vs_anterior(df_out, q_objetivo, brent_ref, ruta_snapshot)
    actualizar_changelog(q_objetivo, fecha_prediccion, len(campos), comp)

    print("\n=== 04_pbi_export.py — Completado ===")
