"""
02_synthetic.py — Inyeccion de puntos sinteticos de ancla fisica (escalera financiera/operacional)

Genera puntos sinteticos con PRECIO_NETO < BK_ANCLA_FIN (piso superior, financiero)
para anclar el modelo a la escalera de viabilidad economica validada con el equipo
financiero:

  TRAMO BAJO: [BK_ANCLA_PDP - RANGO_USD, BK_ANCLA_PDP)
    → VOLUMEN_1P_SENSIBILIDAD = 0  (todas las reservas perdidas)
    → DELTA_SENS = -BASELINE_1P    (perdida total: PDP+PNP+PND mueren)
    Justificacion: por debajo del breakeven operacional (piso de abandono), incluso
    las reservas PDP en produccion dejan de ser economicas. Ningun recurso 1P
    sobrevive.

  TRAMO ESCALERA: [BK_ANCLA_PDP, BK_ANCLA_FIN)
    → VOLUMEN_1P_SENSIBILIDAD = BASELINE_PDP  (solo PDP sobrevive)
    → DELTA_SENS = -(BASELINE_1P - BASELINE_PDP)
    Justificacion: entre el piso de abandono (operacional) y el limite economico
    (financiero), el capex hundido de PDP sigue produciendo (no se cierra), pero
    PNP/PND dejan de ser viables porque el NPV de nueva inversion es negativo a
    este precio.

  Sobre BK_ANCLA_FIN (financiero, piso superior) NO se inyecta ancla dura: la
  prediccion del modelo queda libre (puede subir o bajar con Brent), gobernada
  por la banda de datos reales.

  BRENT_INSENSITIVE: campos donde el ingreso de gas/GLP fijo domina sobre el aceite.
    El "breakeven de Brent" no tiene sentido para estos campos. No se inyectan
    sinteticos para no crear un ancla falsa.

Supuestos (validados con equipo financiero, 2026-06-09):
  - RANGO_USD=5: la banda del Tramo BAJO se extiende 5 USD por debajo de BK_ANCLA_PDP.
  - PASO_USD=1: un punto sintetico por cada USD de precio neto.
  - BRENT implicito: reconstruido desde Precio Neto usando medianas de descuentos del campo.

Re-arquitectura 2026-06-09 (escalera financiera/operacional):
  - Convencion finanzas: BK_ANCLA_FIN=piso superior (delta=0), BK_ANCLA_PDP=piso
    inferior (abandono). El swap de etiquetas se hizo en 01_etl.py::leer_breakeven.
  - Tramo BAJO = ancla total (todas las clases mueren); Tramo ESCALERA = ancla PDP
    (solo PDP sobrevive entre los dos pisos).
  - ESCALERA_DEGENERADA (BK_ANCLA_FIN<=BK_ANCLA_PDP): un solo piso, sin tramo ESCALERA.
  - BRENT_INSENSITIVE → sin ancla (explícito, no silencio)
  - Idempotente: elimina sinteticos previos antes de regenerar
  - RANGO_USD: 20 → 5, para no opacar los puntos reales en el entrenamiento
    (ver peso sintetico dinamico en 03_modelo.py).

Ver docs/MAESTRO.md §8 para la justificacion detallada del anclaje fisico por clase.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
STAGING  = BASE_DIR / "datos" / "staging"

RANGO_USD = 5    # puntos en Tramo BAJO: [bk_inf - RANGO, bk_inf)
PASO_USD  = 1


def calcular_medianas_precio(df: pd.DataFrame) -> pd.DataFrame:
    """Medianas de Calidad y Transporte por campo (filas reales con precio disponible)."""
    df_real = df[
        (~df["ES_SINTETICO"]) &
        df["BRENT_FLAT_USD_BBL"].notna() &
        df["DESCUENTO_CALIDAD_USD_BBL"].notna() &
        df["DESCUENTO_TRANSPORTE_USD_BBL"].notna()
    ]
    return (df_real
            .groupby("CAMPO")[["BRENT_FLAT_USD_BBL",
                                "DESCUENTO_CALIDAD_USD_BBL",
                                "DESCUENTO_TRANSPORTE_USD_BBL"]]
            .median()
            .reset_index()
            .rename(columns={
                "BRENT_FLAT_USD_BBL":             "MED_BRENT",
                "DESCUENTO_CALIDAD_USD_BBL":      "MED_CALIDAD",
                "DESCUENTO_TRANSPORTE_USD_BBL":   "MED_TRANSPORTE",
            }))


def calcular_baselines_latest(df: pd.DataFrame) -> dict:
    """
    Ultimo VOLUMEN_1P_OFICIAL_MBPE certificado por campo.
    Es el BASELINE total contra el que se calcula el DELTA_SENS en espacio delta.
    """
    df_base = df[(df["ESCENARIO"] == "BASE") & df["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
    return (df_base.sort_values("AÑO")
            .groupby("CAMPO")["VOLUMEN_1P_OFICIAL_MBPE"]
            .last()
            .to_dict())


def calcular_baselines_por_clase(df: pd.DataFrame) -> dict:
    """
    Ultimo VOLUMEN_PDP_MBPE y VOLUMEN_PNP+PND_MBPE por campo.
    Usados en el Tramo 2: el campo pierde PDP pero retiene PNP+PND.
    Retorna dict: {campo: {"pdp": float, "pnp_pnd": float}}
    """
    df_base = df[(df["ESCENARIO"] == "BASE")]
    result = {}
    for campo, sub in df_base.groupby("CAMPO"):
        sub = sub.sort_values("AÑO")
        # Ultimo anio con datos de PDP
        pdp_vals = sub["VOLUMEN_PDP_MBPE"].dropna().values
        pdp = float(pdp_vals[-1]) if len(pdp_vals) > 0 else 0.0
        # PNP + PND del mismo ultimo anio
        pnp_vals = sub["VOLUMEN_PNP_MBPE"].dropna().values
        pnd_vals = sub["VOLUMEN_PND_MBPE"].dropna().values
        pnp = float(pnp_vals[-1]) if len(pnp_vals) > 0 else 0.0
        pnd = float(pnd_vals[-1]) if len(pnd_vals) > 0 else 0.0
        result[campo] = {"pdp": pdp, "pnp_pnd": pnp + pnd}
    return result


def _fila_sintetica(campo, pneto, med_cal, med_tra, vbk_v, bk_fin, bk_op_val,
                   vol_1p, delta, baseline) -> dict:
    """Construye una fila sintetica con todos los campos canonicos del tablon."""
    brent_impl = pneto - med_cal - med_tra
    return {
        "CAMPO":                          campo,
        "CAMPO_ORIGEN_RAW":               campo,
        "AÑO":                            9999,
        "VIGENCIA":                       "SINTETICO",
        "ESCENARIO":                      "SINTETICO",
        "ES_BASELINE":                    False,
        "ES_SINTETICO":                   True,
        "NIVEL_DEFINICIONAL":             "",
        "BRENT_FLAT_USD_BBL":             round(brent_impl, 4),
        "DESCUENTO_CALIDAD_USD_BBL":      med_cal,
        "DESCUENTO_TRANSPORTE_USD_BBL":   med_tra,
        "PRECIO_NETO_USD_BBL":            round(pneto, 4),
        "VOLUMEN_PDP_MBPE":               np.nan,
        "VOLUMEN_PNP_MBPE":               np.nan,
        "VOLUMEN_PND_MBPE":               np.nan,
        "VOLUMEN_1P_OFICIAL_MBPE":        np.nan,
        "BASELINE_1P_VIGENCIA_MBPE":      baseline,
        "VOLUMEN_1P_SENSIBILIDAD_MBPE":   vol_1p,
        "DELTA_SENS_MBPE":                delta,
        "BREAKEVEN_FINANCIERO_USD_BBL":   bk_fin,
        "BREAKEVEN_OPERACIONAL_USD_BBL":  bk_op_val,
        # Redondear a 4dp para que coincida con PRECIO_NETO_USD_BBL y las comparaciones
        # >= BK_ANCLA_PDP / < BK_ANCLA_FIN no fallen por error de float en los bordes
        # de los tramos BAJO/ESCALERA.
        "BK_ANCLA_FIN_USD_BBL":           round(bk_fin, 4),
        "BK_ANCLA_PDP_USD_BBL":           round(bk_op_val, 4) if pd.notna(bk_op_val) else bk_op_val,
        "BRENT_INSENSITIVE":              False,
        "VIGENCIA_BREAKEVEN":             vbk_v,
        "PRED_XGBOOST_MBPE":              np.nan,
        "PRED_ISOTONICA_MBPE":            np.nan,
        "DELTA_XGBOOST_VS_OFICIAL":       np.nan,
        "DELTA_ISOTONICA_VS_OFICIAL":     np.nan,
        "ALERTA":                         "",
        "HOMOLOG_FLAG":                   "OK",
    }


def generar_sinteticos(df: pd.DataFrame, medianas: pd.DataFrame,
                       baselines: dict, baselines_clase: dict) -> pd.DataFrame:
    """
    Por cada campo genera puntos sub-financiero en la ESCALERA de dos tramos:

      Tramo BAJO: [BK_ANCLA_PDP - RANGO, BK_ANCLA_PDP)
        → Vol 1P = 0 (abandono total: PDP+PNP+PND mueren)

      Tramo ESCALERA: [BK_ANCLA_PDP, BK_ANCLA_FIN)  — solo si bk_sup > bk_inf + PASO
        y BASELINE_PDP > 0
        → Vol 1P = BASELINE_PDP (solo PDP sobrevive; PNP/PND dejan de ser viables)

    Sobre BK_ANCLA_FIN no se inyecta ancla: la prediccion del modelo queda libre.
    Campos BRENT_INSENSITIVE: se omiten con advertencia explicita.
    Si BK_ANCLA_FIN es nan, se usa BK_ANCLA_PDP como unico piso (tramo BAJO solo).
    """
    filas = []
    for campo in df["CAMPO"].unique():
        sub = df[df["CAMPO"] == campo]

        # Verificar si el campo es Brent-insensible
        insens_vals = sub["BRENT_INSENSITIVE"].dropna().values
        if len(insens_vals) > 0 and bool(insens_vals[0]):
            print(f"  [WARN] {campo}: Brent-insensible, sin ancla sintetica")
            continue

        # Leer anclas por clase del tablon (post-swap: FIN=piso superior, PDP=piso inferior).
        # FIN y PDP deben venir de la MISMA vigencia (calcular_breakeven_ponderado los
        # calcula en pareja por CAMPO×VIGENCIA): mezclar vigencias puede invertir los
        # pisos (ej. FIN de 2024 < PDP de 2025). Se usa la vigencia mas reciente.
        anclas_sub = sub.dropna(subset=["BK_ANCLA_FIN_USD_BBL", "BK_ANCLA_PDP_USD_BBL"], how="all")

        # Si no hay ningun ancla, omitir
        if anclas_sub.empty:
            print(f"  [WARN] {campo}: sin breakeven disponible, omitiendo sinteticos")
            continue

        fila_ancla = anclas_sub.sort_values("VIGENCIA_BREAKEVEN", ascending=False).iloc[0]
        bk_sup = float(fila_ancla["BK_ANCLA_FIN_USD_BBL"]) if pd.notna(fila_ancla["BK_ANCLA_FIN_USD_BBL"]) else np.nan
        bk_inf = float(fila_ancla["BK_ANCLA_PDP_USD_BBL"]) if pd.notna(fila_ancla["BK_ANCLA_PDP_USD_BBL"]) else np.nan

        # Si solo uno esta disponible, usar el mismo para ambos (tramo unico)
        if np.isnan(bk_sup):
            bk_sup = bk_inf
        if np.isnan(bk_inf):
            bk_inf = bk_sup

        med_row = medianas[medianas["CAMPO"] == campo]
        if med_row.empty:
            print(f"  [WARN] {campo}: sin medianas de precio, omitiendo sinteticos")
            continue

        med_cal = float(med_row["MED_CALIDAD"].values[0])
        med_tra = float(med_row["MED_TRANSPORTE"].values[0])

        baseline_total = baselines.get(campo, np.nan)
        bl_clase       = baselines_clase.get(campo, {"pdp": 0.0, "pnp_pnd": 0.0})
        baseline_pdp   = bl_clase["pdp"]

        vbk_vals = sub["VIGENCIA_BREAKEVEN"].dropna().values
        vbk_v    = str(vbk_vals[0]) if len(vbk_vals) > 0 else "2024"

        # ── Tramo BAJO: [bk_inf - RANGO, bk_inf) ──────────────────────────────
        # Bajo el piso de abandono (operacional): ninguna reserva 1P es economica.
        delta_total = (-float(baseline_total)) if pd.notna(baseline_total) else np.nan
        for pneto in np.arange(bk_inf - RANGO_USD, bk_inf, PASO_USD):
            filas.append(_fila_sintetica(campo, pneto, med_cal, med_tra, vbk_v,
                                         bk_sup, bk_inf, 0.0, delta_total,
                                         baseline_total))

        # ── Tramo ESCALERA: [bk_inf, bk_sup) ──────────────────────────────────
        # Entre el piso de abandono y el limite economico (financiero): el capex
        # hundido de PDP sigue produciendo, pero PNP+PND dejan de ser viables.
        # Solo se genera si hay una brecha real entre pisos (> PASO_USD) y PDP > 0.
        # Si PDP = 0 (campo solo-PNP/PND), ESCALERA seria identico a BAJO → omitir.
        if bk_sup - bk_inf > PASO_USD and (baseline_pdp or 0.0) > 0.0:
            vol_pdp   = baseline_pdp if pd.notna(baseline_pdp) else np.nan
            delta_esc = (-(float(baseline_total) - float(baseline_pdp))
                          if pd.notna(baseline_total) else np.nan)
            for pneto in np.arange(bk_inf, bk_sup, PASO_USD):
                filas.append(_fila_sintetica(campo, pneto, med_cal, med_tra, vbk_v,
                                              bk_sup, bk_inf, vol_pdp, delta_esc,
                                              baseline_total))

    return pd.DataFrame(filas)


def validar_sinteticos(df_sint: pd.DataFrame) -> None:
    sep = "-" * 70
    print(f"\n{sep}\n  Resumen sinteticos generados\n{sep}")
    print(f"  Total filas sinteticas : {len(df_sint)}")
    for campo in sorted(df_sint["CAMPO"].unique()):
        sub   = df_sint[df_sint["CAMPO"] == campo]
        bajo  = sub[sub["VOLUMEN_1P_SENSIBILIDAD_MBPE"] == 0.0]
        esc   = sub[sub["VOLUMEN_1P_SENSIBILIDAD_MBPE"] != 0.0]
        bk_f  = sub["BK_ANCLA_FIN_USD_BBL"].values[0]
        bk_p  = sub["BK_ANCLA_PDP_USD_BBL"].values[0]
        print(f"  {campo:<20} | BAJO={len(bajo):3d} pts (vol=0)  "
              f"ESCALERA={len(esc):3d} pts (vol=PDP) "
              f"| BK_FIN={bk_f:.2f}  BK_PDP={bk_p:.2f}")
    print(sep + "\n")


if __name__ == "__main__":
    print("=== 02_synthetic.py — Inyeccion de sinteticos (escalera financiero/operacional) ===\n")

    ruta = STAGING / "tablon_unico.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"Ejecutar 01_etl.py primero. No encontrado: {ruta}")

    df = pd.read_parquet(ruta)

    # Eliminar sinteticos previos (idempotente)
    n_prev = int(df["ES_SINTETICO"].sum())
    if n_prev > 0:
        print(f"  [INFO] Eliminando {n_prev} sinteticos previos")
        df = df[~df["ES_SINTETICO"]].copy()

    print("[1/4] Calculando medianas de precio por campo...")
    medianas = calcular_medianas_precio(df)
    for _, r in medianas.iterrows():
        print(f"  {r['CAMPO']:<20} | med_brent={r['MED_BRENT']:.2f} | "
              f"cal={r['MED_CALIDAD']:.2f} | tra={r['MED_TRANSPORTE']:.2f}")

    print("\n[2/4] Calculando baselines totales (ultimo OFICIAL por campo)...")
    baselines = calcular_baselines_latest(df)
    for campo, bl in sorted(baselines.items()):
        print(f"  {campo:<20} | baseline_total={bl:.2f} MBPE")

    print("\n[3/4] Calculando baselines por clase (PDP / PNP+PND)...")
    baselines_clase = calcular_baselines_por_clase(df)
    for campo, bl in sorted(baselines_clase.items()):
        print(f"  {campo:<20} | PDP={bl['pdp']:.2f}  PNP+PND={bl['pnp_pnd']:.2f} MBPE")

    print(f"\n[4/4] Generando sinteticos (BAJO: rango BK_PDP-{RANGO_USD}; "
          f"ESCALERA: entre BK_PDP y BK_FIN)...")
    df_sint = generar_sinteticos(df, medianas, baselines, baselines_clase)

    validar_sinteticos(df_sint)

    df_completo = pd.concat([df, df_sint], ignore_index=True)
    df_completo = df_completo.sort_values(
        ["CAMPO", "ESCENARIO", "PRECIO_NETO_USD_BBL"]).reset_index(drop=True)

    df_completo.to_parquet(STAGING / "tablon_unico.parquet", index=False)
    df_completo.to_csv(STAGING / "tablon_unico.csv", index=False, encoding="utf-8-sig")

    total_sint = int(df_completo["ES_SINTETICO"].sum())
    print(f"  Tablon actualizado: {len(df_completo)} filas ({total_sint} sinteticas, "
          f"{len(df)} reales)")
    print("\n=== 02_synthetic.py — Completado ===")
