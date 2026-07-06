"""
diag_escalon.py — Diagnostico del "acantilado isotonico" en campos Pareto (analisis offline)

NO forma parte de run_pipeline.py ni altera el modelo. Responde la pregunta del informe
de gerencia 2026-07-01 (WS1, prioridad #1): el salto grande de Δreservas en CASTILLA
(+24.7%), CASTILLA NORTE (+11.4%) y CHICHIMENE SW (+31.1%) al cruzar Brent ~$72-75,
¿es efecto PRECIO real, o el punto de entrenamiento que crea el escalon coincide con
un quiebre DEFINICIONAL (2026_REGALIAS, monetizacion de regalias en el 1P certificado,
ver MAESTRO §6 columna NIVEL_DEFINICIONAL) mezclado en el target DELTA_SENS_MBPE?

Metodo: para cada campo, toma los puntos REALES de entrenamiento (ES_BASELINE=False,
ES_SINTETICO=False — los mismos que ve 03_modelo.py), los ordena por PRECIO_NETO_USD_BBL
y localiza el mayor salto absoluto de DELTA_SENS_MBPE entre puntos consecutivos en precio.
Ese es "el punto del escalon": el codo que la isotonica aprende como salto real. Se cruza
con VIGENCIA y NIVEL_DEFINICIONAL para ver si el salto coincide con el año de regalias.

Salida: resultados/diag_escalon.csv — una fila por campo Pareto con el veredicto.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR   = Path(__file__).parent
STAGING    = BASE_DIR / "datos" / "staging"
RESULTADOS = BASE_DIR / "resultados"

# Campos Pareto del informe 2026-07-01 (pareto_top10.csv) + RUBIALES como control
# (salto moderado +8.1%, sin sospecha de regalias) y CAÑO SUR ESTE como control sano
# (rampa suave, sin acantilado).
CAMPOS_DIAGNOSTICO = [
    "CASTILLA", "CASTILLA NORTE", "CHICHIMENE SW",  # sospecha alta (>10% de salto)
    "RUBIALES", "CAÑO SUR ESTE",                     # control
]


def puntos_entrenamiento(df: pd.DataFrame, campo: str) -> pd.DataFrame:
    """Replica el filtro de puntos reales que ve 03_modelo.py::entrenar_final_campo:
    ES_BASELINE=False (no es fila de nivel BASE), ES_SINTETICO=False (no inyectado),
    DELTA_SENS_MBPE no-nulo (target valido)."""
    sub = df[(df["CAMPO"] == campo) & (~df["ES_BASELINE"]) & (~df["ES_SINTETICO"])
             & df["DELTA_SENS_MBPE"].notna() & df["PRECIO_NETO_USD_BBL"].notna()]
    return sub.sort_values("PRECIO_NETO_USD_BBL").reset_index(drop=True)


def diagnosticar_campo(df: pd.DataFrame, campo: str) -> dict:
    pts = puntos_entrenamiento(df, campo)
    if len(pts) < 2:
        return {"CAMPO": campo, "N_PUNTOS_REALES": len(pts), "VEREDICTO": "INSUFICIENTE_DATA"}

    pts["DELTA_VECINO"] = pts["DELTA_SENS_MBPE"].diff().abs()
    pts["NETO_VECINO"] = pts["PRECIO_NETO_USD_BBL"].diff()
    idx_max = pts["DELTA_VECINO"].idxmax()
    if pd.isna(idx_max):
        return {"CAMPO": campo, "N_PUNTOS_REALES": len(pts), "VEREDICTO": "SIN_SALTO_DETECTABLE"}

    fila_salto = pts.loc[idx_max]
    fila_previa = pts.loc[idx_max - 1] if idx_max > 0 else None

    vigencia_salto = fila_salto.get("VIGENCIA", "")
    nivel_def_salto = fila_salto.get("NIVEL_DEFINICIONAL", "")
    es_regalias = bool(pd.notna(nivel_def_salto) and "REGALIAS" in str(nivel_def_salto).upper())

    # Vigencias mezcladas alrededor del salto (mismo hallazgo cualitativo que el caso
    # AKACIAS documentado en MAESTRO §7.0: si las dos vigencias vecinas al salto son
    # distintas, el "salto de precio" puede ser en realidad un cambio de vigencia de
    # certificacion, no una respuesta al Brent).
    vigencia_previa = fila_previa.get("VIGENCIA", "") if fila_previa is not None else ""
    vigencias_distintas = bool(vigencia_previa and vigencia_salto and vigencia_previa != vigencia_salto)

    if es_regalias:
        veredicto = "REGALIAS_CONFIRMADO"
    elif vigencias_distintas:
        veredicto = "SOSPECHA_QUIEBRE_VIGENCIA"
    else:
        veredicto = "PRECIO_LIMPIO"

    return {
        "CAMPO": campo,
        "N_PUNTOS_REALES": len(pts),
        "NETO_PREVIO_USD_BBL": round(float(fila_previa["PRECIO_NETO_USD_BBL"]), 2) if fila_previa is not None else None,
        "NETO_SALTO_USD_BBL": round(float(fila_salto["PRECIO_NETO_USD_BBL"]), 2),
        "DELTA_SALTO_MBPE": round(float(fila_salto["DELTA_VECINO"]), 2),
        "VIGENCIA_PREVIA": vigencia_previa,
        "VIGENCIA_SALTO": vigencia_salto,
        "NIVEL_DEFINICIONAL_SALTO": nivel_def_salto if pd.notna(nivel_def_salto) else "",
        "VEREDICTO": veredicto,
    }


if __name__ == "__main__":
    print("=== diag_escalon.py — Diagnostico acantilado isotonico (precio vs regalias) ===\n")

    ruta = STAGING / "tablon_unico.parquet"
    if not ruta.exists():
        raise FileNotFoundError("Ejecutar 01_etl.py primero (tablon_unico.parquet no existe).")
    df = pd.read_parquet(ruta)

    filas = [diagnosticar_campo(df, campo) for campo in CAMPOS_DIAGNOSTICO]
    df_diag = pd.DataFrame(filas)

    ruta_out = RESULTADOS / "diag_escalon.csv"
    df_diag.to_csv(ruta_out, index=False, encoding="utf-8-sig")

    print(df_diag.to_string(index=False))
    print(f"\nGuardado: {ruta_out}")

    n_regalias = (df_diag["VEREDICTO"] == "REGALIAS_CONFIRMADO").sum()
    n_vigencia = (df_diag["VEREDICTO"] == "SOSPECHA_QUIEBRE_VIGENCIA").sum()
    n_limpio = (df_diag["VEREDICTO"] == "PRECIO_LIMPIO").sum()
    print(f"\nResumen: {n_regalias} confirmados por regalias | "
          f"{n_vigencia} con sospecha de quiebre de vigencia (no regalias, pero tampoco precio "
          f"limpio) | {n_limpio} con salto atribuible a precio.")
    if n_regalias or n_vigencia:
        print("\n[ACCION] Ver docs/DIRECTRIZ_MEZCLA_VIGENCIAS.md — hallazgo requiere aprobacion "
              "de finanzas antes de modificar 01_etl.py/03_modelo.py.")
