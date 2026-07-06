"""
diag_confound_vigencia.py — Confound de vigencia en el portafolio completo (analisis offline)

NO forma parte de run_pipeline.py ni altera el modelo. Extiende el hallazgo del informe
gerencial 2026-07-01 §3ter (diag_escalon.py, 5 campos Pareto) a TODO el portafolio:
¿en cuantos campos el "escalon de precio" que aprende M1 es en realidad la diferencia
entre los pronosticos de dos AÑOS de Consolidado distintos (confound año/precio)?

Criterio (mismo que 04_pbi_export.py::calcular_confound_vigencia, unica fuente de la logica):
  - η² ≥ 0.8            : el año de vigencia explica ≥80% de la varianza del delta
  - solape < $2         : las bandas de precio neto de los años casi no se tocan
  - salto ≥ 5% baseline : el escalon inter-año es material

Resultado de referencia (corrida 2026-07-02): 46/112 campos evaluables con flag TRUE;
8 materiales (baseline ≥ 20 MBPE): RUBIALES, CASTILLA, CASTILLA NORTE, AKACIAS,
CHICHIMENE SW, LA CIRA, CUSIANA, YARIGUI-CANTAGALLO. Controles que validan el criterio:
CAÑO SUR ESTE η²=0.074 (fuera — genuinamente insensible al año) y CHICHIMENE (salto
inmaterial 3.9%, fuera).

Salida: resultados/diag_confound_vigencia.csv (una fila por campo evaluable).
"""

import importlib
from pathlib import Path

import pandas as pd

BASE_DIR   = Path(__file__).parent
STAGING    = BASE_DIR / "datos" / "staging"
RESULTADOS = BASE_DIR / "resultados"

# 04_pbi_export.py no es un identificador Python valido (empieza por digito): import dinamico
pbi = importlib.import_module("04_pbi_export")


if __name__ == "__main__":
    print("=== diag_confound_vigencia.py — Confound año/precio en el portafolio ===\n")

    ruta = STAGING / "tablon_unico.parquet"
    if not ruta.exists():
        raise FileNotFoundError("Ejecutar 01_etl.py primero (tablon_unico.parquet no existe).")
    df = pd.read_parquet(ruta)

    baselines = pbi.cargar_baselines(df)
    confound = pbi.calcular_confound_vigencia(df, baselines)

    rows = [{"CAMPO": campo, "BASELINE_MBPE": round(baselines.get(campo, float("nan")), 2),
             **vals} for campo, vals in confound.items()]
    df_out = (pd.DataFrame(rows)
              .sort_values(["FLAG", "BASELINE_MBPE"], ascending=[False, False]))

    ruta_out = RESULTADOS / "diag_confound_vigencia.csv"
    df_out.to_csv(ruta_out, index=False, encoding="utf-8-sig")

    n_flag = int(df_out["FLAG"].sum())
    materiales = df_out[(df_out["FLAG"]) & (df_out["BASELINE_MBPE"] >= 20)]
    print(f"Campos evaluables (>=4 pts reales, >=2 anios): {len(df_out)}")
    print(f"Con confound de vigencia (flag TRUE):          {n_flag}")
    print(f"Materiales (baseline >= 20 MBPE):              {len(materiales)}\n")
    print(materiales.to_string(index=False))
    print(f"\nGuardado: {ruta_out}")
    print("\n[NOTA] El flag alimenta la etiqueta de confianza (techo MEDIA en materiales,")
    print("motivo 'sensibilidad-no-identificada'). El tratamiento de la PREDICCION misma")
    print("(plano=baseline en banda) requiere aprobacion de finanzas:")
    print("ver docs/DIRECTRIZ_SENSIBILIDAD_NO_IDENTIFICADA.md")
