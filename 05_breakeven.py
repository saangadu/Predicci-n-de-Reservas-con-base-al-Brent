"""
05_breakeven.py — Runner del motor de Breakeven (Fase 0.5 del pipeline)

Procesa los FC de cada vigencia y escribe el tablón
`datos/staging/breakeven_resultados.parquet` (+ .csv) que `01_etl.py` consume.
Reemplaza la corrida manual del VBA (Solver + GoalSeek) en Excel.

Fuente de FC (por orden de prioridad):
  1. Rutas externas read-only (portafolio completo ~142/128 campos):
       2024 → FC_DIR_EXT_2024
       2025 → FC_DIR_EXT_2025
  2. Carpetas locales fallback (datos/raw/Formatos FC/2024|2025, 16 campos c/u).
     Si la ruta externa no existe o está vacía, se usa la local.

Al terminar, genera el Excel de auditoría:
  resultados/auditoria_breakeven_2024_2025.xlsx
  — Compara motor Python vs Solver de Breakeven.xlsm
"""

from pathlib import Path

import numpy as np
import pandas as pd

import motor_breakeven as mbk
from track import sufijo_track, flag

_SUF = sufijo_track()   # '' Produccion; '_calidad' si PRED_TRACK=calidad
BASE_DIR = Path(__file__).parent
STAGING  = BASE_DIR / "datos" / f"staging{_SUF}"
STAGING.mkdir(parents=True, exist_ok=True)
RESULTADOS = BASE_DIR / f"resultados{_SUF}"
RESULTADOS.mkdir(parents=True, exist_ok=True)

# ── Rutas FC (externas read-only, portafolio completo) ────────────────────────
FC_DIR_EXT_2024 = Path(
    r"C:\Users\ECOPETROL\Documents\GALLEGO\RR\2024\FC Finales\Ecopetrol"
)
FC_DIR_EXT_2025 = Path(
    r"C:\Users\ECOPETROL\Documents\GALLEGO\RR\2025\FC Finales"
    r"\Departamento de Recursos y Reservas - Finales GCR"
)
FC_DIR_LOCAL = BASE_DIR / "datos" / "raw" / "Formatos FC"


def _archivos_vigencia(subcarpeta: str) -> list[Path]:
    """Devuelve lista de .xlsx de la vigencia. Prefiere ruta externa; fallback local."""
    ext = FC_DIR_EXT_2024 if subcarpeta == "2024" else FC_DIR_EXT_2025
    if ext.exists():
        archivos = sorted(ext.glob("*.xlsx"))
        if archivos:
            return archivos
        print(f"  [WARN] Ruta externa {ext} existe pero está vacía; usando local.")

    local = FC_DIR_LOCAL / subcarpeta
    if local.exists():
        archivos = sorted(local.glob("*.xlsx"))
        print(f"  [INFO] Usando carpeta local {local} ({len(archivos)} archivos).")
        return archivos

    print(f"  [WARN] No se encontró FC para vigencia {subcarpeta}. Omitiendo.")
    return []


def _generar_auditoria(df_py: pd.DataFrame) -> None:
    """
    Compara breakevens del motor Python vs. Solver de Breakeven.xlsm.
    Escribe resultados/auditoria_breakeven_2024_2025.xlsx.
    """
    xlsm = BASE_DIR / "datos" / "raw" / "Breakeven.xlsm"
    if not xlsm.exists():
        print("  [WARN] Breakeven.xlsm no encontrado; omitiendo auditoría vs Solver.")
        return

    raw = pd.read_excel(xlsm, sheet_name="Resultados Solver", engine="openpyxl")
    raw.columns = [str(c).strip() for c in raw.columns]

    # Extraer campo y clase del Solver
    raw["CAMPO_CRUDO"] = raw.iloc[:, 0].astype(str)
    raw["CLASE_RESERVA"] = raw.iloc[:, 1].astype(str).str.strip()
    # Columna Var. Final S1 = BK Operacional; Var. Final B.O. = BK Financiero
    col_op  = raw.columns[5]   # Var. Final S1 (EL3) = BREAKEVEN OPERACIONAL
    col_msg_s1 = raw.columns[7]
    col_fin = raw.columns[11]  # Var. Final B.O. (EL3) = BREAKEVEN FINANCIERO NPV=0
    col_msg_bo = raw.columns[13]

    raw = raw.rename(columns={
        col_op:     "SOLVER_BK_OPERACIONAL",
        col_msg_s1: "SOLVER_MSG_S1",
        col_fin:    "SOLVER_BK_FINANCIERO",
        col_msg_bo: "SOLVER_MSG_BO",
    })

    # Homologar nombre de campo del Solver
    from homologacion import Homologador
    H = Homologador()
    raw["CAMPO"] = raw["CAMPO_CRUDO"].apply(lambda x: H.homologar(x)[0])

    # Solo clases que el motor Python también procesa
    raw = raw[raw["CLASE_RESERVA"].isin({"D-PDP", "D-PNP", "D-PND"})].copy()
    for c in ["SOLVER_BK_OPERACIONAL", "SOLVER_BK_FINANCIERO"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # Usar solo vigencia 2024 del motor Python para comparar vs. Solver (Cierre 2024)
    py24 = df_py[df_py["VIGENCIA_BREAKEVEN"] == "2024"][
        ["CAMPO", "VIGENCIA_BREAKEVEN", "CLASE_RESERVA",
         "BREAKEVEN_OPERACIONAL_USD_BBL", "BREAKEVEN_FINANCIERO_USD_BBL",
         "MENSAJE_OPE", "MENSAJE_FIN", "BRENT_INSENSITIVE"]
    ].copy()
    py24 = py24.rename(columns={
        "BREAKEVEN_OPERACIONAL_USD_BBL": "PY_BK_OPERACIONAL",
        "BREAKEVEN_FINANCIERO_USD_BBL":  "PY_BK_FINANCIERO",
        "MENSAJE_OPE": "PY_MSG_OPE",
        "MENSAJE_FIN": "PY_MSG_FIN",
    })

    merged = raw.merge(py24, on=["CAMPO", "CLASE_RESERVA"], how="outer")

    # Deltas
    merged["DELTA_BK_FINANCIERO"] = merged["PY_BK_FINANCIERO"] - merged["SOLVER_BK_FINANCIERO"]
    merged["DELTA_BK_OPERACIONAL"] = merged["PY_BK_OPERACIONAL"] - merged["SOLVER_BK_OPERACIONAL"]

    # Resumen por hoja separada
    fin_ok  = merged["PY_MSG_FIN"].isin(mbk.MENSAJES_OK)
    sol_ok  = merged["SOLVER_MSG_BO"] == "OK"
    ope_ok  = merged["PY_MSG_OPE"].isin(mbk.MENSAJES_OK)

    # Comparación solo donde ambos resolvieron
    ambos_fin = merged[fin_ok & sol_ok & merged["DELTA_BK_FINANCIERO"].notna()]
    ambos_ope = merged[ope_ok & (merged["SOLVER_MSG_S1"] == "OK") & merged["DELTA_BK_OPERACIONAL"].notna()]

    def _pct(df_comp, col_delta, tol):
        if df_comp.empty:
            return 0.0
        return round(100 * (df_comp[col_delta].abs() <= tol).sum() / len(df_comp), 1)

    resumen = pd.DataFrame([
        {"Métrica": "Campos con FC Python procesados (vigencia 2024)", "Valor": merged["CAMPO"].nunique()},
        {"Métrica": "Financiero OK en Python",   "Valor": int(fin_ok.sum())},
        {"Métrica": "Financiero OK en Solver",   "Valor": int(sol_ok.sum())},
        {"Métrica": "Solver falló pero Python OK (Financiero)",
         "Valor": int((~sol_ok & fin_ok).sum())},
        {"Métrica": "Operacional OK en Python",  "Valor": int(ope_ok.sum())},
        {"Métrica": "Operacional OK en Solver",  "Valor": int((merged['SOLVER_MSG_S1'] == 'OK').sum())},
        {"Métrica": "Solver S1 falló pero Python OK (Operacional)",
         "Valor": int(((merged['SOLVER_MSG_S1'] != 'OK') & ope_ok).sum())},
        {"Métrica": "BRENT_INSENSITIVE en Python", "Valor": int(merged["BRENT_INSENSITIVE"].fillna(False).sum())},
        {"Métrica": "Delta Fin <= ±0.5 USD/bbl (sobre ambos-OK)", "Valor": f"{_pct(ambos_fin, 'DELTA_BK_FINANCIERO', 0.5)}%"},
        {"Métrica": "Delta Fin <= ±2.0 USD/bbl (sobre ambos-OK)", "Valor": f"{_pct(ambos_fin, 'DELTA_BK_FINANCIERO', 2.0)}%"},
        {"Métrica": "Delta Ope <= ±0.5 USD/bbl (sobre ambos-OK)", "Valor": f"{_pct(ambos_ope, 'DELTA_BK_OPERACIONAL', 0.5)}%"},
        {"Métrica": "Delta Ope <= ±2.0 USD/bbl (sobre ambos-OK)", "Valor": f"{_pct(ambos_ope, 'DELTA_BK_OPERACIONAL', 2.0)}%"},
    ])

    # Estadísticas de delta
    for label, df_comp, col_delta in [
        ("Financiero", ambos_fin, "DELTA_BK_FINANCIERO"),
        ("Operacional", ambos_ope, "DELTA_BK_OPERACIONAL"),
    ]:
        if not df_comp.empty:
            d = df_comp[col_delta].abs()
            resumen = pd.concat([resumen, pd.DataFrame([
                {"Métrica": f"MAE {label} (USD/bbl)",  "Valor": round(float(d.mean()), 3)},
                {"Métrica": f"Max delta {label} (USD/bbl)", "Valor": round(float(d.max()), 3)},
            ])], ignore_index=True)

    ruta_audit = RESULTADOS / "auditoria_breakeven_2024_2025.xlsx"
    with pd.ExcelWriter(ruta_audit, engine="openpyxl") as xw:
        resumen.to_excel(xw, sheet_name="Resumen", index=False)
        merged.to_excel(xw, sheet_name="Detalle_2024", index=False)
        df_py.to_excel(xw, sheet_name="Motor_Python_Todos", index=False)

    print(f"\n  Auditoría guardada: {ruta_audit}")
    print("\n  === Resumen comparación Python vs Solver ===")
    for _, r in resumen.iterrows():
        print(f"    {r['Métrica']:<55} {r['Valor']}")


def combinar_multi_fc(filas: list[dict], con_perfil: bool = False) -> list[dict]:
    """
    Colapsa los duplicados (CAMPO, VIGENCIA, CLASE) que aparecen cuando un UNIFICADO
    tiene varios archivos FC físicos (CHICHIMENE+SW, LISAMA+NUTRIA+TESORO, NARE×4...).

    Directriz §4bis.3 (2026-07-11): los componentes certifican juntos → se SUMAN los
    flujos de caja año a año (mbk.combinar_hojas) y se calcula UNA raíz sobre el FC
    combinado. Antes, 01_etl tomaba la primera fila alfabética y botaba el resto en
    silencio (espejo del bug coalesce v3).

    Requiere filas de procesar_fc(retener_datos=True). Descarta "_DATOS" del output.

    con_perfil=True (track Calidad, directriz §4bis.6): agrega columnas BK_P10..BK_P90
    (perfil de salida volumétrico) calculadas del MISMO FC (combinado si aplica).
    """
    grupos: dict[tuple, list[dict]] = {}
    for f in filas:
        grupos.setdefault(
            (f["CAMPO"], f["VIGENCIA_BREAKEVEN"], f["CLASE_RESERVA"]), []).append(f)

    out = []
    n_combinados = 0
    for (campo, vig, clase), grupo in grupos.items():
        con_datos = [f for f in grupo if f.get("_DATOS") is not None]
        if len(grupo) == 1 or len(con_datos) <= 1:
            # Mono-archivo, o un solo componente con reservas (el resto SIN_RESERVAS):
            # conservar la fila con datos si existe; si no, la primera.
            fila = con_datos[0] if con_datos else grupo[0]
            d_uni = fila.get("_DATOS")
            fila.pop("_DATOS", None)
            if con_perfil and d_uni is not None:
                fila.update(mbk.perfil_salida(d_uni))
            out.append(fila)
            continue

        ds = [f["_DATOS"] for f in con_datos]
        try:
            d_comb = mbk.combinar_hojas(ds)
        except ValueError as e:
            print(f"    [WARN] {campo} [{vig}] {clase}: combinación falló ({e}); "
                  f"usando el componente de mayor horizonte.")
            d_comb = max(ds, key=lambda d: d["_ult"])
        fin, msg_fin = mbk.breakeven_financiero(d_comb)
        ope, msg_ope = mbk.breakeven_operacional(d_comb)
        fila = mbk._fila(f"COMBINADO(n={len(con_datos)})", campo, vig, clase,
                         fin, ope, msg_fin, msg_ope)
        if con_perfil:
            fila.update(mbk.perfil_salida(d_comb))
        out.append(fila)
        n_combinados += 1
        indiv = ", ".join(f"{f['BREAKEVEN_OPERACIONAL_USD_BBL']:.1f}"
                          if pd.notna(f['BREAKEVEN_OPERACIONAL_USD_BBL']) else "N/A"
                          for f in con_datos)
        ope_str = f"{ope:.2f}" if pd.notna(ope) else "N/A"
        print(f"    [COMBINADO] {campo} [{vig}] {clase}: {len(con_datos)} FCs "
              f"(ope indiv: {indiv}) -> ope_comb={ope_str}")

    if n_combinados:
        print(f"  [INFO] {n_combinados} grupos multi-FC combinados por suma de flujos "
              f"(directriz §4bis.3)")
    return out


if __name__ == "__main__":
    print("=== 05_breakeven.py — Motor de Breakeven (Python) ===\n")

    filas = []
    for sub in ["2024", "2025"]:
        archivos = _archivos_vigencia(sub)
        print(f"  {sub}: {len(archivos)} archivos FC")
        for ruta in archivos:
            try:
                filas.extend(mbk.procesar_fc(ruta, retener_datos=True))
            except Exception as e:
                print(f"    [ERROR] {ruta.name}: {e}")

    if not filas:
        raise FileNotFoundError(
            "No se encontraron FC en rutas externas ni locales.")

    print("\n  Combinando componentes multi-FC de campos UNIFICADO...")
    _perfil = flag("PRED_PERFIL_SALIDA")
    if _perfil:
        print("  [CALIDAD] Perfil de salida volumétrico ON (BK_P10..BK_P90, directriz §4bis.6)")
    filas = combinar_multi_fc(filas, con_perfil=_perfil)

    df = pd.DataFrame(filas)

    ruta_pq  = STAGING / "breakeven_resultados.parquet"
    ruta_csv = STAGING / "breakeven_resultados.csv"
    df.to_parquet(ruta_pq, index=False)
    df.to_csv(ruta_csv, index=False, encoding="utf-8-sig")

    # ── Resumen ──────────────────────────────────────────────────────────────────
    print(f"\nResultados: {len(df)} filas (campo x vigencia x clase)")
    print(f"  Campos únicos: {df['CAMPO'].nunique()}")
    print(f"  Guardado: {ruta_pq}")

    ok_fin = df["MENSAJE_FIN"].isin(mbk.MENSAJES_OK).sum()
    ok_ope = df["MENSAJE_OPE"].isin(mbk.MENSAJES_OK).sum()
    print(f"\n  Financiero  OK: {ok_fin}/{len(df)}")
    print(f"  Operacional OK: {ok_ope}/{len(df)}")
    print(f"  BRENT_INSENSITIVE: {df['BRENT_INSENSITIVE'].sum()}")
    print("\n  Por vigencia:")
    print(df.groupby("VIGENCIA_BREAKEVEN")["MENSAJE_FIN"]
          .apply(lambda s: f"{(s == 'OK').sum()}/{len(s)} OK").to_string())

    # Breakeven Financiero D-PDP por campo (el más relevante para CAPEX)
    pdp = df[(df["CLASE_RESERVA"] == "D-PDP") & df["MENSAJE_FIN"].isin(mbk.MENSAJES_OK)]
    print(f"\n  Breakeven Financiero D-PDP (USD/bbl NET) — {pdp['CAMPO'].nunique()} campos:")
    for _, r in pdp.sort_values(["CAMPO", "VIGENCIA_BREAKEVEN"]).iterrows():
        ope_val = r['BREAKEVEN_OPERACIONAL_USD_BBL']
        ope_str = f"{ope_val:.2f}" if pd.notna(ope_val) else "N/A"
        print(f"    {r['CAMPO']:<30} [{r['VIGENCIA_BREAKEVEN']}] "
              f"Fin={r['BREAKEVEN_FINANCIERO_USD_BBL']:.2f}  Ope={ope_str}")

    # Auditoría vs Solver
    print("\n  Generando auditoría vs Solver de Breakeven.xlsm...")
    _generar_auditoria(df)

    print("\n=== 05_breakeven.py — Completado ===")
