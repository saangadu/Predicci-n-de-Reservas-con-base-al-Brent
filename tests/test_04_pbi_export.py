"""
test_04_pbi_export.py — Gate Fase 4: verifica los CSV de exportacion para Power BI

Pruebas: ES_VIABLE correcto (marco neto), ES_EXTRAPOLADO presente, monotonia,
re-anclaje Vol(BRENT_REF)=baseline, piso duro sub-abandono, tag M2_ES_FALLBACK,
3 matrices (M1 puro / M2 puro / cadena completa), 0 NaN en export.
Escenarios BAJO/ALTO retirados 2026-06-12.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

STAGING    = ROOT / "datos" / "staging"
RESULTADOS = ROOT / "resultados"
CAMPOS_PILOTO = ["CASTILLA", "CASTILLA NORTE", "CASTILLA ESTE", "RUBIALES"]


@pytest.fixture(scope="module")
def df_export():
    ruta = RESULTADOS / "output_matriz_prediccion.csv"
    assert ruta.exists(), "output_matriz_prediccion.csv no existe: correr 04_pbi_export.py"
    return pd.read_csv(ruta)


@pytest.fixture(scope="module")
def tablon():
    ruta = STAGING / "tablon_unico.parquet"
    assert ruta.exists()
    return pd.read_parquet(ruta)


def test_csv_existe():
    assert (RESULTADOS / "output_matriz_prediccion.csv").exists()


def test_tres_matrices_existen():
    """Directriz 2026-06-12: matrices aisladas por modelo para ubicar el origen de errores."""
    assert (RESULTADOS / "output_matriz_modelo1.csv").exists(), "Falta matriz M1 pura"
    assert (RESULTADOS / "output_matriz_modelo2.csv").exists(), "Falta matriz M2 pura"
    m1 = pd.read_csv(RESULTADOS / "output_matriz_modelo1.csv", nrows=5)
    m2 = pd.read_csv(RESULTADOS / "output_matriz_modelo2.csv", nrows=5)
    for c in ["CAMPO", "MOTOR", "PRECIO_ACEITE_USD_BBL", "DELTA_ANCLADO_MBPE",
              "VOLUMEN_1P_PREDICHO_MBPE", "P_REF_USD_BBL"]:
        assert c in m1.columns, f"Falta {c} en output_matriz_modelo1.csv"
    for c in ["CAMPO", "BRENT_USD_BBL", "PRECIO_ACEITE_USD_BBL", "M2_METODO",
              "M2_ES_FALLBACK", "R2", "R2_LOO"]:
        assert c in m2.columns, f"Falta {c} en output_matriz_modelo2.csv"


def test_columnas_obligatorias(df_export):
    cols = ["CAMPO", "MOTOR", "BRENT_USD_BBL",
            "PRECIO_NETO_EFECTIVO_USD_BBL", "DELTA_PRED_MBPE",
            "VOLUMEN_1P_BASELINE_MBPE", "VOLUMEN_1P_PREDICHO_MBPE",
            "BRENT_REF_USD_BBL", "P_REF_USD_BBL", "M2_METODO", "M2_ES_FALLBACK",
            "BREAKEVEN_FINANCIERO_USD_BBL", "BREAKEVEN_OPERACIONAL_USD_BBL",
            "ES_VIABLE", "ES_FULL_RESERVAS", "ES_EXTRAPOLADO",
            "NIVEL_CONFIANZA"]
    for c in cols:
        assert c in df_export.columns, f"Columna {c} faltante en export"


def test_sin_columnas_de_escenarios(df_export):
    """Escenarios BAJO/ALTO y descuento implicito retirados del flujo (2026-06-12)."""
    for c in ["ESCENARIO_DESCUENTO", "DESCUENTO_IMPLICITO_USD_BBL"]:
        assert c not in df_export.columns, f"Columna de escenarios '{c}' aun presente"


def test_sin_columnas_opp1(df_export):
    """Opp-1 (correccion de bias post-hoc) fue retirado: estas columnas no deben existir."""
    for c in ["VOLUMEN_1P_CORREGIDO_MBPE", "BIAS_CORRECCION_MBPE"]:
        assert c not in df_export.columns, f"Columna Opp-1 '{c}' aun presente en export"


def test_sin_nan_en_vol_predicho(df_export):
    """VOLUMEN_1P_PREDICHO_MBPE no puede ser NaN (seria un campo sin modelo)."""
    nulos = df_export["VOLUMEN_1P_PREDICHO_MBPE"].isnull().sum()
    assert nulos == 0, f"VOLUMEN_1P_PREDICHO_MBPE: {nulos} NaN en export"


def test_motores_presentes(df_export):
    """Arquitectura 1D: motores Isotonica (primario) y Suave (validacion). XGBoost retirado."""
    motores = set(df_export["MOTOR"].unique())
    assert "Isotonica" in motores
    assert "Suave" in motores
    assert "XGBoost" not in motores, "XGBoost fue retirado; no debe aparecer en el export"


def test_vol_predicho_no_negativo(df_export):
    """Volumenes predichos deben ser >= 0 (reservas no pueden ser negativas)."""
    neg = (df_export["VOLUMEN_1P_PREDICHO_MBPE"] < 0).sum()
    assert neg == 0, f"{neg} filas con VOLUMEN_1P_PREDICHO < 0"


def test_anclaje_vol_pref_igual_baseline(df_export):
    """
    Re-anclaje (2026-06-12): en la fila Brent=BRENT_REF (incluida EXACTA en la grilla),
    Vol = baseline para todo campo con baseline — salvo campos cuyo p_ref cae bajo el
    piso de abandono (alli la regla dura impone Vol=0).
    """
    brent_ref = df_export["BRENT_REF_USD_BBL"].dropna().iloc[0]
    sub = df_export[(df_export["BRENT_USD_BBL"] == brent_ref)
                    & df_export["VOLUMEN_1P_BASELINE_MBPE"].notna()]
    assert not sub.empty, f"No existe la fila Brent={brent_ref} en el export"
    piso = sub["PRECIO_NETO_EFECTIVO_USD_BBL"] < \
        sub["BREAKEVEN_OPERACIONAL_USD_BBL"].fillna(-np.inf)
    dif = (sub["VOLUMEN_1P_PREDICHO_MBPE"] - sub["VOLUMEN_1P_BASELINE_MBPE"]).abs()
    bad = sub[~piso & (dif > 0.02)]
    assert bad.empty, (
        f"{len(bad)} series no devuelven el baseline al precio actual "
        f"(Brent={brent_ref}): {bad['CAMPO'].unique()[:5].tolist()}")


def test_hard_zero_sub_abandono(df_export):
    """Piso duro del re-anclaje: Precio Aceite < BK abandono -> Vol = 0 exacto."""
    sub = df_export[df_export["BREAKEVEN_OPERACIONAL_USD_BBL"].notna()
                    & df_export["VOLUMEN_1P_BASELINE_MBPE"].notna()]
    bajo_piso = sub[sub["PRECIO_NETO_EFECTIVO_USD_BBL"]
                    < sub["BREAKEVEN_OPERACIONAL_USD_BBL"]]
    if bajo_piso.empty:
        pytest.skip("Ninguna fila bajo el piso de abandono en la grilla")
    no_cero = bajo_piso[bajo_piso["VOLUMEN_1P_PREDICHO_MBPE"] > 0]
    assert no_cero.empty, (
        f"{len(no_cero)} filas con vol>0 bajo el piso de abandono: "
        f"{no_cero['CAMPO'].unique()[:5].tolist()}")


def test_m2_fallback_visible(df_export):
    """Tag visible del fallback M2: METODO y boolean coherentes en el export."""
    fb = df_export[df_export["M2_ES_FALLBACK"]]
    no_fb = df_export[~df_export["M2_ES_FALLBACK"]]
    assert (fb["M2_METODO"] == "FALLBACK_BETA_PORTAFOLIO").all(), \
        "M2_ES_FALLBACK=True con METODO distinto de FALLBACK_BETA_PORTAFOLIO"
    assert (no_fb["M2_METODO"] != "FALLBACK_BETA_PORTAFOLIO").all(), \
        "FALLBACK_BETA_PORTAFOLIO sin M2_ES_FALLBACK=True"


def test_es_viable_usa_operacional(df_export):
    """
    ES_VIABLE = PRECIO_NETO >= BREAKEVEN_OPERACIONAL (piso inferior, abandono).
    El campo conserva alguna reserva (PDP) por encima de este piso.
    """
    sub = df_export[df_export["BREAKEVEN_OPERACIONAL_USD_BBL"].notna()]
    esperado = sub["PRECIO_NETO_EFECTIVO_USD_BBL"] >= sub["BREAKEVEN_OPERACIONAL_USD_BBL"]
    bad = sub[sub["ES_VIABLE"] != esperado]
    assert bad.empty, f"ES_VIABLE inconsistente con BREAKEVEN_OPERACIONAL en {len(bad)} filas"


def test_es_full_reservas_usa_financiero(df_export):
    """
    ES_FULL_RESERVAS = PRECIO_NETO >= BREAKEVEN_FINANCIERO (piso superior, delta=0).
    Escalera completa, sin castigo de PNP+PND.
    """
    sub = df_export[df_export["BREAKEVEN_FINANCIERO_USD_BBL"].notna()]
    esperado = sub["PRECIO_NETO_EFECTIVO_USD_BBL"] >= sub["BREAKEVEN_FINANCIERO_USD_BBL"]
    bad = sub[sub["ES_FULL_RESERVAS"] != esperado]
    assert bad.empty, f"ES_FULL_RESERVAS inconsistente con BREAKEVEN_FINANCIERO en {len(bad)} filas"


def test_es_full_reservas_implica_es_viable(df_export):
    """
    Escalera: ES_FULL_RESERVAS=True implica ES_VIABLE=True (piso financiero >= piso
    operacional, salvo ESCALERA_DEGENERADA donde son iguales).
    """
    sub = df_export[df_export["ES_FULL_RESERVAS"]]
    bad = sub[~sub["ES_VIABLE"]]
    assert bad.empty, f"{len(bad)} filas con ES_FULL_RESERVAS=True pero ES_VIABLE=False"


def test_es_extrapolado_existe(df_export):
    """ES_EXTRAPOLADO debe existir y tener valores True (hay extrapolacion en los extremos)."""
    assert "ES_EXTRAPOLADO" in df_export.columns
    n_extrap = df_export["ES_EXTRAPOLADO"].sum()
    assert n_extrap > 0, "ES_EXTRAPOLADO=True en 0 filas (deberia haber extrapolacion)"


def test_monotonia_export_por_campo_motor(df_export):
    """
    Vol predicho debe ser monotonico no-decreciente por Brent para cada CAMPO×MOTOR.
    (Modelos monotonos + recta M2 con β>0 + shift constante del re-anclaje + piso duro
    en el extremo bajo preservan la monotonia.)
    """
    for (campo, motor), sub in df_export.groupby(["CAMPO", "MOTOR"]):
        sub_ord = sub.sort_values("BRENT_USD_BBL")
        vols    = sub_ord["VOLUMEN_1P_PREDICHO_MBPE"].values
        decreasing = np.sum(np.diff(vols) < -0.5)  # tolerancia 0.5 MBPE por redondeo
        assert decreasing == 0, \
            f"{campo}/{motor}: {decreasing} tramos decrecientes en el export"


def test_reconstruccion_baseline_delta(df_export):
    """
    Vol predicho = max(0, baseline + delta_anclado), con piso duro vol=0 bajo el
    BK de abandono (alli baseline+delta puede ser >0 pero la regla fisica manda).
    """
    sub = df_export[df_export["VOLUMEN_1P_BASELINE_MBPE"].notna()].copy()
    if sub.empty:
        pytest.skip("No hay filas con BASELINE en export")
    reconstruido = np.maximum(sub["VOLUMEN_1P_BASELINE_MBPE"] + sub["DELTA_PRED_MBPE"], 0)
    piso = sub["PRECIO_NETO_EFECTIVO_USD_BBL"] < \
        sub["BREAKEVEN_OPERACIONAL_USD_BBL"].fillna(-np.inf)
    reconstruido = np.where(piso, 0.0, reconstruido)
    diff = (reconstruido - sub["VOLUMEN_1P_PREDICHO_MBPE"]).abs()
    assert (diff < 0.1).all(), \
        f"Reconstruccion baseline+delta no coincide en {(diff >= 0.1).sum()} filas"


def test_brent_max_vol_positivo(df_export):
    """
    En el extremo superior de la grilla de Brent, campos con datos reales de sensibilidad
    (N_REAL_DELTA > 0 en metricas.csv) y ES_VIABLE=True deben predecir vol > 0.
    Campos entrenados solo con sintéticos predicen correctamente vol=0 fuera
    de su rango de entrenamiento — no se validan aquí.
    """
    met_path = STAGING / "metricas.csv"
    if not met_path.exists():
        pytest.skip("metricas.csv no existe")

    metricas = pd.read_csv(met_path)
    # Campos con al menos 1 punto real de sensibilidad
    campos_con_real = set(metricas.loc[metricas["N_REAL_DELTA"].fillna(0) > 0, "CAMPO"])

    brent_hi = df_export["BRENT_USD_BBL"].max()
    sub_hi = df_export[df_export["BRENT_USD_BBL"] == brent_hi].copy()

    viables = sub_hi[
        sub_hi["CAMPO"].isin(campos_con_real) &
        sub_hi["ES_VIABLE"] &
        (sub_hi["VOLUMEN_1P_BASELINE_MBPE"].fillna(0) >= 0.5)  # excluir micro-campos (<0.5 MBPE)
    ]
    if viables.empty:
        pytest.skip(f"Sin campos con datos reales y baseline significativo viables a Brent=${brent_hi}")
    cero = (viables["VOLUMEN_1P_PREDICHO_MBPE"] <= 0).sum()
    assert cero == 0, \
        f"{cero} campo/motor (datos reales, baseline>=0.5 MBPE, ES_VIABLE) con vol=0 en Brent=${brent_hi}"


def test_nivel_confianza_valido(df_export):
    """
    NIVEL_CONFIANZA debe existir, sin NaN, y solo contener los 4 valores validos.
    Campos con N_REAL_DELTA==0 en metricas.csv deben clasificarse SOLO_SINTETICO.
    """
    assert "NIVEL_CONFIANZA" in df_export.columns, "NIVEL_CONFIANZA faltante en export"
    nulos = df_export["NIVEL_CONFIANZA"].isnull().sum()
    assert nulos == 0, f"NIVEL_CONFIANZA: {nulos} NaN en export"

    valores_validos = {"ALTA", "MEDIA", "BAJA", "SOLO_SINTETICO"}
    invalidos = set(df_export["NIVEL_CONFIANZA"].unique()) - valores_validos
    assert not invalidos, f"Valores invalidos en NIVEL_CONFIANZA: {invalidos}"

    # Campos SOLO_SINTETICO deben coincidir con N_REAL_DELTA==0 en metricas.csv
    met_path = STAGING / "metricas.csv"
    if not met_path.exists():
        pytest.skip("metricas.csv no existe")
    met = pd.read_csv(met_path)
    sin_real = set(met.loc[met["N_REAL_DELTA"].fillna(0) == 0, "CAMPO"])

    solo_sint_export = set(
        df_export.loc[df_export["NIVEL_CONFIANZA"] == "SOLO_SINTETICO", "CAMPO"].unique()
    )
    # Todos los SOLO_SINTETICO del export deben tener N_REAL==0 en metricas
    falsos = solo_sint_export - sin_real
    assert not falsos, f"SOLO_SINTETICO en export pero N_REAL>0 en metricas: {falsos}"


def test_gate_skill_en_confianza():
    """Gate de skill (2026-06-11): ALTA exige SKILL>0 salvo campo plano
    (MAE_NAIVE < 2 MBPE, donde la media ingenua es imbatible por construccion)."""
    import importlib
    mod = importlib.import_module("04_pbi_export")
    base = dict(n_real=8, mae_rel=0.10, divergencia=0.10, baseline=100.0, mae_abs=10.0)
    # Sin skill + naive grande → MEDIA (caso AKACIAS)
    nivel, motivo = mod.clasificar_confianza(**base, skill=-0.01, mae_naive=14.0)
    assert nivel == "MEDIA" and "sin skill" in motivo
    # Con skill → ALTA
    nivel, _ = mod.clasificar_confianza(**base, skill=0.5, mae_naive=14.0)
    assert nivel == "ALTA"
    # Sin skill pero campo plano (naive pequeño) → exento, ALTA (caso CAÑO SUR ESTE)
    nivel, _ = mod.clasificar_confianza(**base, skill=-0.4, mae_naive=1.2)
    assert nivel == "ALTA"
    # SKILL NaN no penaliza
    nivel, _ = mod.clasificar_confianza(**base)
    assert nivel == "ALTA"


def test_cap_outlier_por_materialidad():
    """Cap de outlier (2026-06-11): OUTLIER_LOO solo degrada si MAE_REL >= 5%.
    Caso RUBIALES: ratio disparado por un fold en la frontera de vigencias con
    error relativo 0.9% — inmaterial, no debe bajar de ALTA."""
    import importlib
    mod = importlib.import_module("04_pbi_export")
    base = dict(n_real=8, divergencia=0.10, baseline=323.0, skill=0.8, mae_naive=15.0)
    # Outlier + error inmaterial (<5%) → ALTA, motivo con OUTLIER_LOO_INMATERIAL
    nivel, motivo = mod.clasificar_confianza(**base, mae_rel=0.009, mae_abs=2.9,
                                             outlier_lloo=True)
    assert nivel == "ALTA" and "OUTLIER_LOO_INMATERIAL" in motivo
    # Outlier + error material (>=5%) → cap MEDIA (comportamiento original)
    nivel, motivo = mod.clasificar_confianza(**base, mae_rel=0.15, mae_abs=48.0,
                                             outlier_lloo=True)
    assert nivel == "MEDIA" and "OUTLIER_LOO" in motivo \
        and "INMATERIAL" not in motivo


def test_comparacion_vs_anterior_existe():
    """comparacion_vs_anterior.csv debe existir con las columnas de versionamiento (Opp #4)."""
    ruta = RESULTADOS / "comparacion_vs_anterior.csv"
    assert ruta.exists(), "resultados/comparacion_vs_anterior.csv no existe"
    cols = pd.read_csv(ruta, nrows=0).columns.tolist()
    esperadas = ["CAMPO", "MOTOR", "Q_OBJETIVO_ANTERIOR", "Q_OBJETIVO_NUEVO",
                  "VOL_ANTERIOR_MBPE", "VOL_NUEVO_MBPE", "DIF_ABS_MBPE", "DIF_PCT",
                  "NIVEL_CONFIANZA_ANTERIOR", "NIVEL_CONFIANZA_NUEVO"]
    faltantes = [c for c in esperadas if c not in cols]
    assert not faltantes, f"Columnas faltantes en comparacion_vs_anterior.csv: {faltantes}"


def test_changelog_predicciones_existe():
    """docs/CHANGELOG_PREDICCIONES.md debe existir tras correr 04_pbi_export.py."""
    ruta = ROOT / "docs" / "CHANGELOG_PREDICCIONES.md"
    assert ruta.exists(), "docs/CHANGELOG_PREDICCIONES.md no existe"


def test_export_sin_punto_y_coma(df_export):
    """
    Ninguna celda del export puede contener ';'. Excel en locale es-CO interpreta
    ';' como delimitador de columnas y, al abrir el CSV (separado por comas),
    parte celdas como MOTIVO_CONFIANZA en columnas espurias y pierde datos
    (ver MAESTRO §10, fix 2026-06-11: separador interno cambiado a '|').
    """
    for col in df_export.columns:
        con_punto_coma = df_export[col].astype(str).str.contains(";", na=False)
        n = int(con_punto_coma.sum())
        assert n == 0, (
            f"Columna '{col}': {n} celdas con ';' (rompe Excel es-CO). "
            f"Ej: {df_export.loc[con_punto_coma, col].iloc[0]!r}")
