"""
test_01_etl.py — Gate Fase 1: verifica invariantes del tablon_unico producido por 01_etl.py

Todos los asserts son duros. Fallo de cualquiera bloquea el pipeline.
Requiere que 01_etl.py haya corrido antes (tablon_unico.parquet existe).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

STAGING = ROOT / "datos" / "staging"

# Gate dorado: estos 4 campos son el baseline de validación (siempre deben estar)
CAMPOS_GATE = {"CASTILLA", "CASTILLA NORTE", "CASTILLA ESTE", "RUBIALES"}
# Compatibilidad hacia atrás (algunos tests usan este nombre)
CAMPOS_PILOTO = CAMPOS_GATE


@pytest.fixture(scope="module")
def tablon():
    ruta = STAGING / "tablon_unico.parquet"
    assert ruta.exists(), "tablon_unico.parquet no existe: correr 01_etl.py primero"
    df = pd.read_parquet(ruta)
    return df[~df["ES_SINTETICO"]].copy()  # solo reales en el gate del ETL


def test_parquet_existe():
    assert (STAGING / "tablon_unico.parquet").exists()
    assert (STAGING / "tablon_unico.csv").exists()


def test_campos_piloto_presentes(tablon):
    """Gate dorado: los 4 campos de referencia deben estar en el tablon."""
    campos = set(tablon["CAMPO"].unique())
    faltantes = CAMPOS_GATE - campos
    assert not faltantes, f"Campos faltantes: {faltantes}"


def test_fase1_cobertura_minima(tablon):
    """Fase 1: el tablon debe tener >= 50 campos con datos de sensibilidad (DELTA no-nulo)."""
    campos_con_delta = tablon.loc[tablon["DELTA_SENS_MBPE"].notna(), "CAMPO"].nunique()
    assert campos_con_delta >= 50, \
        f"Solo {campos_con_delta} campos con DELTA en Fase 1 (esperado >= 50)"


def test_fase1_consolidado_filas(tablon):
    """Fase 1: deben existir filas CONSOLIDADO para >= 50 campos distintos."""
    campos_cons = tablon.loc[
        tablon["ESCENARIO"].str.startswith("CONSOLIDADO", na=False), "CAMPO"
    ].nunique()
    assert campos_cons >= 50, \
        f"Solo {campos_cons} campos con CONSOLIDADO en Fase 1 (esperado >= 50)"


def test_brent_flat_sin_nulos(tablon):
    """
    BRENT_FLAT sin nulos para campos que tienen sensibilidad (CONSOLIDADO).
    Las filas BASE de campos sin 'Precio Net' son esperadas y no bloquean el modelo.
    Gate dorado: los 4 campos de referencia deben tener BRENT_FLAT en sus filas BASE.
    """
    # Campos que aparecen en CONSOLIDADO (tienen datos de sensibilidad → se entrenan)
    campos_cons = set(tablon.loc[
        tablon["ESCENARIO"].str.startswith("CONSOLIDADO", na=False), "CAMPO"
    ].unique())
    base_cons = tablon[
        (tablon["ESCENARIO"] == "BASE") & tablon["CAMPO"].isin(campos_cons)
    ]
    if base_cons.empty:
        pytest.skip("Sin filas BASE para campos con CONSOLIDADO")
    nulos = base_cons["BRENT_FLAT_USD_BBL"].isnull().sum()
    # Permitir hasta 20% de nulos: campos en Consolidado sin hoja "Precio Net" en HIST 1P
    # (filiales, campos de gas, campos muy recientes). Gate dorado verifica los 4 de referencia.
    assert nulos / len(base_cons) < 0.20, \
        f"BRENT_FLAT nulos en filas BASE de campos con CONSOLIDADO: {nulos}/{len(base_cons)}"
    # Gate dorado: los campos de referencia no deben tener nulos
    for campo in CAMPOS_GATE:
        sub = base_cons[base_cons["CAMPO"] == campo]
        if sub.empty:
            continue
        n = sub["BRENT_FLAT_USD_BBL"].isnull().sum()
        assert n == 0, f"{campo}: BRENT_FLAT nulos en filas BASE: {n}"


def test_descuentos_signo_negativo(tablon):
    """
    B2: todos los descuentos en el tablon deben ser <= 0.
    Convención única: Brent = PrecioNeto − Calidad − Transporte (descuentos negativos).
    """
    cal_pos = (tablon["DESCUENTO_CALIDAD_USD_BBL"].dropna() > 0).sum()
    tra_pos = (tablon["DESCUENTO_TRANSPORTE_USD_BBL"].dropna() > 0).sum()
    assert cal_pos == 0, f"DESCUENTO_CALIDAD positivos: {cal_pos} filas"
    assert tra_pos == 0, f"DESCUENTO_TRANSPORTE positivos: {tra_pos} filas"


def test_vigencia_poblada_100pct(tablon):
    """VIGENCIA debe estar poblada en todas las filas."""
    nulos = (tablon["VIGENCIA"].isna() | (tablon["VIGENCIA"] == "")).sum()
    assert nulos == 0, f"VIGENCIA nula en {nulos} filas"


def test_vigencia_base_es_año(tablon):
    """Filas BASE: VIGENCIA debe ser str del AÑO (ej. '2024')."""
    base = tablon[tablon["ESCENARIO"] == "BASE"].copy()
    base["VIGENCIA_ESPERADA"] = base["AÑO"].apply(lambda x: str(int(x)) if pd.notna(x) else "")
    coincide = (base["VIGENCIA"] == base["VIGENCIA_ESPERADA"]).all()
    assert coincide, "VIGENCIA en filas BASE no coincide con str(AÑO)"


def test_vigencia_consolidado_es_quarter(tablon):
    """Filas CONSOLIDADO: VIGENCIA debe ser del formato YYYY_Qq (ej. '2024_Q2')."""
    cons = tablon[tablon["ESCENARIO"].str.startswith("CONSOLIDADO", na=False)]
    if cons.empty:
        pytest.skip("No hay filas CONSOLIDADO")
    patron_ok = cons["VIGENCIA"].str.match(r"^\d{4}_Q\d$", na=False)
    assert patron_ok.all(), f"VIGENCIA malformada en {(~patron_ok).sum()} filas CONSOLIDADO"


def test_breakeven_presente(tablon):
    """BREAKEVEN_FINANCIERO debe estar presente para todos los campos piloto."""
    for campo in CAMPOS_PILOTO:
        sub = tablon[tablon["CAMPO"] == campo]
        if sub.empty:
            continue
        bk = sub["BREAKEVEN_FINANCIERO_USD_BBL"].dropna()
        assert len(bk) > 0, f"{campo}: sin BREAKEVEN_FINANCIERO_USD_BBL"


def test_castilla_este_breakeven_fallback_flaggeado(tablon):
    """B7: CASTILLA ESTE debe tener ALERTA=BREAKEVEN_FALLBACK (solo D-PDP en Breakeven.xlsm)."""
    sub = tablon[tablon["CAMPO"] == "CASTILLA ESTE"]
    if sub.empty:
        pytest.skip("CASTILLA ESTE no en tablon")
    tiene_flag = sub["ALERTA"].str.contains("BREAKEVEN_FALLBACK", na=False).any()
    assert tiene_flag, "CASTILLA ESTE no tiene ALERTA=BREAKEVEN_FALLBACK"


def test_target_nulo_flaggeado(tablon):
    """B6: Filas CONSOLIDADO sin reservas deben tener ALERTA=TARGET_NULO, no dropearse."""
    cons = tablon[tablon["ESCENARIO"].str.startswith("CONSOLIDADO", na=False)]
    if cons.empty:
        pytest.skip("No hay filas CONSOLIDADO")
    # Debe haber al menos alguna fila con TARGET_NULO si hay quarters sin datos de reservas
    # (6 de 9 quarters no tienen datos en el piloto)
    nulo_sens = cons["VOLUMEN_1P_SENSIBILIDAD_MBPE"].isna()
    if nulo_sens.any():
        flaggeados = cons.loc[nulo_sens, "ALERTA"].str.contains("TARGET_NULO", na=False).all()
        assert flaggeados, "Filas CONSOLIDADO sin SENSIBILIDAD no tienen ALERTA=TARGET_NULO"


def test_dif_neto_salto_flaggeado(tablon):
    """Alerta calidad de datos (2026-06-11): salto > 5 USD en el diferencial
    neto−brent entre quarters consecutivos → ALERTA=DIF_NETO_SALTO.
    Caso conocido: RUBIALES 2025_Q4 (dif pasa de −8 a +0.03)."""
    rub = tablon[(tablon["CAMPO"] == "RUBIALES") &
                 (tablon["ESCENARIO"] == "CONSOLIDADO_2025_Q4")]
    if rub.empty:
        pytest.skip("RUBIALES 2025_Q4 no en tablon")
    assert rub["ALERTA"].str.contains("DIF_NETO_SALTO", na=False).any(), \
        "RUBIALES 2025_Q4 sin ALERTA=DIF_NETO_SALTO (salto de diferencial conocido)"


def test_no_homologados_son_cero(tablon):
    """No debe haber filas NO_HOMOLOGADO en campos piloto."""
    no_hom = tablon[tablon["ALERTA"].str.contains("NO_HOMOLOGADO", na=False)]
    pilotos_no_hom = no_hom[no_hom["CAMPO"].isin(CAMPOS_PILOTO)]
    assert len(pilotos_no_hom) == 0, \
        f"Campos piloto con NO_HOMOLOGADO: {pilotos_no_hom['CAMPO'].unique()}"


def test_es_baseline_true_en_base(tablon):
    """Todas las filas ESCENARIO=BASE deben tener ES_BASELINE=True."""
    base = tablon[tablon["ESCENARIO"] == "BASE"]
    assert base["ES_BASELINE"].all(), "Hay filas BASE con ES_BASELINE=False"


def test_es_baseline_false_en_consolidado(tablon):
    """Todas las filas CONSOLIDADO deben tener ES_BASELINE=False."""
    cons = tablon[tablon["ESCENARIO"].str.startswith("CONSOLIDADO", na=False)]
    if cons.empty:
        pytest.skip("No hay filas CONSOLIDADO")
    assert (~cons["ES_BASELINE"]).all(), "Hay filas CONSOLIDADO con ES_BASELINE=True"


def test_baseline_1p_vigencia_presente_en_base(tablon):
    """Filas BASE con VOLUMEN_1P_OFICIAL deben tener BASELINE_1P_VIGENCIA no nulo."""
    base = tablon[(tablon["ESCENARIO"] == "BASE") & tablon["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
    nulos = base["BASELINE_1P_VIGENCIA_MBPE"].isnull().sum()
    assert nulos == 0, f"BASELINE_1P_VIGENCIA nulo en {nulos} filas BASE con OFICIAL"


def test_delta_nulo_en_base(tablon):
    """Filas BASE deben tener DELTA_SENS_MBPE = NaN (no hay punto de sensibilidad)."""
    base = tablon[tablon["ESCENARIO"] == "BASE"]
    non_nan = base["DELTA_SENS_MBPE"].notna().sum()
    assert non_nan == 0, f"DELTA_SENS_MBPE no-nulo en {non_nan} filas BASE"


def test_delta_definido_cuando_hay_sensibilidad(tablon):
    """Filas CONSOLIDADO con SENSIBILIDAD y BASELINE ambos disponibles → DELTA definido."""
    cons = tablon[tablon["ESCENARIO"].str.startswith("CONSOLIDADO", na=False)]
    con_ambos = cons[
        cons["VOLUMEN_1P_SENSIBILIDAD_MBPE"].notna() &
        cons["BASELINE_1P_VIGENCIA_MBPE"].notna()
    ]
    if con_ambos.empty:
        pytest.skip("No hay filas CONSOLIDADO con SENSIBILIDAD y BASELINE")
    nulos_delta = con_ambos["DELTA_SENS_MBPE"].isnull().sum()
    assert nulos_delta == 0, \
        f"DELTA_SENS nulo en {nulos_delta} filas con SENSIBILIDAD y BASELINE disponibles"


def test_stale_breakeven_flaggeado(tablon):
    """STALE: filas donde AÑO > VIGENCIA_BREAKEVEN disponible deben llevar ALERTA.
    Con motor Python 2024+2025, AÑO=2025 ya tiene BK propio → no es stale.
    Solo AÑO=2026+ (sin Cierre-2026 aún) serán stale."""
    stale_rows = tablon[
        tablon["ESCENARIO"].str.startswith("CONSOLIDADO", na=False) &
        tablon["ALERTA"].str.contains("BREAKEVEN_VIGENCIA_STALE", na=False)
    ]
    no_stale_rows = tablon[
        tablon["ESCENARIO"].str.startswith("CONSOLIDADO", na=False) &
        ~tablon["ALERTA"].str.contains("BREAKEVEN_VIGENCIA_STALE", na=False)
    ]
    # Invariante: filas STALE → AÑO > VIGENCIA_BREAKEVEN
    if not stale_rows.empty:
        bad = stale_rows[
            stale_rows["AÑO"].fillna(0).astype(int).astype(str)
            <= stale_rows["VIGENCIA_BREAKEVEN"].fillna("0")
        ]
        assert bad.empty, f"STALE marcado pero AÑO <= VIG_BK en: {bad[['CAMPO','AÑO','VIGENCIA_BREAKEVEN']].to_dict('records')}"
    # Invariante inverso: filas no-STALE → AÑO <= VIGENCIA_BREAKEVEN
    if not no_stale_rows.empty:
        bad2 = no_stale_rows[
            no_stale_rows["AÑO"].fillna(0).astype(int).astype(str)
            > no_stale_rows["VIGENCIA_BREAKEVEN"].fillna("0")
        ]
        assert bad2.empty, f"Sin STALE pero AÑO > VIG_BK en: {bad2[['CAMPO','AÑO','VIGENCIA_BREAKEVEN']].to_dict('records')}"


def test_columnas_obligatorias(tablon):
    """El tablon debe tener todas las columnas del esquema canonico (§6 MAESTRO)."""
    obligatorias = [
        "CAMPO", "AÑO", "VIGENCIA", "ESCENARIO", "ES_BASELINE", "ES_SINTETICO",
        "NIVEL_DEFINICIONAL", "BRENT_FLAT_USD_BBL", "DESCUENTO_CALIDAD_USD_BBL",
        "DESCUENTO_TRANSPORTE_USD_BBL", "PRECIO_NETO_USD_BBL",
        "VOLUMEN_1P_OFICIAL_MBPE", "BASELINE_1P_VIGENCIA_MBPE",
        "VOLUMEN_1P_SENSIBILIDAD_MBPE", "DELTA_SENS_MBPE",
        "BREAKEVEN_FINANCIERO_USD_BBL", "BREAKEVEN_OPERACIONAL_USD_BBL",
        "BK_ANCLA_FIN_USD_BBL", "BK_ANCLA_PDP_USD_BBL",
        "VIGENCIA_BREAKEVEN", "ALERTA",
    ]
    faltantes = [c for c in obligatorias if c not in tablon.columns]
    assert not faltantes, f"Columnas faltantes en tablon: {faltantes}"


def test_breakeven_swap_financiero_mayor_operacional(tablon):
    """
    Convencion finanzas (2026-06-09): BREAKEVEN_FINANCIERO/BK_ANCLA_FIN = piso
    SUPERIOR (delta=0) > BREAKEVEN_OPERACIONAL/BK_ANCLA_PDP = piso INFERIOR
    (abandono) para los campos gate dorado (ninguno tiene ESCALERA_DEGENERADA).
    """
    for campo in CAMPOS_GATE:
        sub = tablon[tablon["CAMPO"] == campo]
        if sub.empty:
            continue
        bk_fin = sub["BK_ANCLA_FIN_USD_BBL"].dropna()
        bk_pdp = sub["BK_ANCLA_PDP_USD_BBL"].dropna()
        if bk_fin.empty or bk_pdp.empty:
            continue
        assert bk_fin.iloc[0] > bk_pdp.iloc[0], \
            f"{campo}: BK_ANCLA_FIN ({bk_fin.iloc[0]}) <= BK_ANCLA_PDP ({bk_pdp.iloc[0]})"


def test_escalera_degenerada_colapsa_pisos(tablon):
    """Campos con ALERTA=ESCALERA_DEGENERADA deben tener BK_ANCLA_FIN == BK_ANCLA_PDP."""
    deg = tablon[tablon["ALERTA"].str.contains("ESCALERA_DEGENERADA", na=False)]
    if deg.empty:
        pytest.skip("Ningun campo con ESCALERA_DEGENERADA en esta corrida")
    for campo, sub in deg.groupby("CAMPO"):
        bk_fin = sub["BK_ANCLA_FIN_USD_BBL"].dropna()
        bk_pdp = sub["BK_ANCLA_PDP_USD_BBL"].dropna()
        if bk_fin.empty or bk_pdp.empty:
            continue
        assert bk_fin.iloc[0] == pytest.approx(bk_pdp.iloc[0]), \
            f"{campo}: ESCALERA_DEGENERADA pero BK_ANCLA_FIN != BK_ANCLA_PDP"
