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

# Paths por track (Produccion vs Calidad): centralizados en tests/conftest.py
from rutas_track import STAGING  # noqa: E402

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
    """BREAKEVEN (piso superior) debe estar presente para todos los campos piloto."""
    for campo in CAMPOS_PILOTO:
        sub = tablon[tablon["CAMPO"] == campo]
        if sub.empty:
            continue
        bk = sub["BREAKEVEN_USD_BBL"].dropna()
        assert len(bk) > 0, f"{campo}: sin BREAKEVEN_USD_BBL"


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


def test_checkpoint_presente_en_base(tablon):
    """Filas BASE con VOLUMEN_1P_OFICIAL deben tener CHECKPOINT (cierre A) no nulo."""
    base = tablon[(tablon["ESCENARIO"] == "BASE") & tablon["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
    nulos = base["CHECKPOINT_1P_MBPE"].isnull().sum()
    assert nulos == 0, f"CHECKPOINT_1P nulo en {nulos} filas BASE con OFICIAL"


def test_delta_nulo_en_base(tablon):
    """Filas BASE deben tener DELTA_SENS_MBPE = NaN (no hay punto de sensibilidad)."""
    base = tablon[tablon["ESCENARIO"] == "BASE"]
    non_nan = base["DELTA_SENS_MBPE"].notna().sum()
    assert non_nan == 0, f"DELTA_SENS_MBPE no-nulo en {non_nan} filas BASE"


def test_delta_definido_cuando_hay_sensibilidad(tablon):
    """Filas CONSOLIDADO con SENSIBILIDAD y BASELINE (cierre A−1) → DELTA definido."""
    cons = tablon[tablon["ESCENARIO"].str.startswith("CONSOLIDADO", na=False)]
    con_ambos = cons[
        cons["VOLUMEN_1P_SENSIBILIDAD_MBPE"].notna() &
        cons["BASELINE_1P_MBPE"].notna()
    ]
    if con_ambos.empty:
        pytest.skip("No hay filas CONSOLIDADO con SENSIBILIDAD y BASELINE")
    nulos_delta = con_ambos["DELTA_SENS_MBPE"].isnull().sum()
    assert nulos_delta == 0, \
        f"DELTA_SENS nulo en {nulos_delta} filas con SENSIBILIDAD y BASELINE disponibles"


def test_delta_usa_cierre_anterior(tablon):
    """Normalización v2 (2026-07-09): DELTA_SENS = VOL_SENS − cierre A−1, no cierre A.
    CASTILLA 2024_Q1: 174.6 − 175.249 (cierre 2023) ≈ −0.65 MBPE. Con la semántica
    vieja (cierre 2024 = 158.647) daría +15.95 — el confound."""
    fila = tablon[(tablon["CAMPO"] == "CASTILLA")
                  & (tablon["VIGENCIA"] == "2024_Q1")]
    if fila.empty:
        pytest.skip("CASTILLA 2024_Q1 no está en el tablón")
    r = fila.iloc[0]
    esperado = r["VOLUMEN_1P_SENSIBILIDAD_MBPE"] - r["BASELINE_1P_MBPE"]
    assert abs(r["DELTA_SENS_MBPE"] - esperado) < 1e-6
    assert r["DELTA_SENS_MBPE"] < 0, \
        f"DELTA 2024_Q1 debería ser ≈−0.65 (vs cierre 2023); da {r['DELTA_SENS_MBPE']:.2f}"
    assert abs(r["BASELINE_1P_MBPE"] - 175.249) < 0.01, \
        f"BASELINE de vigencia 2024 debe ser el cierre 2023 (175.249), da {r['BASELINE_1P_MBPE']}"
    assert abs(r["CHECKPOINT_1P_MBPE"] - 158.647) < 0.01, \
        f"CHECKPOINT de vigencia 2024 debe ser el cierre 2024 (158.647), da {r['CHECKPOINT_1P_MBPE']}"


def test_sin_baseline_anterior_flaggeado(tablon):
    """Quarters con sensibilidad pero sin cierre A−1 → ALERTA=SIN_BASELINE_ANTERIOR
    y DELTA NaN (no entrenan, nunca descartados en silencio)."""
    cons = tablon[tablon["ESCENARIO"].str.startswith("CONSOLIDADO", na=False)]
    sin_base = cons[cons["VOLUMEN_1P_SENSIBILIDAD_MBPE"].notna()
                    & cons["BASELINE_1P_MBPE"].isna()]
    if sin_base.empty:
        pytest.skip("Todos los quarters con sensibilidad tienen cierre A−1")
    assert sin_base["DELTA_SENS_MBPE"].isna().all()
    sin_flag = sin_base[~sin_base["ALERTA"].str.contains("SIN_BASELINE_ANTERIOR", na=False)]
    assert sin_flag.empty, \
        f"{len(sin_flag)} filas sin cierre A−1 no llevan ALERTA=SIN_BASELINE_ANTERIOR"


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
        "VOLUMEN_1P_OFICIAL_MBPE", "CHECKPOINT_1P_MBPE", "BASELINE_1P_MBPE",
        "VOLUMEN_1P_SENSIBILIDAD_MBPE", "DELTA_SENS_MBPE",
        "BREAKEVEN_USD_BBL", "PRECIO_EQUILIBRIO_USD_BBL",
        "BK_ANCLA_FIN_USD_BBL", "BK_ANCLA_PDP_USD_BBL", "BK_ANCLA_CLASE",
        "VIGENCIA_BREAKEVEN", "ALERTA",
    ]
    faltantes = [c for c in obligatorias if c not in tablon.columns]
    assert not faltantes, f"Columnas faltantes en tablon: {faltantes}"


def test_breakeven_swap_financiero_mayor_operacional(tablon):
    """
    Convencion equipo (2026-06-09; renombre 2026-07-10): BREAKEVEN/BK_ANCLA_FIN = piso
    SUPERIOR (delta=0; L.E.) > PRECIO_EQUILIBRIO/BK_ANCLA_PDP = piso INFERIOR
    (VPN=0; abandono) para los campos gate dorado (ninguno tiene ESCALERA_DEGENERADA).
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


def test_bk_ancla_clase_mayor_incertidumbre(tablon):
    """
    BK_ANCLA_FIN (2026-06-11) usa el criterio de clase de mayor incertidumbre
    (PND>PNP>PDP). El valor anclado debe ser EXACTAMENTE el BK financiero por clase
    de la clase declarada en BK_ANCLA_CLASE (no un promedio ponderado). Verificamos
    la coherencia para el gate dorado, cuyas anclas vienen de PND.
    """
    u = tablon.drop_duplicates(["CAMPO", "VIGENCIA_BREAKEVEN"])
    # Todo registro con ancla FIN definida debe declarar la clase que la fijo
    con_ancla = u[u["BK_ANCLA_FIN_USD_BBL"].notna()]
    sin_clase = con_ancla[con_ancla["BK_ANCLA_CLASE"].isna()]
    assert sin_clase.empty, \
        f"Anclas FIN sin BK_ANCLA_CLASE: {sin_clase['CAMPO'].tolist()}"
    # Las clases validas son las tres reservas o el fallback
    validas = {"D-PND", "D-PNP", "D-PDP", "FALLBACK_GLOBAL"}
    invalidas = set(con_ancla["BK_ANCLA_CLASE"].unique()) - validas
    assert not invalidas, f"BK_ANCLA_CLASE con valores invalidos: {invalidas}"
    # Gate dorado: el ancla debe coincidir con la salida por clase declarada
    # (PND -> BK_SALIDA_PND, PNP -> BK_SALIDA_PNP)
    salida_col = {"D-PND": "BK_SALIDA_PND_USD_BBL", "D-PNP": "BK_SALIDA_PNP_USD_BBL"}
    for campo in CAMPOS_GATE:
        sub = u[u["CAMPO"] == campo]
        for _, fila in sub.iterrows():
            clase = fila["BK_ANCLA_CLASE"]
            col = salida_col.get(clase)
            if col is None or pd.isna(fila[col]):
                continue
            assert fila["BK_ANCLA_FIN_USD_BBL"] == pytest.approx(fila[col]), \
                f"{campo}: BK_ANCLA_FIN != BK_SALIDA de la clase {clase}"


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
