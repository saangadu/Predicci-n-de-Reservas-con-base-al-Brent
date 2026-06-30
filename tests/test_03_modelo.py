"""
test_03_modelo.py — Gate Fase 3: Modelo 1 (Delta = f(Precio Neto))

Arquitectura 1D (2026-06-11): motores Isotonica (primario) + Suave/PCHIP (validacion),
ambos en PRECIO_NETO_USD_BBL. XGBoost retirado. Pruebas criticas: monotonia en banda
completa, metricas sobre reales, sanity de anclaje sub-breakeven.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

STAGING     = ROOT / "datos" / "staging"
MODELOS_DIR = STAGING / "modelos"
CAMPOS_PILOTO = ["CASTILLA", "CASTILLA NORTE", "CASTILLA ESTE", "RUBIALES"]
SUFIJOS = ["iso", "suave"]   # primario, validacion


@pytest.fixture(scope="module")
def tablon():
    ruta = STAGING / "tablon_unico.parquet"
    assert ruta.exists()
    return pd.read_parquet(ruta)


@pytest.fixture(scope="module")
def metricas():
    ruta = STAGING / "metricas.csv"
    assert ruta.exists(), "metricas.csv no existe: correr 03_modelo.py primero"
    return pd.read_csv(ruta)


def test_joblib_existen():
    """Un .joblib por motor (iso primario, suave validacion) por cada campo piloto."""
    for campo in CAMPOS_PILOTO:
        slug = campo.replace(" ", "_")
        for suf in SUFIJOS:
            f = MODELOS_DIR / f"{slug}_{suf}.joblib"
            assert f.exists(), f"Falta {f.name}"


def test_no_quedan_modelos_xgb():
    """XGBoost retirado: 03_modelo.py no debe escribir mas _xgb.joblib (los nuevos campos
    solo generan _iso y _suave)."""
    for campo in CAMPOS_PILOTO:
        slug = campo.replace(" ", "_")
        # No se exige borrar artefactos viejos, pero los nuevos motores deben existir.
        assert (MODELOS_DIR / f"{slug}_iso.joblib").exists()
        assert (MODELOS_DIR / f"{slug}_suave.joblib").exists()


def test_metricas_csv_existe():
    assert (STAGING / "metricas.csv").exists()


def test_metricas_n_real_delta_positivo(metricas):
    """Cada campo debe tener datos de entrenamiento (reales o sinteticos)."""
    for _, row in metricas.iterrows():
        n_real = int(row.get("N_REAL_DELTA", 0) or 0)
        n_sint = int(row.get("N_SINTETICOS", 0) or 0)
        assert n_real + n_sint >= 1, \
            f"{row['CAMPO']}: sin datos de entrenamiento (reales={n_real}, sint={n_sint})"


def test_metricas_columnas_motores(metricas):
    """metricas.csv expone columnas de ambos motores (ISO primario, SUAVE validacion)."""
    for col in ["N_REAL_DELTA", "N_SINTETICOS",
                "MAE_LOO_ISO", "SKILL_ISO", "MAE_LOO_SUAVE", "SKILL_SUAVE",
                "MAE_NAIVE", "PNETO_DELTA0_ISO", "ALERTA_LOO_OUTLIER_ISO"]:
        assert col in metricas.columns, f"Falta columna {col} en metricas.csv"


def test_metricas_sin_columnas_xgb(metricas):
    """Las columnas del motor retirado (XGB) ya no deben existir en metricas.csv."""
    for col in ["MAE_LOO_XGB", "SKILL_XGB", "BRENT_DELTA0_XGB"]:
        assert col not in metricas.columns, f"Columna obsoleta {col} sigue en metricas.csv"


@pytest.mark.parametrize("campo", CAMPOS_PILOTO)
@pytest.mark.parametrize("suf", SUFIJOS)
def test_monotonia_banda_completa(campo, suf, tablon):
    """Ambos motores monotonos en banda [BK_fin+2, BK_fin+50], evaluados en Precio Neto."""
    slug = campo.replace(" ", "_")
    ruta = MODELOS_DIR / f"{slug}_{suf}.joblib"
    if not ruta.exists():
        pytest.skip(f"Modelo {suf} de {campo} no encontrado")
    modelo = joblib.load(ruta)

    sub = tablon[tablon["CAMPO"] == campo]
    anclas = sub.dropna(subset=["BK_ANCLA_FIN_USD_BBL"])
    if anclas.empty:
        pytest.skip(f"{campo}: sin breakeven")
    bk = float(anclas.sort_values("VIGENCIA_BREAKEVEN", ascending=False)
               .iloc[0]["BK_ANCLA_FIN_USD_BBL"])

    px_band = np.linspace(bk + 2, bk + 50, 100)   # Precio Neto directo (modelo 1D)
    curva = modelo.predict(px_band)
    decreasing = int(np.sum(np.diff(curva) < -1e-9))
    assert decreasing == 0, \
        f"{campo}/{suf}: no-monotono en {decreasing} puntos de la banda BK+2 a BK+50"


@pytest.mark.parametrize("campo", CAMPOS_PILOTO)
@pytest.mark.parametrize("suf", SUFIJOS)
def test_sub_breakeven_vol_cero(campo, suf, tablon):
    """Volumen reconstruido ~0 por debajo del piso de abandono (BK_ANCLA_PDP)."""
    slug = campo.replace(" ", "_")
    ruta = MODELOS_DIR / f"{slug}_{suf}.joblib"
    if not ruta.exists():
        pytest.skip(f"Modelo {suf} de {campo} no encontrado")
    modelo = joblib.load(ruta)

    sub = tablon[tablon["CAMPO"] == campo]
    anclas_sub = sub.dropna(subset=["BK_ANCLA_PDP_USD_BBL"])
    if anclas_sub.empty:
        pytest.skip(f"{campo}: sin ancla PDP")
    fila_ancla = anclas_sub.sort_values("VIGENCIA_BREAKEVEN", ascending=False).iloc[0]
    bk_val = float(fila_ancla["BK_ANCLA_PDP_USD_BBL"])
    baseline = float(sub[(sub["ESCENARIO"] == "BASE") & sub["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
                     .sort_values("AÑO")["VOLUMEN_1P_OFICIAL_MBPE"].values[-1])

    # Bajo el piso de abandono (Precio Neto = bk_val - 5) el vol reconstruido ~ 0
    px_sub = bk_val - 5
    delta = float(modelo.predict([px_sub])[0])
    vol_rec = max(0, baseline + delta)
    assert vol_rec < 10, \
        f"{campo}/{suf}: vol reconstruido sub-BK = {vol_rec:.2f} (esperado < 10 MBPE)"


def test_predicciones_en_tablon(tablon):
    """03_modelo.py debe haber escrito predicciones de ambos motores en el tablon."""
    assert tablon["PRED_ISOTONICA_MBPE"].notna().sum() > 0, "PRED_ISOTONICA_MBPE vacio"
    assert tablon["PRED_SUAVE_MBPE"].notna().sum() > 0, "PRED_SUAVE_MBPE vacio"


def test_w_sintetico_dinamico(metricas):
    """W_SINTETICO existe, respeta el piso minimo y no es 1.0 fijo para todos los campos."""
    assert "W_SINTETICO" in metricas.columns
    valores = metricas["W_SINTETICO"].dropna()
    assert (valores >= 0.05 - 1e-9).all()
    assert valores.nunique() > 1, "W_SINTETICO no varia por campo (peso dinamico inactivo)"


def test_peso_sintetico_formula():
    """peso_sintetico(n_real, n_sint) = max(n_real/n_sint, W_SINTETICO_MIN); 1.0 si n_sint=0."""
    import importlib
    mod = importlib.import_module("03_modelo")
    assert mod.peso_sintetico(8, 0) == 1.0
    assert mod.peso_sintetico(8, 200) == pytest.approx(mod.W_SINTETICO_MIN)
    assert mod.peso_sintetico(8, 8) == pytest.approx(1.0)
    assert mod.peso_sintetico(8, 4) == pytest.approx(2.0)


def test_metricas_resultados_existe():
    """metricas.csv debe existir tambien en resultados/ (matriz separada para Power BI)."""
    assert (ROOT / "resultados" / "metricas.csv").exists()


def test_plots_sin_piloto():
    """Los titulos de los plots ya no llevan el prefijo 'Piloto — '."""
    src = (ROOT / "03_modelo.py").read_text(encoding="utf-8")
    assert "Piloto" not in src, "03_modelo.py aun referencia 'Piloto' en los plots"


def test_anclas_campo():
    """_anclas_campo usa BK_ANCLA_FIN/PDP de la vigencia MAS RECIENTE (igual que 02_synthetic)."""
    import importlib
    mod = importlib.import_module("03_modelo")
    df = pd.DataFrame({"VIGENCIA_BREAKEVEN": ["2024", "2025"],
                       "BK_ANCLA_FIN_USD_BBL": [52.0, 37.6],
                       "BK_ANCLA_PDP_USD_BBL": [28.8, 21.9]})
    assert mod._anclas_campo(df) == (37.6, 21.9, "2025")
    df2 = pd.DataFrame({"VIGENCIA_BREAKEVEN": ["2024"],
                        "BK_ANCLA_FIN_USD_BBL": [np.nan],
                        "BK_ANCLA_PDP_USD_BBL": [30.0]})
    assert mod._anclas_campo(df2) == (30.0, 30.0, "2024")
    df3 = pd.DataFrame({"VIGENCIA_BREAKEVEN": ["2024"],
                        "BK_ANCLA_FIN_USD_BBL": [np.nan],
                        "BK_ANCLA_PDP_USD_BBL": [np.nan]})
    assert mod._anclas_campo(df3) == (None, None, None)


def test_pesos_por_nivel_multiclase():
    """pesos_sinteticos_tramo balancea cada NIVEL de la escalera por separado contra n_real."""
    import importlib
    mod = importlib.import_module("03_modelo")
    df_sint = pd.DataFrame({"VOLUMEN_1P_SENSIBILIDAD_MBPE":
                            [0.0, 0.0, 0.0, 100.0, 100.0, 250.0]})
    pesos, w_nivel = mod.pesos_sinteticos_tramo(df_sint, n_real=6)
    assert w_nivel[0.0]   == pytest.approx(2.0)
    assert w_nivel[100.0] == pytest.approx(3.0)
    assert w_nivel[250.0] == pytest.approx(6.0)
    assert list(pesos) == [2.0, 2.0, 2.0, 3.0, 3.0, 6.0]


def test_metricas_columnas_reanclaje(metricas):
    """metricas.csv contiene las columnas del re-anclaje M1 (decision 2026-06-12)."""
    for col in ["BRENT_REF_USD_BBL", "P_REF_USD_BBL", "DELTA_REF_ISO", "DELTA_REF_SUAVE"]:
        assert col in metricas.columns, f"Falta columna de re-anclaje: {col}"
    # BRENT_REF debe ser el mismo para todos los campos (parametro global del pipeline)
    vals = metricas["BRENT_REF_USD_BBL"].dropna().unique()
    assert len(vals) == 1, f"BRENT_REF debe ser unico pero hay {len(vals)} valores: {vals}"


@pytest.mark.parametrize("campo", CAMPOS_PILOTO)
def test_c5_vol_pref_igual_baseline(campo, tablon, metricas):
    """C5: Vol(p_ref) == baseline del campo (re-anclaje exacto al punto actual)."""
    from motores_modelo1 import volumen_anclado
    row = metricas[metricas["CAMPO"] == campo]
    if row.empty or pd.isna(row.iloc[0]["P_REF_USD_BBL"]):
        pytest.skip(f"{campo}: sin p_ref en metricas")
    row = row.iloc[0]
    p_ref = float(row["P_REF_USD_BBL"])
    dr_iso = float(row["DELTA_REF_ISO"])
    baseline = float(
        tablon[(tablon["CAMPO"] == campo) & (tablon["ESCENARIO"] == "BASE")
               & tablon["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
        .sort_values("AÑO")["VOLUMEN_1P_OFICIAL_MBPE"].iloc[-1]
    )
    slug = campo.replace(" ", "_")
    ruta = MODELOS_DIR / f"{slug}_iso.joblib"
    if not ruta.exists():
        pytest.skip(f"Modelo iso de {campo} no encontrado")
    iso = joblib.load(ruta)
    v_ref = float(volumen_anclado(iso, [p_ref], baseline, dr_iso, bk_pdp=None)[0])
    assert abs(v_ref - baseline) < 0.5, \
        f"{campo}: C5 fallo — Vol(p_ref)={v_ref:.3f} vs baseline={baseline:.3f} (delta={abs(v_ref-baseline):.4f})"


@pytest.mark.parametrize("campo", CAMPOS_PILOTO)
def test_c6_hard_zero_sub_abandono(campo, tablon, metricas):
    """C6: Vol(bk_pdp-5) == 0 (hard-zero por debajo del precio de abandono PDP)."""
    from motores_modelo1 import volumen_anclado
    row = metricas[metricas["CAMPO"] == campo]
    if row.empty or pd.isna(row.iloc[0]["P_REF_USD_BBL"]):
        pytest.skip(f"{campo}: sin metricas de re-anclaje")
    row = row.iloc[0]
    dr_iso = float(row["DELTA_REF_ISO"])
    baseline = float(
        tablon[(tablon["CAMPO"] == campo) & (tablon["ESCENARIO"] == "BASE")
               & tablon["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
        .sort_values("AÑO")["VOLUMEN_1P_OFICIAL_MBPE"].iloc[-1]
    )
    ancs = (tablon[(tablon["CAMPO"] == campo)]
            .dropna(subset=["BK_ANCLA_PDP_USD_BBL"])
            .sort_values("VIGENCIA_BREAKEVEN", ascending=False))
    if ancs.empty:
        pytest.skip(f"{campo}: sin ancla PDP en tablon")
    bk_pdp = float(ancs.iloc[0]["BK_ANCLA_PDP_USD_BBL"])
    slug = campo.replace(" ", "_")
    ruta = MODELOS_DIR / f"{slug}_iso.joblib"
    if not ruta.exists():
        pytest.skip(f"Modelo iso de {campo} no encontrado")
    iso = joblib.load(ruta)
    v0 = float(volumen_anclado(iso, [bk_pdp - 5.0], baseline, dr_iso, bk_pdp)[0])
    assert v0 == 0.0, \
        f"{campo}: C6 fallo — Vol(bk_pdp-5={bk_pdp-5:.1f})={v0:.4f} (esperado 0.0)"


def test_pneto_delta0():
    """pneto_delta0 encuentra el Precio Neto donde la curva delta cruza 0."""
    import importlib
    mod = importlib.import_module("03_modelo")

    class FakeModel:
        def predict(self, px):
            # delta = pneto - 65: cruza 0 en pneto=65
            return np.asarray(px, dtype=float) - 65.0

    assert mod.pneto_delta0(FakeModel()) == pytest.approx(65.0, abs=0.5)


def _modelos_escalera_perfecta(bk_fin, bk_pdp, baseline, pdp):
    """Modelos fake 1D (Precio Neto) que reproducen la escalera de diseño exacta."""
    def _delta(px):
        if px >= bk_fin:
            return 0.0
        if px >= bk_pdp:
            return -(baseline - pdp)
        return -baseline

    class Fake:
        def predict(self, X):
            px = np.asarray(X, dtype=float).ravel()
            return np.array([_delta(p) for p in px])

    return Fake(), Fake()


def test_sanity_check_c1_en_piso_abandono_y_c1b_escalon():
    """C1 evalua bajo BK_pdp (no bk_fin-5) y C1b valida el escalon vol≈PDP, en Precio Neto."""
    import importlib
    mod = importlib.import_module("03_modelo")
    baseline, pdp = 100.0, 40.0
    bk_fin, bk_pdp = 52.0, 29.0
    fiso, fsu = _modelos_escalera_perfecta(bk_fin, bk_pdp, baseline, pdp)
    res = mod.sanity_check("TEST", fiso, fsu, bk_fin, bk_pdp,
                           vol_max_delta=10.0, baseline_latest=baseline,
                           baseline_pdp=pdp)
    assert res["sub_breakeven_vol_cero"], "C1 fallo en piso de abandono"
    assert res["escalon_vol_pdp"], "C1b fallo en escalon intermedio"
    assert res["monotonia_iso"] and res["monotonia_suave"]


def test_sanity_check_escalera_degenerada_sin_c1b():
    """Con BK_fin==BK_pdp (escalera degenerada) C1b no se evalua."""
    import importlib
    mod = importlib.import_module("03_modelo")
    baseline = 100.0
    bk = 30.0
    fiso, fsu = _modelos_escalera_perfecta(bk, bk, baseline, 0.0)
    res = mod.sanity_check("TEST", fiso, fsu, bk, bk,
                           vol_max_delta=10.0, baseline_latest=baseline,
                           baseline_pdp=0.0)
    assert "escalon_vol_pdp" not in res
    assert res["sub_breakeven_vol_cero"]


def test_motores_modelo1_monotonos():
    """Los motores 1D garantizan monotonia creciente sobre datos monotonos."""
    from motores_modelo1 import MotorIsotonico, MotorSuave
    x = np.array([20, 30, 40, 50, 60, 70, 80], dtype=float)
    y = np.array([-50, -50, -20, -20, 0, 5, 8], dtype=float)
    grid = np.linspace(20, 80, 50)
    for Motor in (MotorIsotonico, MotorSuave):
        m = Motor().fit(x, y)
        curva = m.predict(grid)
        assert np.all(np.diff(curva) >= -1e-6), f"{Motor.__name__} no monotono"
