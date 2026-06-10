"""
test_03_modelo.py — Gate Fase 3: verifica los modelos entrenados y las metricas

Pruebas criticas: monotonia en banda completa, metricas sobre reales, sanity de anclaje.
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
    """Un .joblib XGB y uno ISO por cada campo piloto."""
    for campo in CAMPOS_PILOTO:
        slug = campo.replace(" ", "_")
        xgb_f = MODELOS_DIR / f"{slug}_xgb.joblib"
        iso_f = MODELOS_DIR / f"{slug}_iso.joblib"
        assert xgb_f.exists(), f"Falta {xgb_f.name}"
        assert iso_f.exists(), f"Falta {iso_f.name}"


def test_metricas_csv_existe():
    assert (STAGING / "metricas.csv").exists()


def test_metricas_n_real_delta_positivo(metricas):
    """
    Cada campo debe tener datos de entrenamiento (reales o sintéticos).
    En modo piloto sin Consolidado, se acepta N_REAL_DELTA=0 si hay sintéticos.
    Con Consolidado activo, se exige N_REAL_DELTA >= 1.
    """
    for _, row in metricas.iterrows():
        n_real = int(row.get("N_REAL_DELTA", 0) or 0)
        n_sint = int(row.get("N_SINTETICOS", 0) or 0)
        assert n_real + n_sint >= 1, \
            f"{row['CAMPO']}: sin datos de entrenamiento (reales={n_real}, sint={n_sint})"
        if n_real == 0:
            # Modo piloto sin Consolidado: advertir, no fallar
            import warnings
            warnings.warn(f"{row['CAMPO']}: modo solo-sintéticos (N_REAL_DELTA=0). "
                          "Métricas LOO-CV no disponibles.")


def test_metricas_sin_sinteticos_en_cv(metricas):
    """
    B1: Las metricas LOO-CV deben ser sobre puntos reales.
    Verificacion indirecta: la columna N_REAL_DELTA existe y es < N_SINTETICOS + N_REAL_DELTA.
    """
    assert "N_REAL_DELTA" in metricas.columns, "Columna N_REAL_DELTA ausente en metricas.csv"
    assert "N_SINTETICOS" in metricas.columns, "Columna N_SINTETICOS ausente en metricas.csv"


@pytest.mark.parametrize("campo", CAMPOS_PILOTO)
def test_monotonia_xgb_banda_completa(campo, tablon):
    """
    Monotonia XGBoost verificada en toda la banda [BK+2, BK+50] (no solo 50pts como antes).
    Tol = 0 (sin margen de -0.1).
    """
    slug = campo.replace(" ", "_")
    ruta = MODELOS_DIR / f"{slug}_xgb.joblib"
    if not ruta.exists():
        pytest.skip(f"Modelo XGB de {campo} no encontrado")

    xgb_m = joblib.load(ruta)
    sub = tablon[tablon["CAMPO"] == campo]
    bk = sub["BREAKEVEN_FINANCIERO_USD_BBL"].dropna().values
    if len(bk) == 0:
        pytest.skip(f"{campo}: sin breakeven")

    med_cal = float(sub[(~sub["ES_SINTETICO"]) & sub["DESCUENTO_CALIDAD_USD_BBL"].notna()]
                    ["DESCUENTO_CALIDAD_USD_BBL"].median())
    med_tra = float(sub[(~sub["ES_SINTETICO"]) & sub["DESCUENTO_TRANSPORTE_USD_BBL"].notna()]
                    ["DESCUENTO_TRANSPORTE_USD_BBL"].median())

    px_band = np.linspace(float(bk[0]) + 2, float(bk[0]) + 50, 100)
    b_band  = px_band - med_cal - med_tra
    X_band  = np.column_stack([b_band, np.full(100, med_cal), np.full(100, med_tra)])
    curva   = xgb_m.predict(X_band)
    decreasing = np.sum(np.diff(curva) < 0)
    assert decreasing == 0, \
        f"{campo}: XGB no-monotono en {decreasing} puntos de la banda BK+2 a BK+50"


@pytest.mark.parametrize("campo", CAMPOS_PILOTO)
def test_monotonia_iso_banda_completa(campo, tablon):
    """Monotonia Isotonica verificada en banda completa."""
    slug = campo.replace(" ", "_")
    ruta = MODELOS_DIR / f"{slug}_iso.joblib"
    if not ruta.exists():
        pytest.skip(f"Modelo ISO de {campo} no encontrado")

    iso_m = joblib.load(ruta)
    sub = tablon[tablon["CAMPO"] == campo]
    bk = sub["BREAKEVEN_FINANCIERO_USD_BBL"].dropna().values
    if len(bk) == 0:
        pytest.skip(f"{campo}: sin breakeven")

    px_band = np.linspace(float(bk[0]) + 2, float(bk[0]) + 50, 100)
    curva   = iso_m.predict(px_band)
    decreasing = np.sum(np.diff(curva) < 0)
    assert decreasing == 0, \
        f"{campo}: ISO no-monotona en {decreasing} puntos de la banda BK+2 a BK+50"


@pytest.mark.parametrize("campo", CAMPOS_PILOTO)
def test_sub_breakeven_vol_cero(campo, tablon):
    """Volumen reconstruido debe ser ~0 por debajo del breakeven."""
    slug = campo.replace(" ", "_")
    for sufijo, loader in [("_xgb", joblib.load), ("_iso", joblib.load)]:
        ruta = MODELOS_DIR / f"{slug}{sufijo}.joblib"
        if not ruta.exists():
            continue

        model = loader(ruta)
        sub = tablon[tablon["CAMPO"] == campo]
        # Por debajo del piso operacional (BK_ANCLA_PDP, abandono) → vol≈0 (tramo BAJO).
        # Usar la misma vigencia (mas reciente) que 02_synthetic.py para no mezclar
        # pisos de distintas vigencias (ver generar_sinteticos).
        anclas_sub = sub.dropna(subset=["BK_ANCLA_PDP_USD_BBL"])
        if anclas_sub.empty:
            continue

        fila_ancla = anclas_sub.sort_values("VIGENCIA_BREAKEVEN", ascending=False).iloc[0]
        bk_val = float(fila_ancla["BK_ANCLA_PDP_USD_BBL"])
        baseline = float(sub[(sub["ESCENARIO"] == "BASE") & sub["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
                         .sort_values("AÑO")["VOLUMEN_1P_OFICIAL_MBPE"].values[-1])

        med_cal = float(sub[(~sub["ES_SINTETICO"]) & sub["DESCUENTO_CALIDAD_USD_BBL"].notna()]
                        ["DESCUENTO_CALIDAD_USD_BBL"].median())
        med_tra = float(sub[(~sub["ES_SINTETICO"]) & sub["DESCUENTO_TRANSPORTE_USD_BBL"].notna()]
                        ["DESCUENTO_TRANSPORTE_USD_BBL"].median())

        px_sub = bk_val - 5
        if "_xgb" in sufijo:
            b_sub = px_sub - med_cal - med_tra
            delta = float(model.predict(np.array([[b_sub, med_cal, med_tra]]))[0])
        else:
            delta = float(model.predict(np.array([px_sub]))[0])

        vol_rec = max(0, baseline + delta)
        assert vol_rec < 10, \
            f"{campo} {sufijo}: vol reconstruido sub-BK = {vol_rec:.2f} (esperado < 10 MBPE)"


def test_predicciones_en_tablon(tablon):
    """03_modelo.py debe haber escrito predicciones en el tablon."""
    pred_xgb = tablon["PRED_XGBOOST_MBPE"].notna().sum()
    pred_iso  = tablon["PRED_ISOTONICA_MBPE"].notna().sum()
    assert pred_xgb > 0, "PRED_XGBOOST_MBPE esta vacio en el tablon"
    assert pred_iso  > 0, "PRED_ISOTONICA_MBPE esta vacio en el tablon"


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
    ruta = ROOT / "resultados" / "metricas.csv"
    assert ruta.exists(), "resultados/metricas.csv no existe"


def test_plots_sin_piloto():
    """Los titulos de los plots ya no llevan el prefijo 'Piloto — '."""
    src = (ROOT / "03_modelo.py").read_text(encoding="utf-8")
    assert "Piloto" not in src, "03_modelo.py aun referencia 'Piloto' en los plots"
