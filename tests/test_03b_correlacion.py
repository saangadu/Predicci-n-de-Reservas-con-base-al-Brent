"""
test_03b_correlacion.py — Gate Fase 3b: Modelo 2 (Precio Neto = g(Brent))

Verifica el ajuste Theil-Sen por campo, la monotonia (β>0), el orden de escenarios
(BAJO<=BASE<=ALTO) y la coherencia del descuento implicito.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

STAGING    = ROOT / "datos" / "staging"
RESULTADOS = ROOT / "resultados"
GATE_DORADO = ["CASTILLA", "CASTILLA NORTE", "CASTILLA ESTE", "RUBIALES"]

m2 = importlib.import_module("03b_correlacion_brent")


@pytest.fixture(scope="module")
def coefs():
    ruta = STAGING / "correlacion_brent.csv"
    assert ruta.exists(), "correlacion_brent.csv no existe: correr 03b_correlacion_brent.py"
    return pd.read_csv(ruta)


def test_csv_existe_staging_y_resultados():
    assert (STAGING / "correlacion_brent.csv").exists()
    assert (RESULTADOS / "correlacion_brent.csv").exists()


def test_columnas_obligatorias(coefs):
    cols = ["CAMPO", "N_PUNTOS", "METODO", "ALPHA", "BETA", "R2", "RMSE",
            "RESID_P10", "RESID_P50", "RESID_P90", "DESCUENTO_IMPLICITO_REF"]
    for c in cols:
        assert c in coefs.columns, f"Falta columna {c} en correlacion_brent.csv"


def test_beta_positiva(coefs):
    """Invariante: Brent↑ -> Neto↑ (β>0) para preservar la monotonia de la composicion."""
    neg = coefs[coefs["BETA"] <= 0]
    assert neg.empty, f"{len(neg)} campos con BETA<=0 (rompe Brent↑->Neto↑): " \
                      f"{neg['CAMPO'].tolist()[:5]}"


def test_metodos_validos(coefs):
    validos = {"THEILSEN", "PROPORCIONAL", "FALLBACK_MEDIANAS"}
    inval = set(coefs["METODO"].unique()) - validos
    assert not inval, f"Metodos invalidos: {inval}"


def test_gate_dorado_theilsen(coefs):
    """Los campos del gate dorado tienen suficientes puntos para Theil-Sen con buen ajuste."""
    for campo in GATE_DORADO:
        r = coefs[coefs["CAMPO"] == campo]
        assert not r.empty, f"{campo} ausente en correlacion_brent.csv"
        r = r.iloc[0]
        assert r["METODO"] == "THEILSEN", f"{campo}: metodo {r['METODO']} (esperado THEILSEN)"
        assert r["R2"] > 0.5, f"{campo}: R2={r['R2']} demasiado bajo"
        assert 0.3 < r["BETA"] < 1.5, f"{campo}: BETA={r['BETA']} fuera de rango razonable"


def test_orden_escenarios():
    """neto_desde_brent: BAJO<=BASE<=ALTO por construccion (centrado en P50)."""
    coef = {"ALPHA": 4.0, "BETA": 0.8,
            "RESID_P10": 2.0, "RESID_P50": 3.0, "RESID_P90": 5.0}  # P10>0 a proposito
    brent = np.array([55.0, 70.0, 85.0])
    bajo = m2.neto_desde_brent(coef, brent, "BAJO")
    base = m2.neto_desde_brent(coef, brent, "BASE")
    alto = m2.neto_desde_brent(coef, brent, "ALTO")
    assert np.all(bajo <= base + 1e-9), "BAJO debe ser <= BASE incluso con P10>0"
    assert np.all(base <= alto + 1e-9), "BASE debe ser <= ALTO"


def test_ajustar_campo_theilsen():
    """ajustar_campo recupera una recta limpia Neto=2+0.8*Brent (sin ruido)."""
    b = np.array([50, 60, 70, 80, 90, 100], dtype=float)
    y = 2.0 + 0.8 * b
    g = pd.DataFrame({"BRENT_FLAT_USD_BBL": b, "PRECIO_NETO_USD_BBL": y})
    coef = m2.ajustar_campo(g, med_cal=-7.0, med_tra=-3.0)
    assert coef["METODO"] == "THEILSEN"
    assert coef["BETA"] == pytest.approx(0.8, abs=0.02)
    assert coef["ALPHA"] == pytest.approx(2.0, abs=0.5)
    assert coef["R2"] == pytest.approx(1.0, abs=1e-6)


def test_ajustar_campo_beta_no_positiva_degrada():
    """Si la nube tiene pendiente <=0, se degrada a proporcional (β>0) con alerta."""
    b = np.array([50, 60, 70, 80, 90], dtype=float)
    y = np.array([60, 58, 55, 54, 52], dtype=float)   # decreciente
    g = pd.DataFrame({"BRENT_FLAT_USD_BBL": b, "PRECIO_NETO_USD_BBL": y})
    coef = m2.ajustar_campo(g, med_cal=-7.0, med_tra=-3.0)
    assert coef["BETA"] > 0, "La degradacion debe garantizar BETA>0"
    assert coef["ALERTA"] == "BETA_NO_POSITIVA"


def test_ajustar_campo_pocos_puntos():
    """Con 2<=n<5 usa proporcional Neto=k*Brent."""
    b = np.array([60.0, 80.0], dtype=float)
    y = np.array([50.0, 68.0], dtype=float)
    g = pd.DataFrame({"BRENT_FLAT_USD_BBL": b, "PRECIO_NETO_USD_BBL": y})
    coef = m2.ajustar_campo(g, med_cal=-7.0, med_tra=-3.0)
    assert coef["METODO"] == "PROPORCIONAL"
    assert coef["ALPHA"] == 0.0
    assert coef["BETA"] > 0


def test_descuento_implicito_ref(coefs):
    """DESCUENTO_IMPLICITO_REF = BRENT_REF - NETO_REF_BASE."""
    sub = coefs.dropna(subset=["BRENT_REF", "NETO_REF_BASE", "DESCUENTO_IMPLICITO_REF"])
    esperado = sub["BRENT_REF"] - sub["NETO_REF_BASE"]
    diff = (esperado - sub["DESCUENTO_IMPLICITO_REF"]).abs()
    assert (diff < 0.05).all(), "DESCUENTO_IMPLICITO_REF no coincide con Brent_ref - Neto_ref"
