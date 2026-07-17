"""
test_03b_correlacion.py — Gate Fase 3b: Modelo 2 (Precio Aceite = g(Brent))

Verifica el ajuste Theil-Sen por campo, la monotonia (β>0), el entrenamiento
solo-HIST (sin quarters Consolidado), el fallback de portafolio taggeado y la
coherencia del descuento implicito. Escenarios BAJO/ALTO retirados 2026-06-12.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Paths por track (Produccion vs Calidad): centralizados en tests/conftest.py
from rutas_track import STAGING, RESULTADOS, ES_CALIDAD  # noqa: E402
# Gate Dorado = pareto-9 (directriz 2026-07-09; ver tests/test_norte.py y docs/NORTE.md).
# CHICHIMENE SW re-fusionado en CHICHIMENE (agregacion v3, 2026-07-09 s3): ya no es campo.
GATE_DORADO = ["RUBIALES", "CASTILLA", "CAÑO SUR ESTE", "CASTILLA NORTE", "AKACIAS",
               "CHICHIMENE", "LA CIRA", "CUPIAGUA", "YARIGUI-CANTAGALLO"]
# THEILSEN/LOO exigidos solo con historia suficiente (n>=5, regla G5 NORTE).
GATE_N_MIN_THEILSEN = 5

m2 = importlib.import_module("03b_correlacion_brent")


@pytest.fixture(scope="module")
def coefs():
    ruta = STAGING / "correlacion_brent.csv"
    assert ruta.exists(), "correlacion_brent.csv no existe: correr 03b_correlacion_brent.py"
    return pd.read_csv(ruta)


@pytest.fixture(scope="module")
def tablon():
    ruta = STAGING / "tablon_unico.parquet"
    assert ruta.exists(), "tablon_unico.parquet no existe: correr 01_etl.py"
    return pd.read_parquet(ruta)


def test_csv_existe_staging_y_resultados():
    assert (STAGING / "correlacion_brent.csv").exists()
    assert (RESULTADOS / "correlacion_brent.csv").exists()


def test_columnas_obligatorias(coefs):
    cols = ["CAMPO", "N_PUNTOS", "METODO", "ALPHA", "BETA", "R2", "RMSE",
            "R2_LOO", "MAE_LOO", "ES_FALLBACK", "ALERTA", "DESCUENTO_IMPLICITO_REF"]
    for c in cols:
        assert c in coefs.columns, f"Falta columna {c} en correlacion_brent.csv"


def test_sin_columnas_de_escenarios(coefs):
    """Escenarios BAJO/ALTO retirados 2026-06-12: sin residuales ni cobertura de banda."""
    for c in ["RESID_P10", "RESID_P50", "RESID_P90", "PCT_EN_BANDA"]:
        assert c not in coefs.columns, f"Columna de escenarios {c} no debe existir"


def test_beta_positiva(coefs):
    """Invariante: Brent↑ -> Aceite↑ (β>0) para preservar la monotonia de la composicion."""
    neg = coefs[coefs["BETA"] <= 0]
    assert neg.empty, f"{len(neg)} campos con BETA<=0 (rompe Brent↑->Aceite↑): " \
                      f"{neg['CAMPO'].tolist()[:5]}"


# Metodos del nucleo lineal + familias M2 ratificadas (promocion s13, 2026-07-17:
# PRED_M2_SELECCION en FLAGS_RATIFICADOS y registro seleccion_metodos_m2.csv en
# ESTADO=ADOPTADO). Ver analisis_m2.py y seleccion_metodos_m2.csv.
METODOS_NUCLEO = {"THEILSEN", "PROPORCIONAL", "FALLBACK_BETA_PORTAFOLIO"}
METODOS_M2_FAMILIAS = {"DESCOMPUESTO", "SEGMENTADA", "HUBER", "CUADRATICA_MONOTONA"}


def test_metodos_validos(coefs):
    # Ambos tracks aceptan nucleo + familias adoptadas por campo via CSV (s13).
    validos = METODOS_NUCLEO | METODOS_M2_FAMILIAS
    inval = set(coefs["METODO"].unique()) - validos
    assert not inval, f"Metodos invalidos: {inval}"


def test_solo_puntos_hist(coefs, tablon):
    """M2 entrena solo con cierres HIST (ES_BASELINE): N_PUNTOS por campo no puede
    superar los puntos HIST reales con Brent y precio disponibles.
    Excepcion Calidad: campos cuya seleccion M2 adopto DATASET=HIST+CONSOLIDADO
    (directriz de selectividad por campo 2026-07-15) legitimamente entrenan con mas
    puntos — se omiten de este check de identidad."""
    hist = tablon[(~tablon["ES_SINTETICO"]) & tablon["ES_BASELINE"]
                  & tablon["BRENT_FLAT_USD_BBL"].notna()
                  & tablon["PRECIO_NETO_USD_BBL"].notna()]
    n_hist = hist.groupby("CAMPO").size()
    for campo in GATE_DORADO:
        r = coefs[coefs["CAMPO"] == campo].iloc[0]
        # Campo adoptado con sensibilidades del Consolidado en el train (s13: ambos tracks)
        if str(r.get("DATASET", "HIST")) == "HIST+CONSOLIDADO":
            continue
        assert r["N_PUNTOS"] == n_hist.get(campo, 0), \
            f"{campo}: N_PUNTOS={r['N_PUNTOS']} != puntos HIST={n_hist.get(campo, 0)} " \
            f"(¿se colaron quarters Consolidado?)"


def test_fallback_taggeado(coefs):
    """Todo FALLBACK_BETA_PORTAFOLIO debe ser visible: ES_FALLBACK=True, ALERTA
    no vacia, ALPHA=0 y BETA igual al k unico de portafolio."""
    fb = coefs[coefs["METODO"] == "FALLBACK_BETA_PORTAFOLIO"]
    if fb.empty:
        pytest.skip("Sin campos en fallback")
    assert fb["ES_FALLBACK"].all(), "Fallback sin ES_FALLBACK=True"
    assert (fb["ALERTA"] == "SIN_HIST_FALLBACK_BETA").all(), "Fallback sin ALERTA"
    assert (fb["ALPHA"] == 0.0).all(), "Fallback con intercepto != 0"
    assert fb["BETA"].nunique() == 1, "El k de portafolio debe ser unico"
    # Y los no-fallback no deben llevar el tag
    no_fb = coefs[coefs["METODO"] != "FALLBACK_BETA_PORTAFOLIO"]
    assert not no_fb["ES_FALLBACK"].any(), "ES_FALLBACK=True en campos con regresion propia"


def test_gate_dorado_theilsen(coefs):
    """Los campos del gate dorado tienen recta propia con buen ajuste; con la
    promocion s13 el metodo puede ser THEILSEN o una familia M2 adoptada por campo,
    en ambos tracks. Se mantienen los checks de calidad (BETA razonable, R2>0.5)."""
    for campo in GATE_DORADO:
        r = coefs[coefs["CAMPO"] == campo]
        assert not r.empty, f"{campo} ausente en correlacion_brent.csv"
        r = r.iloc[0]
        # BETA razonable: para metodos no-lineales la BETA reportada es la pendiente
        # equivalente/promedio y debe seguir en el rango fisico
        assert 0.3 < r["BETA"] < 1.5, f"{campo}: BETA={r['BETA']} fuera de rango razonable"
        if r["N_PUNTOS"] < GATE_N_MIN_THEILSEN:
            continue   # recta PROPORCIONAL propia aceptada (poca historia)
        assert r["METODO"] in (METODOS_NUCLEO | METODOS_M2_FAMILIAS), \
            f"{campo}: metodo {r['METODO']} no reconocido"
        assert r["R2"] > 0.5, f"{campo}: R2={r['R2']} demasiado bajo"


def test_neto_desde_brent_es_recta(coefs):
    """neto_desde_brent (sin escenarios) = ALPHA + BETA*Brent, vectorizado."""
    coef = {"ALPHA": 4.0, "BETA": 0.8}
    brent = np.array([55.0, 70.0, 85.0])
    out = m2.neto_desde_brent(coef, brent)
    assert np.allclose(out, 4.0 + 0.8 * brent)


def test_ajustar_campo_theilsen():
    """ajustar_campo recupera una recta limpia Aceite=2+0.8*Brent (sin ruido)."""
    b = np.array([50, 60, 70, 80, 90, 100], dtype=float)
    y = 2.0 + 0.8 * b
    g = pd.DataFrame({"BRENT_FLAT_USD_BBL": b, "PRECIO_NETO_USD_BBL": y})
    coef = m2.ajustar_campo(g)
    assert coef["METODO"] == "THEILSEN"
    assert coef["BETA"] == pytest.approx(0.8, abs=0.02)
    assert coef["ALPHA"] == pytest.approx(2.0, abs=0.5)
    assert coef["R2"] == pytest.approx(1.0, abs=1e-6)


def test_ajustar_campo_beta_no_positiva_degrada():
    """Si la nube tiene pendiente <=0, se degrada a proporcional (β>0) con alerta."""
    b = np.array([50, 60, 70, 80, 90], dtype=float)
    y = np.array([60, 58, 55, 54, 52], dtype=float)   # decreciente
    g = pd.DataFrame({"BRENT_FLAT_USD_BBL": b, "PRECIO_NETO_USD_BBL": y})
    coef = m2.ajustar_campo(g)
    assert coef["BETA"] > 0, "La degradacion debe garantizar BETA>0"
    assert coef["ALERTA"] == "BETA_NO_POSITIVA"


def test_ajustar_campo_pocos_puntos():
    """Con 2<=n<5 usa proporcional Aceite=k*Brent."""
    b = np.array([60.0, 80.0], dtype=float)
    y = np.array([50.0, 68.0], dtype=float)
    g = pd.DataFrame({"BRENT_FLAT_USD_BBL": b, "PRECIO_NETO_USD_BBL": y})
    coef = m2.ajustar_campo(g)
    assert coef["METODO"] == "PROPORCIONAL"
    assert coef["ALPHA"] == 0.0
    assert coef["BETA"] > 0


def test_ajustar_campo_sin_datos_marca_fallback():
    """Con n<2 queda PENDIENTE_FALLBACK (lo resuelve la segunda pasada del __main__)."""
    g = pd.DataFrame({"BRENT_FLAT_USD_BBL": [70.0], "PRECIO_NETO_USD_BBL": [60.0]})
    coef = m2.ajustar_campo(g)
    assert coef["METODO"] == "PENDIENTE_FALLBACK"
    assert coef["ES_FALLBACK"] is True
    assert coef["ALERTA"] == "SIN_HIST_FALLBACK_BETA"


def test_loo_no_degenerado(coefs):
    """
    El LOO-CV de la recta debe estar disponible y ser informativo para el gate dorado
    (R2_LOO en [-1,1], y razonablemente alto >0.5 dado que el ajuste in-sample es bueno).
    R2_LOO < R2 in-sample es lo esperado (la validacion out-of-sample castiga el optimismo).
    """
    for campo in GATE_DORADO:
        r = coefs[coefs["CAMPO"] == campo]
        assert not r.empty, f"{campo} ausente en correlacion_brent.csv"
        r = r.iloc[0]
        if r["N_PUNTOS"] < GATE_N_MIN_THEILSEN:
            continue   # sin historia suficiente el LOO de la recta no es informativo
        assert pd.notna(r["R2_LOO"]), f"{campo}: R2_LOO ausente (LOO degenerado)"
        assert -1.0 <= r["R2_LOO"] <= 1.0, f"{campo}: R2_LOO={r['R2_LOO']} fuera de rango"
        assert r["R2_LOO"] > 0.5, f"{campo}: R2_LOO={r['R2_LOO']} demasiado bajo (no generaliza)"
        assert pd.notna(r["MAE_LOO"]) and r["MAE_LOO"] >= 0, f"{campo}: MAE_LOO invalido"


def test_loo_recta_recupera_recta_limpia():
    """En una recta sin ruido, el LOO debe ser casi perfecto (R2_LOO≈1, MAE_LOO≈0)."""
    b = np.array([50, 60, 70, 80, 90, 100], dtype=float)
    y = 2.0 + 0.8 * b
    mae_loo, r2_loo = m2._loo_recta(b, y, "THEILSEN")
    assert mae_loo == pytest.approx(0.0, abs=1e-6)
    assert r2_loo == pytest.approx(1.0, abs=1e-6)


def test_descuento_implicito_ref(coefs):
    """DESCUENTO_IMPLICITO_REF = BRENT_REF - NETO_REF_BASE."""
    sub = coefs.dropna(subset=["BRENT_REF", "NETO_REF_BASE", "DESCUENTO_IMPLICITO_REF"])
    esperado = sub["BRENT_REF"] - sub["NETO_REF_BASE"]
    diff = (esperado - sub["DESCUENTO_IMPLICITO_REF"]).abs()
    assert (diff < 0.05).all(), "DESCUENTO_IMPLICITO_REF no coincide con Brent_ref - Neto_ref"
