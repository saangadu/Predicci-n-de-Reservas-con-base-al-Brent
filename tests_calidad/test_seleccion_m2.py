"""
tests_calidad/test_seleccion_m2.py — Gate del track Calidad para la seleccion de
familia M2 (Brent -> Neto) por campo (investigacion 2026-07-15).

Valida el registro resultados_calidad/seleccion_metodos_m2.csv (evidencia de
analisis_m2.py) y su aplicacion en resultados_calidad/correlacion_brent.csv
(dispatch de 03b bajo PRED_M2_SELECCION):
  - integridad del registro: mejora > 5%, parametros parseables, estados validos
  - fisica de cada familia adoptada: monotonia dNeto/dBrent > 0 en $40-120,
    Neto(40) >= 0, descuento implicito acotado (m2_familias.pasa_fisica)
  - coherencia del dispatch: campo adoptado -> METODO/DATASET/M2_PARAMS publicados
    en correlacion_brent.csv con BETA equivalente en rango fisico

Lee resultados_calidad/ directamente (independiente de env), igual que
test_seleccion.py. NO toca Produccion.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
CAL = ROOT / "resultados_calidad"

# m2_familias vive en la raiz del proyecto
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import m2_familias as F  # noqa: E402

MEJORA_MIN = 5.0  # s12: adopcion Calidad si mejora >5% vs champion
ESTADOS_VALIDOS = {"ADOPTADO", "ADOPTADO_CALIDAD", "RECHAZADO"}


@pytest.fixture(scope="module")
def registro():
    ruta = CAL / "seleccion_metodos_m2.csv"
    if not ruta.exists():
        pytest.skip("seleccion_metodos_m2.csv no existe (correr analisis_m2.py)")
    r = pd.read_csv(ruta)
    if r.empty:
        pytest.skip("registro M2 vacio (sin adopciones)")
    return r


@pytest.fixture(scope="module")
def adoptados(registro):
    return registro[registro["ESTADO"].isin(
        ["ADOPTADO", "ADOPTADO_CALIDAD"])].reset_index(drop=True)


@pytest.fixture(scope="module")
def correlacion():
    ruta = CAL / "correlacion_brent.csv"
    if not ruta.exists():
        pytest.skip("correlacion_brent.csv de Calidad no existe (correr pipeline)")
    return pd.read_csv(ruta)


def test_registro_integro(registro, adoptados):
    """Estados validos, familias conocidas, mejora >= umbral en toda adopcion."""
    inval = set(registro["ESTADO"].unique()) - ESTADOS_VALIDOS
    assert not inval, f"Estados invalidos en registro M2: {inval}"
    fam_inval = set(adoptados["METODO"].unique()) - set(F.FAMILIAS_CANDIDATAS)
    assert not fam_inval, f"Familias desconocidas adoptadas: {fam_inval}"
    bajo = adoptados[adoptados["MEJORA_PCT"] <= MEJORA_MIN]
    assert bajo.empty, f"Adopciones M2 sin mejora > {MEJORA_MIN}%:\n{bajo}"
    # Datasets validos (eje 2 de la investigacion)
    ds_inval = set(adoptados["DATASET"].unique()) - {"HIST", "HIST+CONSOLIDADO"}
    assert not ds_inval, f"Datasets invalidos: {ds_inval}"


def test_mejora_vs_champion(adoptados):
    """El MAE_LOO de la familia adoptada < MAE_LOO del champion Theil-Sen."""
    for _, r in adoptados.iterrows():
        assert r["MAE_LOO_METODO"] < r["MAE_LOO_CHAMPION"], \
            f"{r['CAMPO']}: {r['MAE_LOO_METODO']} no mejora champion {r['MAE_LOO_CHAMPION']}"


def test_params_parseables_y_fisica(adoptados):
    """Los M2_PARAMS del registro reconstruyen una curva que pasa la fisica
    (monotonia $40-120, Neto(40)>=0, descuento implicito acotado)."""
    for _, r in adoptados.iterrows():
        params = F.params_desde_json(r["M2_PARAMS"])
        assert F.pasa_fisica(r["METODO"], params), \
            f"{r['CAMPO']}: familia {r['METODO']} viola la fisica con sus parametros"


def test_dispatch_en_correlacion(adoptados, correlacion):
    """Todo campo adoptado presente en correlacion_brent.csv de Calidad debe llevar
    su familia aplicada (METODO), o el guard M2_SELECCION_NO_APLICADA visible."""
    if "M2_PARAMS" not in correlacion.columns:
        pytest.skip("corrida de Calidad sin PRED_M2_SELECCION (columnas M2 ausentes)")
    corr = correlacion.set_index("CAMPO")
    for _, r in adoptados.iterrows():
        campo = r["CAMPO"]
        if campo not in corr.index:
            continue
        fila = corr.loc[campo]
        aplicado = str(fila.get("ALERTA", "")) == "M2_SELECCION"
        guard = str(fila.get("ALERTA", "")) == "M2_SELECCION_NO_APLICADA"
        assert aplicado or guard, \
            f"{campo}: adoptado en registro M2 pero sin rastro en correlacion_brent.csv"
        if aplicado:
            assert fila["METODO"] == r["METODO"], \
                f"{campo}: METODO {fila['METODO']} != registro {r['METODO']}"
            # BETA equivalente publicada en rango fisico (mismo gate del nucleo)
            assert 0.1 < float(fila["BETA"]) < 1.6, \
                f"{campo}: BETA_eq={fila['BETA']} fuera de rango fisico"


def test_curva_aplicada_monotona(correlacion):
    """Toda fila de correlacion con M2_PARAMS debe producir curva monotona via
    neto_desde_brent (la funcion que consumen 03 y 04)."""
    if "M2_PARAMS" not in correlacion.columns:
        pytest.skip("corrida de Calidad sin PRED_M2_SELECCION")
    import importlib
    import numpy as np
    m2 = importlib.import_module("03b_correlacion_brent")
    con_params = correlacion[correlacion["M2_PARAMS"].fillna("").astype(str)
                             .str.strip() != ""]
    grid = np.linspace(40.0, 120.0, 81)
    for _, fila in con_params.iterrows():
        y = m2.neto_desde_brent(fila.to_dict(), grid)
        assert np.all(np.diff(y) > 0), \
            f"{fila['CAMPO']}: curva M2 aplicada no monotona en $40-120"
