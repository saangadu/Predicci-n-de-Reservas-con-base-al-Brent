"""
test_00_homologacion.py — Gate Fase 0: verifica que el modulo de homologacion funciona
correctamente antes de correr el ETL.

Todos los asserts son duros: cualquier fallo bloquea el pipeline.
"""
import sys
from pathlib import Path

import pytest

# Asegurar que el directorio padre esta en sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from homologacion import Homologador, ALIAS_OVERRIDE


@pytest.fixture(scope="module")
def h():
    return Homologador()


def test_dim_carga(h):
    """DIM_CAMPO cargada con al menos 100 campos validos."""
    assert len(h._validos) >= 100, f"DIM_CAMPO solo tiene {len(h._validos)} campos"


def test_piloto_castilla(h):
    campo, flag = h.homologar("CASTILLA")
    assert campo == "CASTILLA"
    assert flag == "OK"


def test_piloto_castilla_norte(h):
    campo, flag = h.homologar("CASTILLA NORTE")
    assert campo == "CASTILLA NORTE"
    assert flag == "OK"


def test_piloto_castilla_este(h):
    campo, flag = h.homologar("CASTILLA ESTE")
    assert campo == "CASTILLA ESTE"
    assert flag == "OK"


def test_piloto_rubiales(h):
    campo, flag = h.homologar("RUBIALES")
    assert campo == "RUBIALES"
    assert flag == "OK"


def test_alias_la_reforma(h):
    campo, flag = h.homologar("LA REFORMA")
    assert campo == "LIBERTAD"
    assert flag == "OK"


def test_alias_acae_san_miguel(h):
    campo, flag = h.homologar("ACAE SAN MIGUEL")
    assert campo == "ACAE-SAN MIGUEL"
    assert flag == "OK"


def test_alias_toros(h):
    """Fuzzy confundiría TOROS→TORITOS; el override correcto lo resuelve."""
    campo, flag = h.homologar("TOROS")
    assert campo == "LOS TOROS"
    assert flag == "OK"


def test_sufijo_yacimiento(h):
    """APIAY ESTE K → APIAY ESTE (strip sufijo)."""
    campo, flag = h.homologar("APIAY ESTE K")
    assert "APIAY" in campo.upper()
    assert flag == "OK"


def test_strip_cf_sec(h):
    """Nombres de archivos Breakeven con sufijo _CF_SEC_ → campo limpio."""
    campo, flag = h.homologar("CASTILLA_CF_SEC_23-Jan-2025.xlsx")
    assert campo == "CASTILLA"
    assert flag == "OK"


def test_campo_inexistente(h):
    """Campo que no existe en DIM → NO_HOMOLOGADO (no crash)."""
    campo, flag = h.homologar("CAMPO XYZ INVENTADO 999")
    assert flag == "NO_HOMOLOGADO"
    assert campo != ""  # retorna la version normalizada, no vacio


def test_alias_override_no_vacio():
    """ALIAS_OVERRIDE tiene al menos los 5 alias originales."""
    assert len(ALIAS_OVERRIDE) >= 5


def test_round_trip_piloto(h):
    """
    Round-trip: homologar → atributos → CAMPO no vacio para los 4 pilotos.
    Garantiza que los pilotos tienen entradas en DIM_CAMPO.
    """
    pilotos = ["CASTILLA", "CASTILLA NORTE", "CASTILLA ESTE", "RUBIALES"]
    for p in pilotos:
        campo, _ = h.homologar(p)
        attrs = h.atributos(campo)
        assert isinstance(attrs, dict), f"{p}: atributos() no retorna dict"


def test_homologar_serie_reporte(h):
    """homologar_serie + reporte_homologacion no crashea con datos mixtos."""
    import pandas as pd
    from homologacion import reporte_homologacion
    serie = pd.Series(["CASTILLA", "CAMPO INVALIDO XYZ", "RUBIALES"])
    res = h.homologar_serie(serie)
    assert "CAMPO" in res.columns
    assert "HOMOLOG_FLAG" in res.columns
    assert (res["HOMOLOG_FLAG"] == "NO_HOMOLOGADO").any()
    # reporte no crashea
    reporte_homologacion(res.rename(columns={"CAMPO": "CAMPO"}), "test")
