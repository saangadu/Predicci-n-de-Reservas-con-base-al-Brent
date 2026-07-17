"""Gates de defaults de track.flag / FLAGS_ON_EN_CALIDAD.

Contrato s14 (promocion 2026-07-17): PRED_M2_SELECCION ratificado -> ON por
default en TODO track (antes s12: solo Calidad); override explicito siempre gana.
"""
import os

import pytest

from track import FLAGS_ON_EN_CALIDAD, FLAGS_RATIFICADOS, es_track_calidad, flag


@pytest.fixture(autouse=True)
def _limpiar_env_flags(monkeypatch):
    """Cada test parte sin PRED_TRACK ni flags, para no heredar el shell del CI."""
    monkeypatch.delenv("PRED_TRACK", raising=False)
    for nombre in (*FLAGS_RATIFICADOS, *FLAGS_ON_EN_CALIDAD, "PRED_REANCLAJE_CONFOUND"):
        monkeypatch.delenv(nombre, raising=False)


def test_produccion_m2_seleccion_on_por_default():
    # s14: PRED_M2_SELECCION ratificado -> ON tambien en Produccion
    assert flag("PRED_M2_SELECCION") is True
    assert es_track_calidad() is False


def test_calidad_m2_seleccion_auto_on(monkeypatch):
    # Footgun s12: solo PRED_TRACK=calidad basta para prender el dispatch M2
    monkeypatch.setenv("PRED_TRACK", "calidad")
    assert es_track_calidad() is True
    assert flag("PRED_M2_SELECCION") is True


def test_calidad_m2_seleccion_override_off(monkeypatch):
    # --sin-flags / override explicito debe apagar incluso en Calidad
    monkeypatch.setenv("PRED_TRACK", "calidad")
    monkeypatch.setenv("PRED_M2_SELECCION", "0")
    assert flag("PRED_M2_SELECCION") is False


def test_calidad_m2_seleccion_override_on(monkeypatch):
    monkeypatch.setenv("PRED_TRACK", "calidad")
    monkeypatch.setenv("PRED_M2_SELECCION", "1")
    assert flag("PRED_M2_SELECCION") is True


def test_ratificados_siguen_on_sin_env():
    # Contrato promocion s10: ratificados ON en cualquier track si no hay env
    for nombre in FLAGS_RATIFICADOS:
        assert flag(nombre) is True
