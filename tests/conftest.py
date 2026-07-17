"""
tests/conftest.py — Garantiza que la raiz del proyecto este en sys.path.

Los paths por track (Produccion vs Calidad) viven en rutas_track.py en la raiz del
proyecto: los tests hacen `from rutas_track import STAGING, RESULTADOS, ...`.
pytest carga este conftest ANTES de importar cualquier modulo de test, asi que la
raiz ya esta disponible aunque pytest se invoque desde otro directorio.
"""
import sys
from pathlib import Path

# Raiz del proyecto: el directorio padre de tests/ (donde viven los scripts 01-05)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
