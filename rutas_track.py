"""
rutas_track.py — Paths de artefactos parametrizados por TRACK (Produccion vs Calidad).

Paridad de rigor (directriz usuario 2026-07-15): el track Calidad debe correr los MISMOS
gates pytest que Produccion. Antes los tests tenian paths hardcodeados a datos/staging/
y resultados/, por lo que run_pipeline --calidad los omitia por completo (asi fue
invisible el bug de MACANA en s10). Este modulo centraliza los paths y los deriva del
track activo (variable de entorno PRED_TRACK via track.sufijo_track()):

    Produccion (sin PRED_TRACK):  datos/staging/,          resultados/
    Calidad (PRED_TRACK=calidad): datos/staging_calidad/,  resultados_calidad/

Las BASELINES CONGELADAS del contrato NORTE (norte_baseline.csv y
norte_etiquetas_baseline.csv) viven SIEMPRE en resultados/ de Produccion: son el punto
de referencia contra el que Calidad se compara (RESULTADOS_PROD, sin sufijo).

Vive en la raiz del proyecto (no en tests/) porque pytest>=9 importa los tests en modo
'importlib' y un conftest.py no es importable como modulo; tests/conftest.py garantiza
que la raiz este en sys.path para que este modulo resuelva desde cualquier test.
"""
from pathlib import Path

# Raiz del proyecto: el directorio donde vive este modulo (junto a los scripts 01-05)
ROOT = Path(__file__).resolve().parent

from track import sufijo_track  # noqa: E402  (import local, mismo directorio)

# Sufijo del track activo: '' en Produccion, '_calidad' con PRED_TRACK=calidad
_SUF = sufijo_track()

# True cuando los gates corren sobre los artefactos del track Calidad
ES_CALIDAD = _SUF != ""

# Artefactos intermedios del pipeline (tablon, metricas, modelos, correlacion M2)
STAGING = ROOT / "datos" / f"staging{_SUF}"

# Salidas finales para Power BI (matrices, metricas espejo, norte_*)
RESULTADOS = ROOT / f"resultados{_SUF}"

# Modelos serializados por campo ({campo}_iso.joblib / {campo}_suave.joblib)
MODELOS_DIR = STAGING / "modelos"

# Baselines congeladas del contrato NORTE: SIEMPRE de Produccion (ver docstring)
RESULTADOS_PROD = ROOT / "resultados"
