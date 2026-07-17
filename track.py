"""track.py — Sufijo de track (Calidad vs Produccion) via variables de entorno.

Track Calidad (ver docs/DIRECTRIZ_ESCALERA_DECK.md; ratificacion Finanzas pendiente
para el registro de seleccion — ver MAESTRO §10):
  - PRED_TRACK=calidad  -> staging y resultados van a directorios paralelos
    (datos/staging_calidad, resultados_calidad). Produccion (sin la variable)
    usa los mismos flags de comportamiento (promocion 2026-07-15).
  - Flags RATIFICADOS (default ON en TODOS los tracks desde 2026-07-15;
    apagado explicito con =0, p.ej. --sin-flags para paridad legacy):
      PRED_PERFIL_SALIDA        -> perfil de salida volumetrico (directriz §4bis.6)
      PRED_SELECCION_METODO     -> seleccion de metodo por campo (solo LISAMA/reanclaje
                                    adoptado; ver resultados_calidad/seleccion_metodos.csv)
      PRED_BANDAS_LOYO          -> bandas de incertidumbre P10/P90 (residuales LOYO)
  - Flags ON solo en track Calidad (default OFF en Produccion; no ratificados):
      PRED_M2_SELECCION         -> dispatch de familias M2 por campo
                                    (seleccion_metodos_m2.csv). Auto-ON si
                                    PRED_TRACK=calidad (s12); override explicito
                                    =0 apaga incluso en Calidad (--sin-flags).
  - Flags NO ratificados (default OFF siempre):
      PRED_REANCLAJE_CONFOUND   -> re-anclaje BLANKET por confound (directriz §4bis.7.1;
                                    superseded por PRED_SELECCION_METODO, s9)

Un flag nuevo NO se prende en Produccion hasta que Finanzas lo ratifique.
"""
import os

# Default ON en Produccion y Calidad (promociones 2026-07-15 s10 y 2026-07-17 s13).
FLAGS_RATIFICADOS = {
    "PRED_PERFIL_SALIDA",
    "PRED_SELECCION_METODO",
    "PRED_BANDAS_LOYO",
    "PRED_M2_SELECCION",     # s13: dispatch familias M2 ratificado
}

# Default ON solo cuando PRED_TRACK=calidad (investigacion no ratificada).
FLAGS_ON_EN_CALIDAD = set()


def sufijo_track() -> str:
    """'' en Produccion; '_calidad' (u otro) si PRED_TRACK esta seteado."""
    t = os.environ.get("PRED_TRACK", "").strip()
    return f"_{t}" if t else ""


def es_track_calidad() -> bool:
    """True si el proceso corre en el track Calidad (PRED_TRACK=calidad)."""
    return os.environ.get("PRED_TRACK", "").strip().lower() == "calidad"


def flag(nombre: str) -> bool:
    """True/False segun la variable de entorno (1/true/yes).

    Si NO esta seteada:
      - ON  si esta en FLAGS_RATIFICADOS (todos los tracks)
      - ON  si esta en FLAGS_ON_EN_CALIDAD y PRED_TRACK=calidad
      - OFF en cualquier otro caso (flags experimentales / Produccion)

    Override explicito (=0 / =1) siempre gana — p.ej. --sin-flags fuerza "0".
    """
    valor = os.environ.get(nombre, "").strip().lower()
    if valor == "":
        if nombre in FLAGS_RATIFICADOS:
            return True
        # Calidad prende investigaciones no ratificadas sin exigir env manual
        if nombre in FLAGS_ON_EN_CALIDAD and es_track_calidad():
            return True
        return False
    return valor in ("1", "true", "yes")
