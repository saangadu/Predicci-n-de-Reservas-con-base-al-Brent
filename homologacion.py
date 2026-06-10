"""
homologacion.py — Capa de homologacion de nombres de campo contra la tabla maestra.

Problema: en Ecopetrol un mismo campo se nombra distinto entre bases de datos
(HIST 1P, Consolidado, Breakeven, Sensibilidad). Sin homologar, los joins por
string fallan silenciosamente y se pierden filas.

Estrategia (decision usuario 2026-06-03):
  1. AUTORIDAD: DIM_CAMPO.xlsx (tabla maestra del Tablero Historico, 900 campos).
     Es el diccionario de nombres CAMPO validos a nivel granular.
  2. ALIAS_OVERRIDE: mapeo manual de variantes que DIM_CAMPO no captura
     (sufijos de yacimiento, alias historicos, filiales). Editable por vigencia.
  3. AUDITORIA: todo nombre que no homologa se marca NO_HOMOLOGADO (no se descarta
     en silencio) para revision.

Uso:
    from homologacion import Homologador
    h = Homologador()                       # carga DIM_CAMPO una sola vez
    campo, flag = h.homologar("LA REFORMA") # -> ("LIBERTAD", "OK") via override
    attrs = h.atributos("CASTILLA")         # -> {UNIFICADO, VICEPRESIDENCIA, ACTIVO, ...}
"""

import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
RAW = BASE_DIR / "datos" / "raw"


# ── Overrides manuales: variante_normalizada -> CAMPO canonico de DIM_CAMPO ──────
# Sembrado desde Consolidado/Hoja1 (columna "Cambios nombre") y los no-match
# detectados en la inspeccion de HIST 1P. Ampliar aqui cuando aparezcan nuevos alias.
ALIAS_OVERRIDE = {
    "LA REFORMA": "LIBERTAD",          # Hoja1: ACTIVO REFORMA agrupa LIBERTAD/LA REFORMA
    "ACAE-SAN MIGUEL (PTO COLON)": "ACAE-SAN MIGUEL",
    "ACAE SAN MIGUEL": "ACAE-SAN MIGUEL",
    "AREA TECA-COCORNA": "TECA",
    "CANAGUEY (COSECHA Y)": "COSECHA",

    # ── Mojibake asimetrico (2026-06-03) ─────────────────────────────────────
    # Excel pierde vocales acentuadas (O/I/U) en la fuente Y pierde N en DIM.
    # Cada lado pierde letras distintas -> ni fold ni fuzzy los puentean.
    # Clave = normalizar_base(nombre_en_fuente); valor = clave exacta en DIM.
    "CANAG\xdcEY (COSECHA Y)": "COSECHA",          # \xdc=U con dieresis; mismo target que clave limpia
    "CA\xd1O LIM\xd3N":  "CA\xd1O LIMON",          # \xd1=N tilde, \xd3=O acento; DIM pierde el acento de O
    "CA\xd1O ROND\xd3N": "CA\xd1O RONDON",
    "CHIPIR\xd3N":        "CHIPIRON",
    "JAZM\xcdN":          "JAZMIN",                 # \xcd=I acento
    "YARIGU\xcd-CANTAGALLO": "YARIGUI-CANTAGALLO",

    # ── N tilde caida a N simple en la fuente (DIM conserva la N tilde) ──────
    "LA CANADA NORTE": "LA CA\xd1ADA NORTE",
    "RIO SALDANA":     "RIO SALDA\xd1A",

    # ── Formato / abreviatura / rollup ────────────────────────────────────────
    "CUPIAGUA RECETOR":  "CUPIAGUA (RECETOR)",  # DIM usa parentesis
    "PAUTO (RECETOR)":   "PAUTO RECETOR",        # DIM sin parentesis
    "GUANDO SW":         "GUANDO SOUTH WEST",
    "TOROS":             "LOS TOROS",
    "MANSOYA UNIFICADO": "MANSOYA",              # rollup -> granular (no existe en DIM)
}

# Sufijos de yacimiento/segmento que cuelgan del nombre de campo y deben removerse
# para homologar contra el campo base (ej. "APIAY ESTE K" -> "APIAY ESTE").
SUFIJOS_YACIMIENTO = (" K", " T")


def _fix_encoding(texto: str) -> str:
    """Repara mojibake latin-1/utf-8 comun en exports de Excel (Ñ, tildes)."""
    if not isinstance(texto, str):
        return ""
    try:
        # Caso tipico: bytes utf-8 mal decodificados como latin-1
        return texto.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto


def normalizar_base(nombre: str) -> str:
    """
    Limpieza base previa a la homologacion: fix encoding + Trim + Upper +
    elimina sufijos de proceso (_CF_SEC_*, extension de archivo, _FILIAL).
    No resuelve alias — eso lo hace homologar().
    """
    if not isinstance(nombre, str):
        return ""
    nombre = _fix_encoding(str(nombre)).strip().upper()
    nombre = re.sub(r"_CF_SEC_.*", "", nombre)   # sufijo de certificacion Solver
    nombre = re.sub(r"\.XLSM?$", "", nombre)     # extension de archivo origen
    nombre = re.sub(r"_FILIAL$", "", nombre)     # marca de filial
    return nombre.strip()


class Homologador:
    """Resuelve nombres crudos -> CAMPO canonico usando DIM_CAMPO + overrides."""

    def __init__(self, ruta_dim: Path = None):
        ruta = ruta_dim or (RAW / "DIM_CAMPO.xlsx")
        dim = pd.read_excel(ruta, sheet_name="DIM_CAMPO", engine="openpyxl")
        dim.columns = [str(c).strip() for c in dim.columns]
        dim["CAMPO_NORM"] = dim["CAMPO"].apply(normalizar_base)
        # Diccionario de nombres validos -> fila de atributos
        self._dim = dim.drop_duplicates("CAMPO_NORM").set_index("CAMPO_NORM")
        self._validos = set(self._dim.index)

    def homologar(self, nombre: str) -> tuple:
        """
        Retorna (campo_canonico, flag) donde flag ∈ {"OK", "NO_HOMOLOGADO"}.
        Orden: normaliza -> override directo -> valida en DIM -> strip sufijo
        yacimiento -> valida -> NO_HOMOLOGADO.
        """
        base = normalizar_base(nombre)
        if not base:
            return "", "NO_HOMOLOGADO"

        # 1. Override explicito
        if base in ALIAS_OVERRIDE:
            return ALIAS_OVERRIDE[base], "OK"

        # 2. Match directo contra la tabla maestra
        if base in self._validos:
            return base, "OK"

        # 3. Quitar sufijo de yacimiento (K/T) e intentar de nuevo
        for suf in SUFIJOS_YACIMIENTO:
            if base.endswith(suf):
                candidato = base[: -len(suf)].strip()
                if candidato in ALIAS_OVERRIDE:
                    return ALIAS_OVERRIDE[candidato], "OK"
                if candidato in self._validos:
                    return candidato, "OK"

        # 4. No se pudo homologar — devolver normalizado y marcar para auditoria
        return base, "NO_HOMOLOGADO"

    def atributos(self, campo_canonico: str) -> dict:
        """Devuelve UNIFICADO/VICEPRESIDENCIA/ACTIVO/... del campo (vacio si no existe)."""
        if campo_canonico in self._dim.index:
            return self._dim.loc[campo_canonico].to_dict()
        return {}

    def homologar_serie(self, serie: pd.Series) -> pd.DataFrame:
        """
        Homologa una columna completa. Retorna DataFrame con columnas
        CAMPO (canonico) y HOMOLOG_FLAG, alineado al indice de la serie.
        """
        res = serie.apply(lambda x: pd.Series(self.homologar(x),
                                              index=["CAMPO", "HOMOLOG_FLAG"]))
        return res


def reporte_homologacion(df: pd.DataFrame, origen: str) -> None:
    """Imprime los nombres que no se pudieron homologar en una fuente (auditoria)."""
    if "HOMOLOG_FLAG" not in df.columns:
        return
    col_raw = "CAMPO_ORIGEN_RAW" if "CAMPO_ORIGEN_RAW" in df.columns else "CAMPO"
    no_hom = df.loc[df["HOMOLOG_FLAG"] == "NO_HOMOLOGADO", col_raw].unique()
    n = len(no_hom)
    print(f"  [{origen}] no homologados: {n}")
    if n > 0:
        print(f"    {sorted(map(str, no_hom))[:20]}")


if __name__ == "__main__":
    # Modo test: verifica homologacion de campos piloto + alias conocidos
    h = Homologador()
    print(f"DIM_CAMPO cargada: {len(h._validos)} campos validos\n")
    pruebas = ["CASTILLA", "CASTILLA ESTE", "CASTILLA NORTE", "RUBIALES",
               "LA REFORMA", "LIBERTAD", "APIAY ESTE K", "Castilla_CF_SEC_23-Jan-2025.xlsx",
               "ARRECIFE_FILIAL", "CAMPO INEXISTENTE XYZ"]
    for p in pruebas:
        campo, flag = h.homologar(p)
        uni = h.atributos(campo).get("UNIFICADO", "—")
        print(f"  {p:<40} -> {campo:<22} [{flag}] uni={uni}")
