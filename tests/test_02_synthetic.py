"""
test_02_synthetic.py — Gate de la Fase 2 (inyección de sintéticos, escalera financiero/operacional)

Convención finanzas (2026-06-09): BK_ANCLA_FIN = piso SUPERIOR (financiero, delta=0),
BK_ANCLA_PDP = piso INFERIOR (operacional, abandono). BK_ANCLA_FIN > BK_ANCLA_PDP.

Verifica que la lógica de dos tramos de anclaje físico sea correcta:
  Tramo BAJO:     [BK_ANCLA_PDP - RANGO, BK_ANCLA_PDP) → VOLUMEN_1P = 0 (todas las reservas)
  Tramo ESCALERA: [BK_ANCLA_PDP, BK_ANCLA_FIN)         → VOLUMEN_1P = BASELINE_PDP (solo PDP sobrevive)

Campos BRENT_INSENSITIVE deben quedar fuera del anclaje (sin filas sintéticas).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib
syn = importlib.import_module("02_synthetic")   # noqa: E402  (nombre empieza con número)

BASE    = Path(__file__).resolve().parent.parent
STAGING = BASE / "datos" / "staging"


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _tablon_minimal(campo: str, bk_sup: float, bk_inf: float,
                    brent_insens: bool = False,
                    baseline_total: float = 500.0,
                    baseline_pdp: float = 200.0,
                    baseline_pnp: float = 300.0) -> pd.DataFrame:
    """
    Tablón mínimo de un campo para probar generar_sinteticos.
    bk_sup = BK_ANCLA_FIN_USD_BBL (financiero, piso superior, delta=0).
    bk_inf = BK_ANCLA_PDP_USD_BBL (operacional, piso inferior, abandono).
    Dos filas BASE (escenario real) con los breakevens y baselines.
    """
    brent = 60.0
    cal   = -5.0
    tra   = -3.0
    pneto = brent + cal + tra   # = 52.0

    filas = []
    for anio in [2023, 2024]:
        filas.append({
            "CAMPO":                          campo,
            "CAMPO_ORIGEN_RAW":               campo,
            "AÑO":                            anio,
            "VIGENCIA":                       str(anio),
            "ESCENARIO":                      "BASE",
            "ES_BASELINE":                    True,
            "ES_SINTETICO":                   False,
            "NIVEL_DEFINICIONAL":             "",
            "BRENT_FLAT_USD_BBL":             brent,
            "DESCUENTO_CALIDAD_USD_BBL":      cal,
            "DESCUENTO_TRANSPORTE_USD_BBL":   tra,
            "PRECIO_NETO_USD_BBL":            pneto,
            "VOLUMEN_PDP_MBPE":               baseline_pdp if anio == 2024 else np.nan,
            "VOLUMEN_PNP_MBPE":               (baseline_pnp * 0.7) if anio == 2024 else np.nan,
            "VOLUMEN_PND_MBPE":               (baseline_pnp * 0.3) if anio == 2024 else np.nan,
            "VOLUMEN_1P_OFICIAL_MBPE":        baseline_total if anio == 2024 else np.nan,
            "BASELINE_1P_VIGENCIA_MBPE":      baseline_total,
            "VOLUMEN_1P_SENSIBILIDAD_MBPE":   np.nan,
            "DELTA_SENS_MBPE":                np.nan,
            "BREAKEVEN_FINANCIERO_USD_BBL":   bk_sup,
            "BREAKEVEN_OPERACIONAL_USD_BBL":  bk_inf,
            "BK_ANCLA_FIN_USD_BBL":           bk_sup,
            "BK_ANCLA_PDP_USD_BBL":           bk_inf,
            "BRENT_INSENSITIVE":              brent_insens,
            "VIGENCIA_BREAKEVEN":             "2024",
            "PRED_XGBOOST_MBPE":              np.nan,
            "PRED_ISOTONICA_MBPE":            np.nan,
            "DELTA_XGBOOST_VS_OFICIAL":       np.nan,
            "DELTA_ISOTONICA_VS_OFICIAL":      np.nan,
            "ALERTA":                         "",
            "HOMOLOG_FLAG":                   "OK",
        })
    return pd.DataFrame(filas)


def _run_sinteticos(df: pd.DataFrame):
    """Ejecuta generar_sinteticos con medianas y baselines calculados."""
    medianas         = syn.calcular_medianas_precio(df)
    baselines        = syn.calcular_baselines_latest(df)
    baselines_clase  = syn.calcular_baselines_por_clase(df)
    return syn.generar_sinteticos(df, medianas, baselines, baselines_clase)


# ─── Constantes ─────────────────────────────────────────────────────────────────

def test_rango_usd_es_5():
    """RANGO_USD se redujo de 20 a 5 (2026-06-09) para no opacar reales en entrenamiento."""
    assert syn.RANGO_USD == 5


# ─── Tests de lógica de dos tramos ─────────────────────────────────────────────

def test_bajo_vol_cero():
    """Tramo BAJO: todos los puntos tienen VOLUMEN_1P_SENSIBILIDAD = 0."""
    bk_sup, bk_inf = 68.0, 25.0
    df = _tablon_minimal("CAMPO_A", bk_sup, bk_inf)
    sint = _run_sinteticos(df)
    assert len(sint) > 0
    bajo = sint[sint["PRECIO_NETO_USD_BBL"] < bk_inf]
    assert len(bajo) > 0, "Tramo BAJO debe tener filas"
    assert (bajo["VOLUMEN_1P_SENSIBILIDAD_MBPE"] == 0.0).all(), \
        "Tramo BAJO: vol 1P debe ser 0 en todos los puntos"


def test_escalera_vol_pdp():
    """Tramo ESCALERA: puntos entre bk_inf y bk_sup tienen VOLUMEN_1P = BASELINE_PDP."""
    bk_sup, bk_inf = 68.0, 25.0
    baseline_pdp = 200.0
    df = _tablon_minimal("CAMPO_B", bk_sup, bk_inf, baseline_pdp=baseline_pdp)
    sint = _run_sinteticos(df)
    esc = sint[(sint["PRECIO_NETO_USD_BBL"] >= bk_inf) &
               (sint["PRECIO_NETO_USD_BBL"] < bk_sup)]
    assert len(esc) > 0, "Tramo ESCALERA debe existir cuando bk_sup > bk_inf + PASO y PDP>0"
    assert np.allclose(esc["VOLUMEN_1P_SENSIBILIDAD_MBPE"].values, baseline_pdp), \
        f"Tramo ESCALERA vol esperado {baseline_pdp}, recibido {esc['VOLUMEN_1P_SENSIBILIDAD_MBPE'].unique()}"


def test_escalera_delta_pdp():
    """Tramo ESCALERA: DELTA_SENS = -(BASELINE_TOTAL - BASELINE_PDP) (solo se pierde PNP+PND)."""
    bk_sup, bk_inf = 68.0, 25.0
    baseline_total = 500.0
    baseline_pdp   = 200.0
    df = _tablon_minimal("CAMPO_C", bk_sup, bk_inf,
                         baseline_total=baseline_total, baseline_pdp=baseline_pdp)
    sint = _run_sinteticos(df)
    esc = sint[(sint["PRECIO_NETO_USD_BBL"] >= bk_inf) &
               (sint["PRECIO_NETO_USD_BBL"] < bk_sup)]
    assert len(esc) > 0
    assert np.allclose(esc["DELTA_SENS_MBPE"].values, -(baseline_total - baseline_pdp))


def test_bajo_delta_total():
    """Tramo BAJO: DELTA_SENS = -BASELINE_TOTAL (pérdida de todo el 1P)."""
    bk_sup, bk_inf = 68.0, 25.0
    baseline_total = 500.0
    df = _tablon_minimal("CAMPO_D", bk_sup, bk_inf, baseline_total=baseline_total)
    sint = _run_sinteticos(df)
    bajo = sint[sint["PRECIO_NETO_USD_BBL"] < bk_inf]
    assert len(bajo) > 0
    assert np.allclose(bajo["DELTA_SENS_MBPE"].values, -baseline_total)


def test_brent_insensitive_sin_sinteticos():
    """Campos BRENT_INSENSITIVE no deben tener filas sintéticas."""
    df = _tablon_minimal("CAMPO_INSENS", 68.0, 25.0, brent_insens=True)
    sint = _run_sinteticos(df)
    assert len(sint) == 0, \
        "Campo BRENT_INSENSITIVE no debe tener ninguna fila sintética"


def test_escalera_ausente_sin_brecha():
    """Si bk_sup <= bk_inf + PASO, no se genera tramo ESCALERA (sin brecha real entre pisos)."""
    bk_inf = 25.0
    bk_sup = bk_inf + 0.5   # brecha < PASO_USD (1 USD)
    df = _tablon_minimal("CAMPO_SIN_ESC", bk_sup, bk_inf)
    sint = _run_sinteticos(df)
    esc = sint[(sint["PRECIO_NETO_USD_BBL"] >= bk_inf) &
               (sint["PRECIO_NETO_USD_BBL"] < bk_sup)]
    assert len(esc) == 0, "Sin brecha real → tramo ESCALERA debe estar vacío"


def test_escalera_ausente_si_pdp_cero():
    """Si BASELINE_PDP=0 (campo solo PNP/PND), tramo ESCALERA no se genera (sería igual a BAJO)."""
    bk_sup, bk_inf = 68.0, 25.0
    df = _tablon_minimal("CAMPO_SIN_PDP", bk_sup, bk_inf, baseline_pdp=0.0)
    sint = _run_sinteticos(df)
    esc = sint[(sint["PRECIO_NETO_USD_BBL"] >= bk_inf) &
               (sint["PRECIO_NETO_USD_BBL"] < bk_sup)]
    assert len(esc) == 0, "PDP=0 → tramo ESCALERA debe estar vacío"


def test_precio_neto_siempre_menor_que_bk_sup():
    """Todo punto sintético tiene precio neto < BK_ANCLA_FIN (jamás dentro de la zona rentable)."""
    bk_sup, bk_inf = 68.0, 25.0
    df = _tablon_minimal("CAMPO_E", bk_sup, bk_inf)
    sint = _run_sinteticos(df)
    assert (sint["PRECIO_NETO_USD_BBL"] < bk_sup).all(), \
        "Todos los sintéticos deben estar por debajo del piso financiero (delta=0)"


def test_es_sintetico_true():
    """Todas las filas generadas tienen ES_SINTETICO=True."""
    df = _tablon_minimal("CAMPO_F", 68.0, 25.0)
    sint = _run_sinteticos(df)
    assert len(sint) > 0
    assert sint["ES_SINTETICO"].all()


def test_brent_implicito_consistente():
    """BRENT = PRECIO_NETO - DESCUENTO_CALIDAD - DESCUENTO_TRANSPORTE (descuentos negativos)."""
    df = _tablon_minimal("CAMPO_G", 68.0, 25.0)
    sint = _run_sinteticos(df)
    assert len(sint) > 0
    brent_recalc = (sint["PRECIO_NETO_USD_BBL"]
                    - sint["DESCUENTO_CALIDAD_USD_BBL"]
                    - sint["DESCUENTO_TRANSPORTE_USD_BBL"])
    pd.testing.assert_series_equal(
        sint["BRENT_FLAT_USD_BBL"].round(4),
        brent_recalc.round(4),
        check_names=False,
    )


# ─── Gate de integración (si el tablón ya fue generado) ────────────────────────

def test_tablon_sinteticos_schema():
    """Si tablon_unico.parquet existe con sintéticos: esquema y lógica de anclas."""
    pq = STAGING / "tablon_unico.parquet"
    if not pq.exists():
        pytest.skip("tablon_unico.parquet no encontrado (correr 01_etl.py y 02_synthetic.py)")

    df = pd.read_parquet(pq)
    sint = df[df["ES_SINTETICO"]]
    if len(sint) == 0:
        pytest.skip("No hay sintéticos en el tablón (correr 02_synthetic.py)")

    # Columnas requeridas
    reqs = {"ES_SINTETICO", "PRECIO_NETO_USD_BBL", "VOLUMEN_1P_SENSIBILIDAD_MBPE",
            "DELTA_SENS_MBPE", "BK_ANCLA_FIN_USD_BBL", "BK_ANCLA_PDP_USD_BBL",
            "BRENT_INSENSITIVE"}
    assert reqs.issubset(df.columns), f"Columnas faltantes: {reqs - set(df.columns)}"

    # Tramo BAJO: precio < BK_ANCLA_PDP → vol=0
    bajo = sint[sint["PRECIO_NETO_USD_BBL"] < sint["BK_ANCLA_PDP_USD_BBL"]]
    if len(bajo) > 0:
        assert (bajo["VOLUMEN_1P_SENSIBILIDAD_MBPE"] == 0.0).all(), \
            "Tramo BAJO real: vol no es 0"

    # Tramo ESCALERA: precio en [BK_ANCLA_PDP, BK_ANCLA_FIN) → vol > 0 (solo PDP)
    esc = sint[(sint["PRECIO_NETO_USD_BBL"] >= sint["BK_ANCLA_PDP_USD_BBL"]) &
               (sint["PRECIO_NETO_USD_BBL"] <  sint["BK_ANCLA_FIN_USD_BBL"])]
    if len(esc) > 0:
        assert (esc["VOLUMEN_1P_SENSIBILIDAD_MBPE"] > 0).all(), \
            "Tramo ESCALERA real: vol debe ser > 0 (PDP)"

    # Ningún sintético cae en zona rentable (precio >= BK_ANCLA_FIN)
    sobre_piso = sint[sint["PRECIO_NETO_USD_BBL"] >= sint["BK_ANCLA_FIN_USD_BBL"]]
    assert len(sobre_piso) == 0, \
        f"{len(sobre_piso)} sintéticos con precio >= BK_ANCLA_FIN (no deberían existir)"

    # BK_ANCLA_FIN > BK_ANCLA_PDP (o ESCALERA_DEGENERADA con ALERTA correspondiente)
    anclas = sint.dropna(subset=["BK_ANCLA_FIN_USD_BBL", "BK_ANCLA_PDP_USD_BBL"])
    invertidas = anclas[anclas["BK_ANCLA_FIN_USD_BBL"] < anclas["BK_ANCLA_PDP_USD_BBL"]]
    assert invertidas.empty, \
        f"{len(invertidas)} filas sinteticas con BK_ANCLA_FIN < BK_ANCLA_PDP"
