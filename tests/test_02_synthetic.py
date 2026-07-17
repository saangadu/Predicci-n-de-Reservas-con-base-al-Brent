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

BASE = Path(__file__).resolve().parent.parent
# Paths por track (Produccion vs Calidad): centralizados en tests/conftest.py
from rutas_track import STAGING  # noqa: E402


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _tablon_minimal(campo: str, bk_sup: float, bk_inf: float,
                    brent_insens: bool = False,
                    baseline_total: float = 500.0,
                    baseline_pdp: float = 200.0,
                    baseline_pnp: float = 300.0,
                    salida_pnp: float = np.nan,
                    salida_pnd: float = np.nan,
                    deltas_reales: list = None) -> pd.DataFrame:
    """
    Tablón mínimo de un campo para probar generar_sinteticos.
    bk_sup = BK_ANCLA_FIN_USD_BBL (financiero, piso superior).
    bk_inf = BK_ANCLA_PDP_USD_BBL (operacional, piso inferior, abandono).
    salida_pnp/pnd = BK_SALIDA_*_USD_BBL (limite economico por clase, Path D).
    deltas_reales = [(pneto, delta), ...] puntos CONSOLIDADO para el cap §4.3.
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
            "CHECKPOINT_1P_MBPE":             baseline_total,
            "BASELINE_1P_MBPE":               baseline_total,
            "VOLUMEN_1P_SENSIBILIDAD_MBPE":   np.nan,
            "DELTA_SENS_MBPE":                np.nan,
            "BREAKEVEN_USD_BBL":              bk_sup,
            "PRECIO_EQUILIBRIO_USD_BBL":     bk_inf,
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
    df = pd.DataFrame(filas)
    df["BK_SALIDA_PNP_USD_BBL"] = salida_pnp
    df["BK_SALIDA_PND_USD_BBL"] = salida_pnd

    # Puntos reales CONSOLIDADO (para el cap de monotonia §4.3)
    extra = []
    for pn_r, delta_r in (deltas_reales or []):
        fila = dict(df.iloc[-1])
        fila.update({
            "ESCENARIO":                    "CONSOLIDADO_2024_Q1",
            "ES_BASELINE":                  False,
            "PRECIO_NETO_USD_BBL":          pn_r,
            "BRENT_FLAT_USD_BBL":           pn_r - cal - tra,
            "VOLUMEN_1P_OFICIAL_MBPE":      np.nan,
            "VOLUMEN_1P_SENSIBILIDAD_MBPE": baseline_total + delta_r,
            "DELTA_SENS_MBPE":              delta_r,
        })
        extra.append(fila)
    if extra:
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    return df


def _run_sinteticos(df: pd.DataFrame):
    """Ejecuta generar_sinteticos con medianas y baselines calculados."""
    medianas         = syn.calcular_medianas_precio(df)
    baselines        = syn.calcular_baselines_latest(df)
    baselines_clase  = syn.calcular_baselines_por_clase(df)
    descuentos_cert  = syn.calcular_descuentos_cert(df)
    min_delta_real   = syn.calcular_min_delta_real(df)
    banda_real_lo    = syn.calcular_banda_real_lo(df)
    return syn.generar_sinteticos(df, medianas, baselines, baselines_clase,
                                  descuentos_cert, min_delta_real, banda_real_lo)


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


def test_escalera_pdp_cero_vol_cero():
    """Si BASELINE_PDP=0 (campo solo PNP/PND), la escalera entre el abandono y las
    salidas de clase mantiene vol=0 (nada sobrevive sin PDP) — Path D 2026-06-11."""
    bk_sup, bk_inf = 68.0, 25.0
    df = _tablon_minimal("CAMPO_SIN_PDP", bk_sup, bk_inf, baseline_pdp=0.0)
    sint = _run_sinteticos(df)
    esc = sint[(sint["PRECIO_NETO_USD_BBL"] >= bk_inf) &
               (sint["PRECIO_NETO_USD_BBL"] < bk_sup)]
    if len(esc) > 0:
        assert (esc["VOLUMEN_1P_SENSIBILIDAD_MBPE"] == 0.0).all(), \
            "PDP=0 → la escalera bajo las salidas de clase debe anclar vol=0"


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


# ─── Tests escalera multi-clase (Path D, 2026-06-11) ───────────────────────────

def test_multiclase_orden_por_precio():
    """El orden de salida lo da el PRECIO, no la clase (caso RUBIALES 2025:
    PNP sale a 47.8 ANTES que PND a 45.4 al bajar el precio)."""
    bk_sup, bk_inf = 50.0, 25.0
    # PNP sale mas ARRIBA (47) que PND (40): al subir el precio entra PND primero
    df = _tablon_minimal("CAMPO_MC", bk_sup, bk_inf,
                         baseline_total=500.0, baseline_pdp=200.0,
                         baseline_pnp=300.0,   # PNP=210, PND=90 (70/30)
                         salida_pnp=47.0, salida_pnd=40.0)
    sint = _run_sinteticos(df)
    esc = sint[sint["PRECIO_NETO_USD_BBL"] >= bk_inf]
    # Tramo solo-PDP: [25, 40) → 200
    t1 = esc[esc["PRECIO_NETO_USD_BBL"] < 40.0]
    assert np.allclose(t1["VOLUMEN_1P_SENSIBILIDAD_MBPE"], 200.0)
    # Tramo PDP+PND: [40, 47) → 200 + 90 = 290 (PND entra antes que PNP)
    t2 = esc[(esc["PRECIO_NETO_USD_BBL"] >= 40.0) & (esc["PRECIO_NETO_USD_BBL"] < 47.0)]
    assert np.allclose(t2["VOLUMEN_1P_SENSIBILIDAD_MBPE"], 290.0)
    # Nada en/sobre la salida mas alta (47): la banda real gobierna
    assert (sint["PRECIO_NETO_USD_BBL"] < 47.0).all()


def test_cap_monotonia_escalon():
    """Cap §4.3: escalon nunca por encima del peor delta real del campo
    (caso CASTILLA NORTE: escalon -20.6 vs min delta real -27.2 → capa)."""
    bk_sup, bk_inf = 50.0, 25.0
    # Escalon PDP+PND = 200+90=290 → delta -210; peor real = -250 → capa a -250
    df = _tablon_minimal("CAMPO_CAP", bk_sup, bk_inf,
                         baseline_total=500.0, baseline_pdp=200.0,
                         baseline_pnp=300.0,
                         salida_pnp=47.0, salida_pnd=40.0,
                         deltas_reales=[(60.0, -250.0), (62.0, -240.0)])
    sint = _run_sinteticos(df)
    esc = sint[sint["PRECIO_NETO_USD_BBL"] >= bk_inf]
    assert (esc["DELTA_SENS_MBPE"] <= -250.0 + 1e-9).all(), \
        "Ningun escalon puede superar el peor delta real (-250)"
    capados = esc[esc["ALERTA"] == "ESCALON_CAPADO"]
    assert len(capados) > 0, "El tramo PDP+PND (delta -210 > -250) debe quedar capado"
    # El tramo solo-PDP (delta -300 < -250) no se capa
    t1 = esc[esc["PRECIO_NETO_USD_BBL"] < 40.0]
    assert (t1["ALERTA"] == "").all()


def test_degradacion_sin_salida_clase():
    """§4.5: clase con volumen pero sin limite economico propio sale en
    BK_ANCLA_FIN (diseño anterior) — escalera de 2 tramos."""
    bk_sup, bk_inf = 68.0, 25.0
    df = _tablon_minimal("CAMPO_DEG", bk_sup, bk_inf,
                         salida_pnp=np.nan, salida_pnd=np.nan)
    sint = _run_sinteticos(df)
    esc = sint[sint["PRECIO_NETO_USD_BBL"] >= bk_inf]
    # Sin salidas propias: todo el tramo [25, 68) es solo-PDP (PNP/PND salen en 68)
    assert np.allclose(esc["VOLUMEN_1P_SENSIBILIDAD_MBPE"], 200.0)
    assert (sint["PRECIO_NETO_USD_BBL"] < bk_sup).all()


def test_salida_fusionada_con_abandono():
    """§4.5: salida de clase <= abandono → la clase se fusiona con el abandono
    (vive en toda la escalera, sin escalon propio)."""
    bk_sup, bk_inf = 50.0, 25.0
    df = _tablon_minimal("CAMPO_FUS", bk_sup, bk_inf,
                         baseline_total=500.0, baseline_pdp=200.0,
                         baseline_pnp=300.0,
                         salida_pnp=20.0, salida_pnd=40.0)   # PNP sale BAJO el abandono
    sint = _run_sinteticos(df)
    esc = sint[sint["PRECIO_NETO_USD_BBL"] >= bk_inf]
    # [25, 40): PDP + PNP (fusionada) = 200 + 210 = 410
    t1 = esc[esc["PRECIO_NETO_USD_BBL"] < 40.0]
    assert np.allclose(t1["VOLUMEN_1P_SENSIBILIDAD_MBPE"], 410.0)


def test_descuentos_cert_de_vigencia():
    """D5: el Brent implicito usa los descuentos certificados de la vigencia del
    breakeven (no las medianas historicas) cuando estan disponibles."""
    df = _tablon_minimal("CAMPO_D5", 68.0, 25.0)
    # Vigencia BK = 2024; alterar el certificado 2024 para diferenciarlo de la mediana
    m24 = (df["AÑO"] == 2024) & (df["ESCENARIO"] == "BASE")
    df.loc[m24, "DESCUENTO_CALIDAD_USD_BBL"]    = -9.0
    df.loc[m24, "DESCUENTO_TRANSPORTE_USD_BBL"] = -6.0
    sint = _run_sinteticos(df)
    assert len(sint) > 0
    assert np.allclose(sint["DESCUENTO_CALIDAD_USD_BBL"], -9.0)
    assert np.allclose(sint["DESCUENTO_TRANSPORTE_USD_BBL"], -6.0)
    brent_recalc = (sint["PRECIO_NETO_USD_BBL"] + 9.0 + 6.0)
    assert np.allclose(sint["BRENT_FLAT_USD_BBL"], brent_recalc)


def test_guard_banda_real():
    """Guard 2026-06-11: ningun sintetico se inyecta en/sobre la banda de datos
    certificados (los reales gobiernan su banda; un ancla 'el libro salio' dentro
    de la banda contradice los puntos certificados)."""
    bk_sup, bk_inf = 80.0, 70.0   # anclas EXTREMAS, dentro de la banda real
    df = _tablon_minimal("CAMPO_GB", bk_sup, bk_inf,
                         deltas_reales=[(68.0, -30.0), (75.0, -10.0)])
    sint = _run_sinteticos(df)
    # banda_lo = 68 → tope = 67: nada en/sobre 67
    if len(sint) > 0:
        assert (sint["PRECIO_NETO_USD_BBL"] < 67.0 + 1e-9).all(), \
            "Sinteticos invadiendo la banda real (>= banda_lo - margen)"


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

    # Escalera multi-clase: precio en [BK_ANCLA_PDP, BK_ANCLA_FIN) → vol >= 0
    # (con PDP>0 es el nivel solo-PDP; con PDP=0 el piso cero se extiende)
    esc = sint[(sint["PRECIO_NETO_USD_BBL"] >= sint["BK_ANCLA_PDP_USD_BBL"]) &
               (sint["PRECIO_NETO_USD_BBL"] <  sint["BK_ANCLA_FIN_USD_BBL"])]
    if len(esc) > 0:
        assert (esc["VOLUMEN_1P_SENSIBILIDAD_MBPE"] >= 0).all(), \
            "Escalera real: vol no puede ser negativo"

    # Monotonia de la escalera por campo: vol no decrece al subir el precio
    for campo, g in sint.groupby("CAMPO"):
        g = g.sort_values("PRECIO_NETO_USD_BBL")
        vols = g["VOLUMEN_1P_SENSIBILIDAD_MBPE"].values
        assert (np.diff(vols) >= -1e-9).all(), \
            f"{campo}: escalera sintetica no monotona"

    # Ningún sintético por encima de la salida mas alta del libro (techo Path D:
    # max de BK_ANCLA_FIN y las salidas por clase de la fila)
    techo = sint["BK_ANCLA_FIN_USD_BBL"]
    for col in ["BK_SALIDA_PNP_USD_BBL", "BK_SALIDA_PND_USD_BBL"]:
        if col in sint.columns:
            techo = np.fmax(techo, sint[col].fillna(-np.inf))
    sobre_techo = sint[sint["PRECIO_NETO_USD_BBL"] >= techo]
    assert len(sobre_techo) == 0, \
        f"{len(sobre_techo)} sintéticos con precio >= salida mas alta (no deberían existir)"

    # BK_ANCLA_FIN > BK_ANCLA_PDP (o ESCALERA_DEGENERADA con ALERTA correspondiente)
    anclas = sint.dropna(subset=["BK_ANCLA_FIN_USD_BBL", "BK_ANCLA_PDP_USD_BBL"])
    invertidas = anclas[anclas["BK_ANCLA_FIN_USD_BBL"] < anclas["BK_ANCLA_PDP_USD_BBL"]]
    assert invertidas.empty, \
        f"{len(invertidas)} filas sinteticas con BK_ANCLA_FIN < BK_ANCLA_PDP"
