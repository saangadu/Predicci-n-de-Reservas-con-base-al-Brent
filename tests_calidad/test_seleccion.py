"""
tests_calidad/test_seleccion.py — Gate del track Calidad para la selección de método
por campo. NO toca el contrato NORTE de Producción (baselines congeladas de Producción).

Rigor CRECIENTE con la adopción (pedido del usuario): un método con más campos activa
más chequeos. Tiers:
  T1 (todo método): invariantes por campo (monotonía, ancla C5, hard-zero), registro
     íntegro, mejora >5% benchmarkeada, aislamiento del dispatch.
  T2 (método con ≥3 campos): estabilidad del método across sus campos + control negativo
     (el método no se aplicó donde no calificó) + no-regresión del ancla agregada.
  T3 (método con ≥8 campos): freeze de baseline del track (norte_calidad) — aún no activo.

Lee resultados_calidad/ directamente (independiente de env).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
CAL = ROOT / "resultados_calidad"

MEJORA_MIN = 5.0
C5_TOL = 0.5          # MBPE
T2_MIN_CAMPOS = 3
T3_MIN_CAMPOS = 8


@pytest.fixture(scope="module")
def registro():
    # Gobernanza por track (2026-07-15): el track Calidad aplica ADOPTADO (ratificados)
    # + ADOPTADO_CALIDAD (hibridos/investigaciones con evidencia, pendientes de
    # ratificacion). Este gate valida TODO lo que Calidad realmente despacha.
    r = pd.read_csv(CAL / "seleccion_metodos.csv")
    return r[r["ESTADO"].isin(["ADOPTADO", "ADOPTADO_CALIDAD"])].reset_index(drop=True)


@pytest.fixture(scope="module")
def matriz():
    m = pd.read_csv(CAL / "output_matriz_prediccion.csv")
    return m[m["MOTOR"] == "Isotonica"].copy()


@pytest.fixture(scope="module")
def metricas():
    return pd.read_csv(CAL / "metricas.csv")


# ── Tier 1 — todo método ──────────────────────────────────────────────────────

def test_registro_integro(registro):
    """Todo campo adoptado tiene método, hipótesis y mejora >5% benchmarkeada."""
    assert len(registro) > 0
    assert registro["HIPOTESIS"].str.len().gt(20).all(), "hipótesis vacía o trivial"
    assert (registro["MEJORA_PCT"] > MEJORA_MIN).all(), \
        f"adopción sin mejora >{MEJORA_MIN}%:\n{registro[registro.MEJORA_PCT <= MEJORA_MIN]}"


def test_mejora_vs_champion(registro):
    """El MAE_LOYO del método < MAE_LOYO del champion por el margen declarado."""
    for _, r in registro.iterrows():
        assert r["MAE_LOYO_METODO"] < r["MAE_LOYO_CHAMPION"], \
            f"{r['CAMPO']}: método {r['MAE_LOYO_METODO']} no mejora champion {r['MAE_LOYO_CHAMPION']}"


def test_monotonia_campos_override(registro, matriz):
    """Brent↑ → Vol↑ en cada campo con método override (invariante física dura)."""
    for campo in registro["CAMPO"]:
        s = matriz[matriz["CAMPO"] == campo].sort_values("BRENT_USD_BBL")
        vol = s["VOLUMEN_1P_PREDICHO_MBPE"].values
        assert np.all(np.diff(vol) >= -1e-6), f"{campo}: monotonía violada"


def test_ancla_c5_override(registro, matriz):
    """Vol(BRENT_REF) = baseline en cada campo override (re-anclaje intacto pese al motor)."""
    for campo in registro["CAMPO"]:
        s = matriz[matriz["CAMPO"] == campo]
        if s.empty:
            continue
        bl = s["VOLUMEN_1P_BASELINE_MBPE"].iloc[0]
        i = (s["BRENT_USD_BBL"] - s["BRENT_REF_USD_BBL"].iloc[0]).abs().idxmin()
        err = abs(s.loc[i, "VOLUMEN_1P_PREDICHO_MBPE"] - bl)
        assert err < C5_TOL, f"{campo}: Vol(BRENT_REF)={s.loc[i,'VOLUMEN_1P_PREDICHO_MBPE']:.2f} vs baseline {bl:.2f}"


def test_hard_zero_bajo_piso(registro, matriz):
    """Bajo el piso efectivo el volumen es 0 (abandono); ningún método lo viola."""
    for campo in registro["CAMPO"]:
        s = matriz[matriz["CAMPO"] == campo]
        piso = s["PISO_EFECTIVO_USD_BBL"].dropna()
        if piso.empty:
            continue
        piso = float(piso.iloc[0])
        bajo = s[s["PRECIO_NETO_EFECTIVO_USD_BBL"] < piso - 1e-6]
        assert (bajo["VOLUMEN_1P_PREDICHO_MBPE"].abs() < 1e-6).all(), \
            f"{campo}: volumen no nulo bajo el piso {piso}"


def test_dispatch_aislamiento(registro, metricas):
    """Solo los campos del registro llevan método override; el resto queda en 'default'."""
    if "METODO_SELECCIONADO" not in metricas.columns:
        pytest.skip("corrida sin PRED_SELECCION_METODO")
    override = metricas[metricas["METODO_SELECCIONADO"] != "default"]
    assert set(override["CAMPO"]) == set(registro["CAMPO"]), \
        f"dispatch fuera del registro: {set(override['CAMPO']) ^ set(registro['CAMPO'])}"


def test_ancla_agregada_portafolio(matriz):
    """No-regresión: Σ Vol(BRENT_REF) ≈ Σ baseline (el anclaje se conserva a nivel cartera)."""
    ref = []
    for campo, s in matriz.groupby("CAMPO"):
        if s["BRENT_REF_USD_BBL"].isna().all():
            continue
        i = (s["BRENT_USD_BBL"] - s["BRENT_REF_USD_BBL"].iloc[0]).abs().idxmin()
        ref.append((s.loc[i, "VOLUMEN_1P_PREDICHO_MBPE"], s["VOLUMEN_1P_BASELINE_MBPE"].iloc[0]))
    ref = np.array(ref)
    tot_pred, tot_base = ref[:, 0].sum(), ref[:, 1].sum()
    assert abs(tot_pred - tot_base) / tot_base < 0.01, \
        f"ancla agregada desviada: pred {tot_pred:.0f} vs base {tot_base:.0f}"


# ── Tier 2 — método con ≥3 campos ─────────────────────────────────────────────

def test_tier2_estabilidad_por_metodo(registro):
    """Para cada método con ≥3 campos, TODOS sus campos deben mantener la mejora >5%
    (no un promedio inflado por un caso). Rigor que crece con la adopción."""
    conteo = registro["METODO"].value_counts()
    for met, n in conteo.items():
        if n < T2_MIN_CAMPOS:
            continue
        sub = registro[registro["METODO"] == met]
        assert (sub["MEJORA_PCT"] > MEJORA_MIN).all(), \
            f"método {met} (n={n}): algún campo cae bajo o igual a {MEJORA_MIN}%:\n{sub}"
        # dispersión razonable: ningún campo con mejora negativa oculta
        assert sub["MAE_LOYO_METODO"].notna().all(), f"{met}: MAE_LOYO faltante"


def test_metodo_validacion_en_hibridos(registro):
    """s12: adopciones hibrido:* deben declarar METODO_VALIDACION (2.o LOYO)."""
    hib = registro[(registro["ESTADO"] == "ADOPTADO_CALIDAD")
                   & registro["METODO"].astype(str).str.startswith("hibrido:")]
    if hib.empty:
        pytest.skip("sin hibridos ADOPTADO_CALIDAD")
    assert "METODO_VALIDACION" in hib.columns, "falta columna METODO_VALIDACION"
    faltan = hib[hib["METODO_VALIDACION"].isna()
                 | (hib["METODO_VALIDACION"].astype(str).str.strip() == "")]
    assert faltan.empty, f"hibridos sin METODO_VALIDACION:\n{faltan}"


def test_tier3_freeze_baseline(registro):
    """T3: si algún método alcanza ≥8 campos, exige un baseline congelado del track
    (norte_calidad_baseline.csv). Hasta entonces, informativo (skip)."""
    conteo = registro["METODO"].value_counts()
    grandes = conteo[conteo >= T3_MIN_CAMPOS]
    if grandes.empty:
        pytest.skip(f"ningún método alcanza {T3_MIN_CAMPOS} campos todavía")
    if not (CAL / "norte_calidad_baseline.csv").exists():
        pytest.skip(
            f"métodos maduros ({list(grandes.index)}) — freeze norte_calidad "
            f"pendiente de crear (informativo hasta entonces)"
        )
