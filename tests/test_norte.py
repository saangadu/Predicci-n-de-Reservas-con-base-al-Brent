"""
test_norte.py — Automatización del contrato NORTE (docs/NORTE.md)

G1 (físicos):     skip explícito mientras 03 no persista sanity_checks.csv
                  (pendiente, en pausa junto con rediseño de escalera).
G2 (estadísticos): umbrales duros sobre el gate dorado en metricas.csv.
G3 (no-regresión): metricas.csv vs resultados/norte_baseline.csv congelado.

POR QUÉ: sin este contrato, cada sesión de optimización puede degradar el modelo
sin que nadie lo note (el pipeline corre verde igual). Estos tests convierten la
calidad del modelo en un criterio bloqueante, no en una opinión.
"""

from pathlib import Path

import pandas as pd
import pytest

BASE_DIR  = Path(__file__).parent.parent
METRICAS  = BASE_DIR / "datos" / "staging" / "metricas.csv"
SANITY    = BASE_DIR / "datos" / "staging" / "sanity_checks.csv"
BASELINE  = BASE_DIR / "resultados" / "norte_baseline.csv"

GATE = ["CASTILLA", "CASTILLA NORTE", "CASTILLA ESTE", "RUBIALES"]

# Umbrales G2 (ver docs/NORTE.md)
# Motores 1D (2026-06-11): ISO primario, SUAVE validacion. XGBoost retirado.
G2_N_REAL_MIN        = 6
G2_MAE_REL_ISO_MAX   = 0.20
G2_MAE_REL_SUAVE_MAX = 0.40

# Umbrales G3 (ver docs/NORTE.md)
G3_MAE_FACTOR   = 1.10
G3_MAE_ABS_TOL  = 0.5    # MBPE: evita falsos rojos en micro-campos (MAE~0.02)
G3_SKILL_CAIDA  = 0.05


@pytest.fixture(scope="module")
def metricas() -> pd.DataFrame:
    if not METRICAS.exists():
        pytest.skip(f"No existe {METRICAS} — correr 03_modelo.py primero.")
    df = pd.read_csv(METRICAS)
    faltantes = [c for c in GATE if c not in df["CAMPO"].values]
    assert not faltantes, f"Gate dorado ausente en metricas.csv: {faltantes}"
    return df.set_index("CAMPO")


@pytest.fixture(scope="module")
def baseline() -> pd.DataFrame:
    if not BASELINE.exists():
        pytest.skip(f"No existe {BASELINE} — congelar baseline (ver docs/NORTE.md G3).")
    return pd.read_csv(BASELINE).set_index("CAMPO")


# ── G1: físicos ────────────────────────────────────────────────────────────────

def test_g1_sanity_checks_persistidos():
    """G1 automatizado requiere que 03 persista sanity_checks.csv (pendiente).
    Mientras tanto, G1 se verifica en los prints [PASS]/[FAIL] de la corrida."""
    if not SANITY.exists():
        pytest.skip("sanity_checks.csv no existe aun — G1 manual via prints de 03 "
                    "(persistencia en pausa junto con rediseño de escalera).")
    df = pd.read_csv(SANITY)
    # Monotonia: bloqueante en TODOS los campos
    mono = df[df["CHECK"].str.startswith("monotonia")]
    fallos = mono[~mono["PASS"]]["CAMPO"].unique().tolist()
    assert not fallos, f"G1 monotonia violada en: {fallos}"
    # C1 y asintota: bloqueantes en el gate dorado
    gate_chk = df[df["CAMPO"].isin(GATE) &
                  df["CHECK"].isin(["sub_breakeven_vol_cero",
                                    "asintota_suave", "asintota_iso"])]
    fallos_g = gate_chk[~gate_chk["PASS"]][["CAMPO", "CHECK"]].values.tolist()
    assert not fallos_g, f"G1 fisicos gate violados: {fallos_g}"


# ── G2: estadísticos (gate dorado) ─────────────────────────────────────────────

@pytest.mark.parametrize("campo", GATE)
def test_g2_n_real_minimo(metricas, campo):
    assert metricas.loc[campo, "N_REAL_DELTA"] >= G2_N_REAL_MIN, (
        f"{campo}: N_REAL={metricas.loc[campo, 'N_REAL_DELTA']} < {G2_N_REAL_MIN}")


@pytest.mark.parametrize("campo", GATE)
def test_g2_skill_iso_positivo(metricas, campo):
    skill = metricas.loc[campo, "SKILL_ISO"]
    assert pd.notna(skill) and skill > 0, (
        f"{campo}: SKILL_ISO={skill} — la Isotonica no supera al predictor ingenuo")


@pytest.mark.parametrize("campo", GATE)
def test_g2_mae_rel_dentro_de_umbral(metricas, campo):
    mae_iso   = metricas.loc[campo, "MAE_REL_ISO"]
    mae_suave = metricas.loc[campo, "MAE_REL_SUAVE"]
    assert pd.notna(mae_iso) and mae_iso < G2_MAE_REL_ISO_MAX, (
        f"{campo}: MAE_REL_ISO={mae_iso} >= {G2_MAE_REL_ISO_MAX}")
    assert pd.notna(mae_suave) and mae_suave < G2_MAE_REL_SUAVE_MAX, (
        f"{campo}: MAE_REL_SUAVE={mae_suave} >= {G2_MAE_REL_SUAVE_MAX}")


# ── G3: no-regresión vs baseline congelado ─────────────────────────────────────

@pytest.mark.parametrize("campo", GATE)
@pytest.mark.parametrize("col", ["MAE_LOO_ISO", "MAE_LOO_SUAVE"])
def test_g3_mae_no_empeora(metricas, baseline, campo, col):
    if col not in baseline.columns:
        pytest.skip(f"{col} ausente en baseline (re-congelar tras cambio de arquitectura)")
    actual = metricas.loc[campo, col]
    base   = baseline.loc[campo, col]
    if pd.isna(base):
        pytest.skip(f"{campo}/{col}: sin valor en baseline")
    limite = base * G3_MAE_FACTOR + G3_MAE_ABS_TOL
    assert pd.notna(actual) and actual <= limite, (
        f"{campo}: {col}={actual} > limite {limite:.2f} "
        f"(baseline={base}) — regresion; ver docs/NORTE.md G3")


@pytest.mark.parametrize("campo", GATE)
def test_g3_skill_iso_no_cae(metricas, baseline, campo):
    actual = metricas.loc[campo, "SKILL_ISO"]
    base   = baseline.loc[campo, "SKILL_ISO"]
    if pd.isna(base):
        pytest.skip(f"{campo}: SKILL_ISO sin valor en baseline")
    assert pd.notna(actual) and actual >= base - G3_SKILL_CAIDA, (
        f"{campo}: SKILL_ISO={actual} < baseline {base} - {G3_SKILL_CAIDA} "
        f"— regresion; ver docs/NORTE.md G3")
