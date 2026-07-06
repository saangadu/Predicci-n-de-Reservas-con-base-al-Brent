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

# G3 ampliado (auditoria NORTE 2026-07-02): el baseline congelado protegia solo el
# gate dorado; los campos materiales con hallazgos (confound de vigencia, M2 fragil)
# no tenian proteccion de regresion. Se congelan tambien sus metricas.
MATERIALES_G3 = ["AKACIAS", "CHICHIMENE", "CHICHIMENE SW", "LA CIRA", "CUSIANA",
                 "YARIGUI-CANTAGALLO", "PAUTO SUR", "CAÑO SUR ESTE"]
G3_CAMPOS = GATE + MATERIALES_G3

# Umbrales G2 (ver docs/NORTE.md)
# Motores 1D (2026-06-11): ISO primario, SUAVE validacion. XGBoost retirado.
G2_N_REAL_MIN        = 6
G2_MAE_REL_ISO_MAX   = 0.20
G2_MAE_REL_SUAVE_MAX = 0.40

# Umbrales G3 (ver docs/NORTE.md)
G3_MAE_FACTOR   = 1.10
G3_MAE_ABS_TOL  = 0.5    # MBPE: evita falsos rojos en micro-campos (MAE~0.02)
G3_SKILL_CAIDA  = 0.05

# Umbrales G5 — etiquetas y M2 (ver docs/NORTE.md, añadido 2026-07-02)
G5_TOL_CAMPOS = 5        # deriva permitida por nivel de confianza / conteo de flags


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

@pytest.mark.parametrize("campo", G3_CAMPOS)
@pytest.mark.parametrize("col", ["MAE_LOO_ISO", "MAE_LOO_SUAVE"])
def test_g3_mae_no_empeora(metricas, baseline, campo, col):
    if col not in baseline.columns:
        pytest.skip(f"{col} ausente en baseline (re-congelar tras cambio de arquitectura)")
    if campo not in baseline.index:
        pytest.skip(f"{campo} ausente en baseline (re-congelar; ver MAESTRO §10)")
    actual = metricas.loc[campo, col]
    base   = baseline.loc[campo, col]
    if pd.isna(base):
        pytest.skip(f"{campo}/{col}: sin valor en baseline")
    limite = base * G3_MAE_FACTOR + G3_MAE_ABS_TOL
    assert pd.notna(actual) and actual <= limite, (
        f"{campo}: {col}={actual} > limite {limite:.2f} "
        f"(baseline={base}) — regresion; ver docs/NORTE.md G3")


@pytest.mark.parametrize("campo", G3_CAMPOS)
def test_g3_skill_iso_no_cae(metricas, baseline, campo):
    if campo not in baseline.index:
        pytest.skip(f"{campo} ausente en baseline (re-congelar; ver MAESTRO §10)")
    actual = metricas.loc[campo, "SKILL_ISO"]
    base   = baseline.loc[campo, "SKILL_ISO"]
    if pd.isna(base):
        pytest.skip(f"{campo}: SKILL_ISO sin valor en baseline")
    assert pd.notna(actual) and actual >= base - G3_SKILL_CAIDA, (
        f"{campo}: SKILL_ISO={actual} < baseline {base} - {G3_SKILL_CAIDA} "
        f"— regresion; ver docs/NORTE.md G3")


# ── G5: etiquetas y Modelo 2 (no-regresion de reporting, 2026-07-02) ──────────────
# La capa de confianza y M2 no tenian gate: un cambio podia voltear 40 etiquetas o
# degradar las rectas Brent->Neto sin que ningun test lo notara.

MATRIZ      = BASE_DIR / "resultados" / "output_matriz_prediccion.csv"
CORRELACION = BASE_DIR / "datos" / "staging" / "correlacion_brent.csv"
ETIQUETAS_BASELINE = BASE_DIR / "resultados" / "norte_etiquetas_baseline.csv"


@pytest.fixture(scope="module")
def etiquetas_baseline() -> dict:
    if not ETIQUETAS_BASELINE.exists():
        pytest.skip(f"No existe {ETIQUETAS_BASELINE} — congelar baseline de etiquetas.")
    df = pd.read_csv(ETIQUETAS_BASELINE)
    return dict(zip(df["METRICA"], df["VALOR"]))


@pytest.fixture(scope="module")
def matriz_campos() -> pd.DataFrame:
    if not MATRIZ.exists():
        pytest.skip(f"No existe {MATRIZ} — correr 04_pbi_export.py primero.")
    return pd.read_csv(MATRIZ).drop_duplicates("CAMPO")


def test_g5_beta_gate_dorado():
    """M2 del gate dorado: recta propia (THEILSEN, no fallback) con beta > 0."""
    if not CORRELACION.exists():
        pytest.skip("correlacion_brent.csv no existe — correr 03b primero.")
    corr = pd.read_csv(CORRELACION).set_index("CAMPO")
    for campo in GATE:
        assert campo in corr.index, f"{campo} sin fila en correlacion_brent.csv"
        fila = corr.loc[campo]
        assert fila["METODO"] == "THEILSEN" and not bool(fila["ES_FALLBACK"]), (
            f"{campo}: M2 degradado a {fila['METODO']} — el gate dorado exige recta propia")
        assert fila["BETA"] > 0, f"{campo}: BETA={fila['BETA']} <= 0 (monotonia M2 violada)"


def test_g5_distribucion_etiquetas(matriz_campos, etiquetas_baseline):
    """La distribucion ALTA/MEDIA/BAJA/SOLO_SINTETICO no deriva mas de
    G5_TOL_CAMPOS por nivel vs el baseline congelado."""
    niveles = matriz_campos["NIVEL_CONFIANZA"].value_counts()
    for nivel, clave in [("ALTA", "N_ALTA"), ("MEDIA", "N_MEDIA"),
                         ("BAJA", "N_BAJA"), ("SOLO_SINTETICO", "N_SOLO_SINTETICO")]:
        actual = int(niveles.get(nivel, 0))
        base = int(etiquetas_baseline[clave])
        assert abs(actual - base) <= G5_TOL_CAMPOS, (
            f"Etiquetas {nivel}: {actual} vs baseline {base} (tol ±{G5_TOL_CAMPOS}) "
            f"— cambio masivo de etiquetas; justificar y re-congelar (MAESTRO §10)")


def test_g5_conteo_fallbacks_m2(etiquetas_baseline):
    """El numero de campos M2 en fallback de portafolio no crece sin justificacion."""
    if not CORRELACION.exists():
        pytest.skip("correlacion_brent.csv no existe — correr 03b primero.")
    corr = pd.read_csv(CORRELACION)
    actual = int(corr["ES_FALLBACK"].sum())
    base = int(etiquetas_baseline["N_M2_FALLBACK"])
    assert actual <= base + G5_TOL_CAMPOS, (
        f"Fallbacks M2: {actual} vs baseline {base} (tol +{G5_TOL_CAMPOS}) — "
        f"campos perdiendo su recta propia")


def test_g5_conteo_flags_confound(matriz_campos, etiquetas_baseline):
    """El conteo de flags SENSIBILIDAD_NO_IDENTIFICADA no deriva sin justificacion."""
    if "SENSIBILIDAD_NO_IDENTIFICADA" not in matriz_campos.columns:
        pytest.skip("Columna SENSIBILIDAD_NO_IDENTIFICADA ausente — re-correr 04.")
    actual = int(matriz_campos["SENSIBILIDAD_NO_IDENTIFICADA"].sum())
    base = int(etiquetas_baseline["N_CONFOUND_FLAGS"])
    assert abs(actual - base) <= G5_TOL_CAMPOS, (
        f"Flags confound: {actual} vs baseline {base} (tol ±{G5_TOL_CAMPOS})")
