"""
test_norte.py — Automatización del contrato NORTE (docs/NORTE.md)

G1 (físicos):     skip explícito mientras 03 no persista sanity_checks.csv
                  (pendiente, en pausa junto con rediseño de escalera).
G2 (estadísticos): umbrales duros sobre el gate dorado en metricas.csv.
G3 (no-regresión): metricas.csv vs resultados/norte_baseline.csv congelado.
G5 (etiquetas/M2): distribución de confianza y salud de M2 vs baseline congelado.

POR QUÉ: sin este contrato, cada sesión de optimización puede degradar el modelo
sin que nadie lo note (el pipeline corre verde igual). Estos tests convierten la
calidad del modelo en un criterio bloqueante, no en una opinión.

MODO POR TRACK (paridad de rigor 2026-07-15, decisión usuario):
- Producción: G1/G2/G3/G5 duros (comportamiento histórico).
- Calidad:    G1/G2 (física, monotonía, N mínimo, MAE_REL, skill) siguen DUROS —
              Calidad no puede violar la física ni la estadística mínima.
              G3/G5 (no-regresión vs baseline congelado de PRODUCCIÓN) corren en
              modo COMPARATIVO: Calidad existe justamente para probar cambios que
              mueven métricas/etiquetas, así que las divergencias se REPORTAN en
              resultados_calidad/norte_divergencias.csv (+ warning pytest) sin
              tumbar el pipeline. La promoción a Producción exige revisar ese CSV.
"""

import os
import warnings
from pathlib import Path

import pandas as pd
import pytest

# Paths por track (Produccion vs Calidad): centralizados en tests/conftest.py.
# Los baselines congelados (norte_baseline / norte_etiquetas_baseline) viven SIEMPRE
# en resultados/ de Produccion: son la referencia contra la que Calidad se compara.
from rutas_track import STAGING, RESULTADOS, RESULTADOS_PROD, ES_CALIDAD

BASE_DIR  = Path(__file__).parent.parent
METRICAS  = STAGING / "metricas.csv"
SANITY    = STAGING / "sanity_checks.csv"
BASELINE  = RESULTADOS_PROD / "norte_baseline.csv"

# Modo comparativo para G3/G5: automatico en Calidad, o forzable via env
# PRED_NORTE_MODO=informativo (util para diagnosticos locales en Produccion).
MODO_INFORMATIVO = ES_CALIDAD or os.environ.get("PRED_NORTE_MODO", "") == "informativo"

# CSV donde el modo comparativo registra las divergencias G3/G5 detectadas
DIVERGENCIAS = RESULTADOS / "norte_divergencias.csv"

# Acumulador en memoria: cada divergencia G3/G5 detectada en modo informativo
_divergencias_rows: list[dict] = []


@pytest.fixture(scope="module", autouse=True)
def _registro_divergencias():
    """Resetea norte_divergencias.csv al inicio del modulo y lo escribe al final.
    Asi cada corrida del gate NORTE deja una foto limpia de sus divergencias
    (sin acumular corridas anteriores)."""
    if DIVERGENCIAS.exists():
        DIVERGENCIAS.unlink()
    yield
    if _divergencias_rows:
        DIVERGENCIAS.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(_divergencias_rows).to_csv(DIVERGENCIAS, index=False)


def _check_comparativo(condicion: bool, gate: str, objeto: str, metrica: str,
                       actual, base, mensaje: str) -> None:
    """Assert de G3/G5 sensible al modo del track.

    Produccion (modo duro): condicion falsa -> el test FALLA (bloqueante).
    Calidad (modo comparativo): condicion falsa -> se registra la divergencia en
    norte_divergencias.csv y se emite un warning; el test PASA. La revision de ese
    CSV es requisito de la promocion Calidad->Produccion."""
    if condicion:
        return
    if MODO_INFORMATIVO:
        _divergencias_rows.append({
            "GATE": gate, "OBJETO": objeto, "METRICA": metrica,
            "ACTUAL": actual, "BASELINE": base, "DETALLE": mensaje,
        })
        warnings.warn(f"[NORTE comparativo] {mensaje}")
    else:
        pytest.fail(mensaje)

# Gate Dorado = pareto-9 (directriz usuario 2026-07-09): los campos de mayor
# 1P certificado (sin filiales / brent-insensitive / PAUTO, cuyo deck reporta
# sensibilidad=0 desde 2025_Q1 — anomalia de datos en revision con planeacion).
# CHICHIMENE SW re-fusionado en CHICHIMENE (agregacion v3, s3): ya no es campo propio.
GATE = ["RUBIALES", "CASTILLA", "CAÑO SUR ESTE", "CASTILLA NORTE", "AKACIAS",
        "CHICHIMENE", "LA CIRA", "CUPIAGUA", "YARIGUI-CANTAGALLO"]

# G3: el gate-10 absorbe los materiales del G3 ampliado 2026-07-02; se conservan
# ademas CASTILLA ESTE (gate historico) y CUSIANA (material con hallazgos).
# PAUTO SUR ya no existe como campo (fusionado en PAUTO via DIM UNIFICADO 2026-07-09).
MATERIALES_G3 = ["CASTILLA ESTE", "CUSIANA"]
G3_CAMPOS = GATE + MATERIALES_G3

# Umbrales G2 (ver docs/NORTE.md)
# Motores 1D (2026-06-11): ISO primario, SUAVE validacion. XGBoost retirado.
G2_N_REAL_MIN        = 6
G2_MAE_REL_ISO_MAX   = 0.20
G2_MAE_REL_SUAVE_MAX = 0.40
# Campo plano (normalizacion v2): con DELTA vs cierre A−1 el target es casi plano en
# muchos campos (esa ES la senal honesta); el naive es imbatible por construccion y
# SKILL>0 deja de ser criterio valido. Exencion identica a la de 04 (confianza).
G2_NAIVE_PLANO_MBPE  = 2.0

# Excepciones G2 documentadas (NORTE.md §G2; re-freeze 2026-07-09). No son umbrales
# relajados en silencio: cada una tiene causa de datos identificada y visible.
G2_EXCEPCIONES = {
    # Tras la re-fusion de CHICHIMENE SW (agregacion v3, s3) el MAE_REL_ISO de CHICHIMENE
    # es 0.042 (excelente) pero el SKILL_ISO LOO queda negativo: la serie fusionada
    # arrastra el salto de recertificacion 2024->2025 (+22 MBPE en cierre) que el LOO
    # penaliza, y el naive del delta (MAE_NAIVE=3.63, apenas > umbral plano 2.0) es
    # imbatible por construccion. SKILL es referencial, no KPI (MAESTRO v2); la metrica
    # honesta MAE_REL pasa limpia. Exencion SOLO del test skill, con causa de datos visible.
    "CHICHIMENE": {"test": "skill"},
}

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

def _excepcion_g2(campo: str, test: str) -> bool:
    exc = G2_EXCEPCIONES.get(campo)
    return exc is not None and exc["test"] == test


@pytest.mark.parametrize("campo", GATE)
def test_g2_n_real_minimo(metricas, campo):
    if _excepcion_g2(campo, "n_real"):
        pytest.skip(f"{campo}: excepcion G2 documentada (ver G2_EXCEPCIONES / NORTE.md)")
    assert metricas.loc[campo, "N_REAL_DELTA"] >= G2_N_REAL_MIN, (
        f"{campo}: N_REAL={metricas.loc[campo, 'N_REAL_DELTA']} < {G2_N_REAL_MIN}")


@pytest.mark.parametrize("campo", GATE)
def test_g2_skill_iso_positivo(metricas, campo):
    skill = metricas.loc[campo, "SKILL_ISO"]
    naive = metricas.loc[campo, "MAE_NAIVE"]
    if pd.notna(naive) and naive < G2_NAIVE_PLANO_MBPE:
        pytest.skip(f"{campo}: campo plano (MAE_NAIVE={naive} < {G2_NAIVE_PLANO_MBPE} "
                    f"MBPE) — el naive es imbatible por construccion")
    if _excepcion_g2(campo, "skill"):
        pytest.skip(f"{campo}: excepcion G2 documentada (ver G2_EXCEPCIONES / NORTE.md)")
    assert pd.notna(skill) and skill > 0, (
        f"{campo}: SKILL_ISO={skill} — la Isotonica no supera al predictor ingenuo")


@pytest.mark.parametrize("campo", GATE)
def test_g2_mae_rel_dentro_de_umbral(metricas, campo):
    if _excepcion_g2(campo, "mae_rel"):
        pytest.skip(f"{campo}: excepcion G2 documentada (ver G2_EXCEPCIONES / NORTE.md)")
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
    _check_comparativo(
        pd.notna(actual) and actual <= limite,
        "G3", campo, col, actual, base,
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
    _check_comparativo(
        pd.notna(actual) and actual >= base - G3_SKILL_CAIDA,
        "G3", campo, "SKILL_ISO", actual, base,
        f"{campo}: SKILL_ISO={actual} < baseline {base} - {G3_SKILL_CAIDA} "
        f"— regresion; ver docs/NORTE.md G3")


# ── G5: etiquetas y Modelo 2 (no-regresion de reporting, 2026-07-02) ──────────────
# La capa de confianza y M2 no tenian gate: un cambio podia voltear 40 etiquetas o
# degradar las rectas Brent->Neto sin que ningun test lo notara.

MATRIZ      = RESULTADOS / "output_matriz_prediccion.csv"
CORRELACION = STAGING / "correlacion_brent.csv"
# Baseline de etiquetas: siempre la foto congelada de Produccion (referencia)
ETIQUETAS_BASELINE = RESULTADOS_PROD / "norte_etiquetas_baseline.csv"


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
    """M2 del gate dorado: recta/curva propia (no fallback de portafolio) con beta > 0.
    s14: familias M2 ratificadas en ambos tracks — la identidad del metodo ya no exige
    THEILSEN; se exige metodo del conjunto permitido (nucleo + familias adoptadas)."""
    if not CORRELACION.exists():
        pytest.skip("correlacion_brent.csv no existe — correr 03b primero.")
    METODOS_PERMITIDOS = {"THEILSEN", "PROPORCIONAL", "DESCOMPUESTO", "SEGMENTADA",
                          "HUBER", "CUADRATICA_MONOTONA"}
    corr = pd.read_csv(CORRELACION).set_index("CAMPO")
    for campo in GATE:
        assert campo in corr.index, f"{campo} sin fila en correlacion_brent.csv"
        fila = corr.loc[campo]
        # Fisica dura en AMBOS tracks: sin fallback de portafolio y monotonia BETA>0
        assert not bool(fila["ES_FALLBACK"]), (
            f"{campo}: M2 en fallback de portafolio — el gate dorado exige recta propia")
        assert fila["BETA"] > 0, f"{campo}: BETA={fila['BETA']} <= 0 (monotonia M2 violada)"
        assert fila["METODO"] in METODOS_PERMITIDOS, (
            f"{campo}: M2 con metodo {fila['METODO']} fuera del conjunto ratificado s14")


def test_g5_distribucion_etiquetas(matriz_campos, etiquetas_baseline):
    """La distribucion ALTA/MEDIA/BAJA/INSENSIBLE_PRECIO/SIN_MODELO no deriva mas de
    G5_TOL_CAMPOS por nivel vs el baseline congelado. INSENSIBLE_PRECIO (antes SOLO_GAS/
    SOLO_SINTETICO, renombre 2026-07-15) y SIN_MODELO (cobertura de portafolio) son los
    niveles nuevos."""
    niveles = matriz_campos["NIVEL_CONFIANZA"].value_counts()
    for nivel, clave in [("ALTA", "N_ALTA"), ("MEDIA", "N_MEDIA"),
                         ("BAJA", "N_BAJA"), ("INSENSIBLE_PRECIO", "N_INSENSIBLE_PRECIO"),
                         ("SIN_MODELO", "N_SIN_MODELO")]:
        actual = int(niveles.get(nivel, 0))
        base = int(etiquetas_baseline[clave])
        _check_comparativo(
            abs(actual - base) <= G5_TOL_CAMPOS,
            "G5", "PORTAFOLIO", f"N_{nivel}", actual, base,
            f"Etiquetas {nivel}: {actual} vs baseline {base} (tol ±{G5_TOL_CAMPOS}) "
            f"— cambio masivo de etiquetas; justificar y re-congelar (MAESTRO §10)")


def test_g5_conteo_fallbacks_m2(etiquetas_baseline):
    """El numero de campos M2 en fallback de portafolio no crece sin justificacion."""
    if not CORRELACION.exists():
        pytest.skip("correlacion_brent.csv no existe — correr 03b primero.")
    corr = pd.read_csv(CORRELACION)
    actual = int(corr["ES_FALLBACK"].sum())
    base = int(etiquetas_baseline["N_M2_FALLBACK"])
    _check_comparativo(
        actual <= base + G5_TOL_CAMPOS,
        "G5", "PORTAFOLIO", "N_M2_FALLBACK", actual, base,
        f"Fallbacks M2: {actual} vs baseline {base} (tol +{G5_TOL_CAMPOS}) — "
        f"campos perdiendo su recta propia")


def test_g5_conteo_flags_confound(matriz_campos, etiquetas_baseline):
    """El conteo de flags SENSIBILIDAD_NO_IDENTIFICADA no deriva sin justificacion."""
    if "SENSIBILIDAD_NO_IDENTIFICADA" not in matriz_campos.columns:
        pytest.skip("Columna SENSIBILIDAD_NO_IDENTIFICADA ausente — re-correr 04.")
    actual = int(matriz_campos["SENSIBILIDAD_NO_IDENTIFICADA"].sum())
    base = int(etiquetas_baseline["N_CONFOUND_FLAGS"])
    _check_comparativo(
        abs(actual - base) <= G5_TOL_CAMPOS,
        "G5", "PORTAFOLIO", "N_CONFOUND_FLAGS", actual, base,
        f"Flags confound: {actual} vs baseline {base} (tol ±{G5_TOL_CAMPOS})")
