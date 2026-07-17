"""
seleccion_reglas.py — Reglas compartidas de adopcion M1/M2 (Calidad s12)

Umbral unico: mejora > 5% vs champion -> ADOPTADO_CALIDAD (dispatch en Calidad).
Sin banda EN_REVISION. Produccion solo aplica ESTADO=ADOPTADO (ratificado Finanzas).

Tambien: universo portafolio con modelo, filtros de hibrido (DECK_PLANO, etc.)
y factory de motores desde el nombre de metodo (primario / validacion dinamica).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import motores_modelo1 as M

BASE = Path(__file__).parent

# Mejora relativa estricta vs champion (>5%). Con N≈9 LOO es ruidoso: barras
# menores (p.ej. >0%) adoptarian ruido de muestreo como señal.
MEJORA_ADOPT = 0.05

# Divergencia primario vs validacion en banda real (sanidad C4 historica)
DIVERGENCIA_MAX = 0.30

N_MIN_REAL_HIBRIDO = 3
N_MIN_HIST_M2 = 5


def estado_por_mejora(mejora: float, checks_ok: bool = True) -> str | None:
    """Retorna ADOPTADO_CALIDAD si mejora redondeada a 1 decimal > 5.0% y checks OK.
    Comparar contra el % publicado (MEJORA_PCT) evita edge cases tipo 5.04% → 5.0%
    en CSV que fallarian el gate de tests."""
    if not checks_ok or pd.isna(mejora):
        return None
    # Mejora relativa en fraccion (0.05 = 5%); el umbral se aplica al % redondeado
    mejora_pct = round(float(mejora) * 100.0, 1)
    if mejora_pct > MEJORA_ADOPT * 100.0:
        return "ADOPTADO_CALIDAD"
    return None


def campos_portafolio_con_modelo(base: Path = None) -> list[str]:
    """Campos con curva de prediccion (EN_PREDICCION=True en cobertura).
    Ordenados por baseline desc; fallback a metricas.csv si no hay cobertura."""
    base = base or BASE
    cob = base / "resultados" / "cobertura_portafolio.csv"
    if cob.exists():
        df = pd.read_csv(cob)
        if "EN_PREDICCION" in df.columns:
            presente = df[df["EN_PREDICCION"] == True].copy()  # noqa: E712
            col_bl = "BASELINE_1P_MBPE" if "BASELINE_1P_MBPE" in presente.columns \
                else None
            if col_bl:
                presente = presente.sort_values(col_bl, ascending=False)
            return presente["CAMPO"].astype(str).tolist()
    # Fallback: campos en metricas con N_REAL > 0
    met = pd.read_csv(base / "resultados" / "metricas.csv")
    sub = met[met["N_REAL_DELTA"] > 0].copy()
    if "BASELINE_LATEST" in sub.columns:
        sub = sub.sort_values("BASELINE_LATEST", ascending=False)
    return sub["CAMPO"].astype(str).tolist()


def excluir_hibrido(campo: str, met: pd.DataFrame, nivel_map: dict = None) -> str | None:
    """Motivo de SKIP para benchmark hibrido, o None si el campo es elegible.
    Excluye DECK_PLANO (usuario 2026-07-15) e INSENSIBLE_PRECIO (sin deck real)."""
    if campo in met.index:
        fila = met.loc[campo]
        if bool(fila.get("DECK_PLANO", False)):
            return "deck_plano"
        if int(fila.get("N_REAL_DELTA", 0) or 0) < N_MIN_REAL_HIBRIDO:
            return "pocos_reales"
    if nivel_map and nivel_map.get(campo) == "INSENSIBLE_PRECIO":
        return "insensible"
    return None


def factory_desde_metodo(metodo: str, p_junta: float = None):
    """Callable sin args que instancia el motor nombrado.
    Acepta: champion | Suave | Plano | Sigmoide | LinealRobusto | HuberIso |
            hibrido:<nombre_sup>."""
    m = str(metodo).strip()

    def _iso():
        return M.MotorIsotonico()

    if m in ("champion", "default", "Isotonica", "isotonica"):
        return _iso
    if m == "Suave":
        return lambda: M.MotorSuave()
    if m == "Plano":
        return lambda: M.MotorPlano()
    if m == "Sigmoide":
        return lambda: M.MotorSigmoide()
    if m == "LinealRobusto":
        return lambda: M.MotorLinealRobusto()
    if m == "HuberIso":
        return lambda: M.MotorHuberIso()
    if m.startswith("hibrido:"):
        nombre = m.split(":", 1)[1]
        if nombre not in M.CANDIDATOS_HIBRIDO:
            return _iso
        f_sup = M.CANDIDATOS_HIBRIDO[nombre]
        pj = p_junta if pd.notna(p_junta) else 0.0
        return lambda: M.MotorHibrido(f_sup, pj)
    return _iso


def divergencia_volumen(modelo_a, modelo_b, baseline: float,
                        pneto_reales: np.ndarray) -> float:
    """Max |Vol_a - Vol_b| / Vol_a en la banda real observada (criterio C4)."""
    if len(pneto_reales) < 2 or pd.isna(baseline):
        return np.nan
    lo, hi = float(np.min(pneto_reales)), float(np.max(pneto_reales))
    if hi - lo < 1.0:
        lo, hi = lo - 1.0, hi + 1.0
    px = np.linspace(lo, hi, 20)
    va = np.maximum(baseline + modelo_a.predict(px), 0.0)
    vb = np.maximum(baseline + modelo_b.predict(px), 0.0)
    return float((np.abs(va - vb) / np.maximum(va, 1.0)).max())
