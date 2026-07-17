"""
m2_familias.py — Familias de funciones candidatas del Modelo 2 (Brent -> Precio Aceite)

Investigacion M2 (directriz usuario 2026-07-15): el Theil-Sen lineal deja 8 campos del
top-20 con R2 < 0.9 (CHICHIMENE 0.77, AKACIAS 0.77, ...). Este modulo define familias
alternativas POR CAMPO — cada campo puede tener su propia funcion si mejora su
correlacion out-of-sample — y el harness LOO que las compara de forma justa.

Vive en un modulo aparte (no en analisis_m2.py ni en 03b) porque lo consumen DOS
scripts: analisis_m2.py (benchmark offline que elige la familia ganadora por campo)
y 03b_correlacion_brent.py (dispatch en el track Calidad bajo PRED_M2_SELECCION).

FAMILIAS (eje 1 de la busqueda; el eje 2 es el dataset de entrenamiento):
  THEILSEN            : champion vigente. Neto = ALPHA + BETA*Brent (mediana de pendientes).
  DESCOMPUESTO        : Neto = Brent − DescTra_mediana − DescCal(Brent), con el descuento
                        de calidad modelado vs Brent (Theil-Sen sobre la serie de
                        descuentos). Hipotesis: el diferencial de crudos pesados se
                        abre/cierra con el nivel de Brent (World Bank/ESMAP 2005) —
                        el canal de informacion es la columna DESCUENTO_CALIDAD_USD_BBL,
                        que la recta directa no usa.
  SEGMENTADA          : recta con 1 quiebre (2 pendientes, ambas > 0, continua en el
                        quiebre). Hipotesis: regimen de precios bajo/alto con
                        transferencia distinta al precio realizado.
  HUBER               : recta por perdida Huber (robusta a outliers con eficiencia
                        gaussiana mejor que Theil-Sen cuando el ruido es mixto).
  CUADRATICA_MONOTONA : parabola restringida a derivada > 0 en la banda fisica
                        [40, 120]; extrapolacion LINEAL fuera del rango observado
                        (la parabola solo gobierna donde hubo datos).

INVARIANTES FISICAS (obligatorias para adoptar cualquier familia):
  1. dNeto/dBrent > 0 en [40, 120]  (monotonia — invariante del MAESTRO)
  2. Neto(40) >= 0                  (precio realizado no negativo)
  3. Extrapolacion acotada: fuera del rango observado la curva continua LINEALMENTE
     con la pendiente del borde (nunca explota — relevante para la cuadratica).

Los parametros de cada familia se serializan como JSON (columna M2_PARAMS de
correlacion_brent.csv) para que neto_desde_brent() de 03b los reconstruya sin
cambiar la interfaz que consumen 03_modelo.py y 04_pbi_export.py.
"""
import json

import numpy as np
from scipy import stats

# Banda fisica donde se verifica la monotonia de toda familia candidata
BANDA_FISICA = (40.0, 120.0)

# Minimo de puntos con descuento de calidad para poder ajustar DESCOMPUESTO
N_MIN_DESCUENTOS = 4

# Minimo de puntos para buscar un quiebre con 2 pendientes estimables
N_MIN_SEGMENTADA = 6

FAMILIAS_CANDIDATAS = ["THEILSEN", "DESCOMPUESTO", "SEGMENTADA", "HUBER",
                       "CUADRATICA_MONOTONA"]


# ── Ajuste por familia ────────────────────────────────────────────────────────
# Cada _ajustar_* retorna un dict de parametros JSON-serializable, o None si la
# familia no es ajustable/valida para esos datos (pocos puntos, monotonia violada).

def _ajustar_theilsen(b, y, extra=None):
    """Champion: recta robusta por mediana de pendientes. BETA > 0 obligatorio."""
    if len(b) < 5 or np.ptp(b) <= 1e-6:
        return None
    slope, intercept, _, _ = stats.theilslopes(y, b)
    if slope <= 0:
        return None
    return {"ALPHA": float(intercept), "BETA": float(slope)}


def _ajustar_descompuesto(b, y, extra=None):
    """
    Neto = Brent − DTRA − (DCAL_A + DCAL_B*Brent)

    DTRA = mediana de |descuento transporte| (estable: es tarifa, no mercado).
    DCAL(B) modelado con Theil-Sen sobre |descuento calidad| vs Brent: captura que el
    diferencial de calidad se ABRE con Brent alto (DCAL_B > 0 en crudos pesados).
    La pendiente resultante de Neto es (1 − DCAL_B): se exige DCAL_B < 1 para no
    violar la monotonia. Si la señal de pendiente del descuento no existe (n bajo),
    degrada a descuento constante (DCAL_B = 0, mediana).
    """
    if extra is None:
        return None
    dcal = np.asarray(extra.get("desc_cal", []), dtype=float)
    dtra = np.asarray(extra.get("desc_tra", []), dtype=float)
    # Solo filas con AMBOS descuentos: el modelo se arma por el canal de descuentos
    mask = np.isfinite(dcal) & np.isfinite(dtra) & np.isfinite(b)
    if mask.sum() < N_MIN_DESCUENTOS:
        return None
    b_d = b[mask]
    # Convencion del tablon: descuentos <= 0 -> magnitud positiva para el modelo
    mag_cal = -dcal[mask]
    mag_tra = -dtra[mask]
    dtra_med = float(np.median(mag_tra))
    if np.ptp(b_d) > 1e-6 and mask.sum() >= N_MIN_DESCUENTOS:
        pend, inter, _, _ = stats.theilslopes(mag_cal, b_d)
    else:
        pend, inter = 0.0, float(np.median(mag_cal))
    # Monotonia de Neto: pendiente = 1 − DCAL_B > 0. Si el descuento crece mas
    # rapido que el Brent (irreal), degradar a descuento constante.
    if pend >= 0.9:
        pend, inter = 0.0, float(np.median(mag_cal))
    # Sesgo residual: el Neto real no es exactamente Brent−descuentos (bases de
    # liquidacion distintas). Se corrige con un offset = mediana del residual,
    # para que la familia compita sin handicap sistematico.
    y_hat = b - dtra_med - (inter + pend * b)
    offset = float(np.median(y - y_hat))
    return {"DTRA": dtra_med, "DCAL_A": float(inter), "DCAL_B": float(pend),
            "OFFSET": offset}


def _ajustar_segmentada(b, y, extra=None):
    """
    Recta continua con 1 quiebre: y = Y0 + B1·(x−X0) si x<=X0; Y0 + B2·(x−X0) si x>X0.
    Busca el quiebre entre los cuantiles 25-75 de Brent (>=2 puntos por lado) y toma
    el de menor SSE. Ambas pendientes deben ser > 0 (monotonia).
    """
    n = len(b)
    if n < N_MIN_SEGMENTADA or np.ptp(b) <= 1e-6:
        return None
    orden = np.argsort(b)
    bs, ys = b[orden], y[orden]
    mejor = None
    # Candidatos de quiebre: puntos interiores que dejan >=2 puntos por lado
    for x0 in np.unique(bs[2:-2]):
        # Diseño continuo en el quiebre: [1, min(x−x0,0), max(x−x0,0)]
        A = np.column_stack([np.ones(n),
                             np.minimum(bs - x0, 0.0),
                             np.maximum(bs - x0, 0.0)])
        try:
            coefs, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        except np.linalg.LinAlgError:
            continue
        y0, b1, b2 = float(coefs[0]), float(coefs[1]), float(coefs[2])
        if b1 <= 0 or b2 <= 0:
            continue   # monotonia: ambas pendientes positivas
        sse = float(np.sum((A @ coefs - ys) ** 2))
        if mejor is None or sse < mejor[0]:
            mejor = (sse, {"X0": float(x0), "Y0": y0, "B1": b1, "B2": b2})
    return mejor[1] if mejor else None


def _ajustar_huber(b, y, extra=None):
    """Recta robusta por perdida Huber (IRLS). BETA > 0 obligatorio."""
    if len(b) < 5 or np.ptp(b) <= 1e-6:
        return None
    from sklearn.linear_model import HuberRegressor
    try:
        h = HuberRegressor().fit(b.reshape(-1, 1), y)
    except Exception:
        return None
    slope = float(h.coef_[0])
    if slope <= 0:
        return None
    return {"ALPHA": float(h.intercept_), "BETA": slope}


def _ajustar_cuadratica(b, y, extra=None):
    """
    Parabola y = C0 + C1·x + C2·x², valida solo si su derivada C1 + 2·C2·x es > 0 en
    toda la banda fisica [40, 120]. Fuera del rango OBSERVADO extrapola linealmente
    con la pendiente del borde (BMIN/BMAX guardados) para que nunca explote.
    """
    if len(b) < 6 or np.ptp(b) <= 1e-6:
        return None
    try:
        c2, c1, c0 = np.polyfit(b, y, 2)
    except Exception:
        return None
    lo, hi = BANDA_FISICA
    # Derivada lineal en x: basta verificarla en los extremos de la banda fisica
    if (c1 + 2 * c2 * lo) <= 0 or (c1 + 2 * c2 * hi) <= 0:
        return None
    return {"C0": float(c0), "C1": float(c1), "C2": float(c2),
            "BMIN": float(np.min(b)), "BMAX": float(np.max(b))}


_AJUSTADORES = {
    "THEILSEN":            _ajustar_theilsen,
    "DESCOMPUESTO":        _ajustar_descompuesto,
    "SEGMENTADA":          _ajustar_segmentada,
    "HUBER":               _ajustar_huber,
    "CUADRATICA_MONOTONA": _ajustar_cuadratica,
}


def ajustar(familia: str, b: np.ndarray, y: np.ndarray, extra: dict = None):
    """Ajusta una familia. Retorna dict de parametros JSON-serializable o None."""
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)
    return _AJUSTADORES[familia](b, y, extra)


# ── Prediccion por familia ────────────────────────────────────────────────────

def predecir(familia: str, params: dict, brent) -> np.ndarray:
    """Evalua la familia con sus parametros. Interfaz unica para el dispatch de 03b."""
    x = np.asarray(brent, dtype=float)
    if familia in ("THEILSEN", "HUBER"):
        return params["ALPHA"] + params["BETA"] * x
    if familia == "DESCOMPUESTO":
        return (x - params["DTRA"] - (params["DCAL_A"] + params["DCAL_B"] * x)
                + params.get("OFFSET", 0.0))
    if familia == "SEGMENTADA":
        x0, y0 = params["X0"], params["Y0"]
        return np.where(x <= x0,
                        y0 + params["B1"] * (x - x0),
                        y0 + params["B2"] * (x - x0))
    if familia == "CUADRATICA_MONOTONA":
        c0, c1, c2 = params["C0"], params["C1"], params["C2"]
        bmin, bmax = params["BMIN"], params["BMAX"]
        xc = np.clip(x, bmin, bmax)
        base = c0 + c1 * xc + c2 * xc ** 2
        # Extrapolacion LINEAL con la pendiente del borde (acotada, monotona)
        pend_lo = c1 + 2 * c2 * bmin
        pend_hi = c1 + 2 * c2 * bmax
        base = base + np.where(x < bmin, (x - bmin) * pend_lo, 0.0)
        base = base + np.where(x > bmax, (x - bmax) * pend_hi, 0.0)
        return base
    raise ValueError(f"Familia M2 desconocida: {familia}")


def beta_equivalente(familia: str, params: dict) -> float:
    """Pendiente promedio de la familia en la banda fisica: dNeto/dBrent equivalente.
    Es lo que se publica en la columna BETA para familias no lineales — los gates
    fisicos (BETA>0, rango razonable) siguen aplicando sin cambios."""
    lo, hi = BANDA_FISICA
    y_lo, y_hi = predecir(familia, params, np.array([lo, hi]))
    return float((y_hi - y_lo) / (hi - lo))


def pasa_fisica(familia: str, params: dict) -> bool:
    """Verifica las invariantes fisicas de la curva en la banda [40, 120]:
    monotonia estricta, Neto(40) >= 0 y sin descuento implicito absurdo (la curva
    no se aleja del Brent mas de 40 USD en ninguna direccion)."""
    lo, hi = BANDA_FISICA
    grid = np.linspace(lo, hi, 81)
    y = predecir(familia, params, grid)
    if np.any(np.diff(y) <= 0):
        return False          # monotonia dNeto/dBrent > 0
    if y[0] < 0:
        return False          # precio realizado no negativo en Brent=40
    if np.any(np.abs(y - grid) > 40.0):
        return False          # descuento/premio implicito acotado (fisica de mercado)
    return True


# ── Harness LOO (validacion justa entre familias Y entre datasets) ────────────

def loo_familia(familia: str, b_hist: np.ndarray, y_hist: np.ndarray,
                extra_hist: dict = None,
                b_cons: np.ndarray = None, y_cons: np.ndarray = None) -> tuple:
    """
    LOO-CV dejando fuera SOLO puntos HIST (los cierres certificados son la verdad a
    predecir). Las sensibilidades del Consolidado (b_cons/y_cons), si se pasan, van
    SIEMPRE al train y NUNCA al test: asi la comparacion solo-HIST vs
    HIST+CONSOLIDADO es justa y no se auto-premia el dataset mas grande.

    Retorna (MAE_LOO, R2_LOO) sobre los puntos HIST. NaN si no es evaluable.
    """
    b_hist = np.asarray(b_hist, dtype=float)
    y_hist = np.asarray(y_hist, dtype=float)
    n = len(b_hist)
    if n < 4 or np.ptp(b_hist) <= 1e-6:
        return np.nan, np.nan
    tiene_cons = b_cons is not None and len(b_cons) > 0
    errores = []
    for i in range(n):
        b_tr = np.delete(b_hist, i)
        y_tr = np.delete(y_hist, i)
        extra_tr = None
        if extra_hist is not None:
            # Los descuentos van alineados a los puntos HIST: excluir el mismo indice
            extra_tr = {k: np.delete(np.asarray(v, dtype=float), i)
                        for k, v in extra_hist.items()}
        if tiene_cons:
            b_tr = np.concatenate([b_tr, b_cons])
            y_tr = np.concatenate([y_tr, y_cons])
            if extra_tr is not None:
                # Las sensibilidades no traen descuentos: NaN (no aportan a DCAL)
                extra_tr = {k: np.concatenate([v, np.full(len(b_cons), np.nan)])
                            for k, v in extra_tr.items()}
        params = ajustar(familia, b_tr, y_tr, extra_tr)
        if params is None:
            return np.nan, np.nan   # familia no estable bajo LOO -> no evaluable
        errores.append(y_hist[i] - float(predecir(familia, params, [b_hist[i]])[0]))
    errores = np.asarray(errores, dtype=float)
    mae_loo = float(np.mean(np.abs(errores)))
    ss_res = float(np.sum(errores ** 2))
    ss_tot = float(np.sum((y_hist - np.mean(y_hist)) ** 2))
    r2_loo = (1 - ss_res / ss_tot) if ss_tot > 1e-9 else np.nan
    return mae_loo, r2_loo


def params_a_json(params: dict) -> str:
    """Serializa los parametros para la columna M2_PARAMS de correlacion_brent.csv."""
    return json.dumps(params, ensure_ascii=False)


def params_desde_json(texto: str) -> dict:
    """Reconstruye los parametros desde la columna M2_PARAMS."""
    return json.loads(texto)
