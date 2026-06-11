"""
motores_modelo1.py — Motores monotonos 1D para el Modelo 1 (Precio Neto -> Delta Reservas)

Tras dividir el motor en 2 modelos (directriz 2026-06-11), el Modelo 1 deja de ser 3D
(Brent, Calidad, Transporte). La unica variable de entrada es PRECIO_NETO_USD_BBL: la
fisica de la sensibilidad de reservas vive en el precio realizado neto, no en el Brent
(ese paso lo cubre el Modelo 2). Con una sola variable, XGBoost pierde su ventaja (era
elegido por ser 3D) y, peor, su restriccion (1,-1,-1) sobre descuentos negativos
invertia la monotonia. Aqui se definen los candidatos que reemplazan ese diseño.

Todos los motores comparten la interfaz minima sklearn-like:
    fit(x, y, sample_weight=None) -> self
    predict(x) -> np.ndarray
y son picklables (joblib) para que 04_pbi_export.py los reconstruya. Por eso viven en un
modulo importable (no en el script 03 con nombre que empieza por digito).

Invariante OBLIGATORIA (MAESTRO §reglas): monotonia creciente Precio Neto -> Delta.
Cada motor la garantiza por construccion:
  - MotorIsotonico : IsotonicRegression(increasing=True) — escalonado, ancla exacta.
  - MotorXGB1D     : XGBRegressor monotone_constraints=(1,) — referencia (arbol step).
  - MotorSuave     : PCHIP sobre la espina isotonica — escalonado suavizado, monotono.
  - MotorSigmoide  : logistica de 4 parametros con k>0 — asintotas fisicas (piso/techo).
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression

# Hiperparametros XGBoost 1D: misma capacidad reducida que el diseño 3D previo, pero
# con UNA sola feature (Precio Neto) y restriccion monotona directa (+1). Sin el bug de
# signo: a mayor Precio Neto, mayor Delta.
XGB1D_PARAMS = dict(
    n_estimators=100,
    max_depth=2,
    min_child_weight=3,
    learning_rate=0.05,
    reg_lambda=2.0,
    gamma=0.1,
    monotone_constraints=(1,),
    objective="reg:squarederror",
    random_state=42,
    verbosity=0,
)


def _as_1d(x) -> np.ndarray:
    """Aplana cualquier entrada (lista, columna, matriz Nx1) a vector 1D float."""
    return np.asarray(x, dtype=float).ravel()


class MotorIsotonico:
    """Regresion isotonica creciente con clip fuera de rango. Motor PRIMARIO candidato:
    respeta exactamente la escalera de anclas sinteticas sin suavizar los escalones."""

    def __init__(self):
        self.iso = IsotonicRegression(increasing=True, out_of_bounds="clip")

    def fit(self, x, y, sample_weight=None):
        self.iso.fit(_as_1d(x), _as_1d(y), sample_weight=sample_weight)
        return self

    def predict(self, x):
        return self.iso.predict(_as_1d(x))


class MotorXGB1D:
    """XGBoost de una sola feature (Precio Neto) con monotonia +1. Se conserva como
    REFERENCIA del benchmark (el motor del diseño anterior, ya en 1D y sin el bug de
    signo). Sigue siendo un arbol escalonado que pelea contra las anclas."""

    def __init__(self, **params):
        from xgboost import XGBRegressor
        self.m = XGBRegressor(**{**XGB1D_PARAMS, **params})

    def fit(self, x, y, sample_weight=None):
        self.m.fit(_as_1d(x).reshape(-1, 1), _as_1d(y), sample_weight=sample_weight)
        return self

    def predict(self, x):
        return self.m.predict(_as_1d(x).reshape(-1, 1))


class MotorSuave:
    """
    Espina isotonica + interpolacion PCHIP monotona. Mantiene la fidelidad a la escalera
    (la isotonica fija los niveles) pero entrega una curva suave y derivable, util como
    motor de VALIDACION: una "segunda opinion" que no es otra funcion escalon como la
    isotonica pura. PCHIP preserva la monotonia de datos monotonos -> no introduce
    tramos decrecientes. Fuera del rango observado hace clip (igual que la isotonica).
    """

    def __init__(self, n_grid: int = 80):
        self.n_grid = n_grid
        self._iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        self._pchip = None
        self._xmin = self._xmax = None
        self._ymin = self._ymax = None

    def fit(self, x, y, sample_weight=None):
        x = _as_1d(x)
        y = _as_1d(y)
        self._iso.fit(x, y, sample_weight=sample_weight)
        self._xmin, self._xmax = float(x.min()), float(x.max())
        # Grid de soporte para la PCHIP: nodos unicos crecientes en la banda observada
        grid = np.linspace(self._xmin, self._xmax, self.n_grid)
        yp = self._iso.predict(grid)
        # PCHIP exige x estrictamente creciente; linspace ya lo garantiza
        if self._xmax - self._xmin < 1e-9:
            # Campo degenerado (un solo precio): la suave colapsa a constante
            self._pchip = None
            self._ymin = self._ymax = float(yp[0])
        else:
            self._pchip = PchipInterpolator(grid, yp, extrapolate=False)
            self._ymin, self._ymax = float(yp[0]), float(yp[-1])
        return self

    def predict(self, x):
        x = _as_1d(x)
        if self._pchip is None:
            return np.full_like(x, self._ymin)
        xc = np.clip(x, self._xmin, self._xmax)   # clip fuera de banda (igual que isotonica)
        return self._pchip(xc)


def _logistica(x, c, amp, k, x0):
    """Logistica creciente de 4 parametros: piso c, amplitud amp>=0, pendiente k>0.
    f(x) = c + amp / (1 + exp(-k (x - x0)))  -> satura en c (bajo) y c+amp (alto)."""
    return c + amp / (1.0 + np.exp(-k * (x - x0)))


class MotorSigmoide:
    """
    Logistica de 4 parametros ajustada por minimos cuadrados ponderados. Impone por
    diseño las dos asintotas FISICAS de la curva de reservas vs precio: un piso inferior
    (abandono, delta ~ -baseline) y un techo superior (saturacion geologica). Es el
    candidato con menos parametros libres y la forma mas interpretable, pero puede no
    capturar escalones bruscos. Si el ajuste no converge, cae a isotonica (fallback).
    """

    def __init__(self):
        self._params = None
        self._fallback = None

    def fit(self, x, y, sample_weight=None):
        x = _as_1d(x)
        y = _as_1d(y)
        w = np.ones_like(x) if sample_weight is None else _as_1d(sample_weight)

        ymin, ymax = float(np.min(y)), float(np.max(y))
        rango = max(ymax - ymin, 1.0)
        x0_0 = float(np.median(x))
        p0 = [ymin, rango, 0.3, x0_0]
        # c libre, amp>=0 (curva creciente), k en (0, 5] (pendiente positiva acotada)
        bounds = ([ymin - rango, 0.0, 1e-3, float(np.min(x)) - 20],
                  [ymax + rango, 3 * rango, 5.0, float(np.max(x)) + 20])
        sigma = 1.0 / np.sqrt(np.clip(w, 1e-6, None))
        try:
            popt, _ = curve_fit(_logistica, x, y, p0=p0, sigma=sigma,
                                bounds=bounds, maxfev=10000)
            self._params = popt
        except Exception:
            # No convergencia (datos casi planos, N minimo): usar isotonica como respaldo
            self._fallback = MotorIsotonico().fit(x, y, sample_weight=w)
        return self

    def predict(self, x):
        x = _as_1d(x)
        if self._params is not None:
            return _logistica(x, *self._params)
        return self._fallback.predict(x)


# Registro de candidatos del benchmark (nombre -> factory)
CANDIDATOS = {
    "Isotonica": MotorIsotonico,
    "XGBoost1D": MotorXGB1D,
    "Suave":     MotorSuave,
    "Sigmoide":  MotorSigmoide,
}
