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


# Colchon del piso capado: el piso duro queda al menos $1 por debajo de p_ref
# para que el ancla y su vecindad inmediata no caigan en cero (auditoria 2026-07-07 H1)
BK_GUARD_GAP_USD = 1.0


def piso_efectivo(bk_pdp, p_ref, baseline):
    """Piso duro coherente con la certificacion (auditoria 2026-07-07, H1).

    Si el ultimo 1P certificado es positivo AL precio de referencia, un breakeven
    de abandono por ENCIMA de ese precio esta falsificado por la realidad: el campo
    tiene reservas booked a p_ref, luego no esta abandonado a p_ref. Sin este guard,
    42/130 campos predecian Vol=0 exactamente en BRENT_REF (contradiciendo el
    re-anclaje Vol(p_ref)=baseline). Se capa el piso a p_ref − BK_GUARD_GAP_USD y
    el export marca ALERTA_BK=BK_SUPERA_PRECIO_REF para que finanzas revise el BK.
    """
    if bk_pdp is None or not np.isfinite(bk_pdp):
        return None
    if (p_ref is not None and np.isfinite(p_ref)
            and baseline is not None and np.isfinite(baseline) and baseline > 0
            and bk_pdp >= p_ref):
        return p_ref - BK_GUARD_GAP_USD
    return float(bk_pdp)


def volumen_anclado(modelo, pneto, baseline: float, delta_ref: float,
                    bk_pdp: float = None, p_ref: float = None) -> np.ndarray:
    """
    Reconstruccion de volumen con RE-ANCLAJE en el punto actual (decision 2026-06-12):

        Vol(p) = max(baseline + [f(p) − f(p_ref)], 0)

    delta_ref = f(p_ref) se calcula una vez por campo×motor (p_ref = M2(BRENT_REF)).
    Garantiza Vol(p_ref) = baseline: al precio actual las reservas son el ultimo 1P
    certificado. El piso fisico vol=0 bajo el precio de abandono deja de salir de la
    curva (el shift la desplaza) y se impone como regla dura p < piso -> 0, donde el
    piso es bk_pdp capado por piso_efectivo() cuando bk_pdp >= p_ref (H1).
    """
    pneto = _as_1d(pneto)
    vol = np.maximum(baseline + modelo.predict(pneto) - delta_ref, 0.0)
    piso = piso_efectivo(bk_pdp, p_ref, baseline)
    if piso is not None:
        vol = np.where(pneto < piso, 0.0, vol)
    return vol


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


class MotorHuberIso:
    """
    Isotonica con re-pesado robusto IRLS (Huber). Motor candidato para campos con 1-2
    puntos de deck ATIPICOS (recertificacion abrupta de una vigencia) que arrastran la
    escalera: en vez de centrar el año entero (re-anclaje), baja el PESO del punto
    outlier segun su residual. Estado del arte de regresion robusta (Huber 1964) aplicado
    sobre PAVA. Conserva monotonia (sigue siendo isotonica) y no borra la vigencia, solo
    la des-enfatiza si contradice al resto. delta=umbral Huber en desviaciones absolutas
    medianas (MAD); n_iter re-pesados.
    """

    def __init__(self, delta: float = 1.5, n_iter: int = 3):
        self.delta = delta
        self.n_iter = n_iter
        self.iso = IsotonicRegression(increasing=True, out_of_bounds="clip")

    def fit(self, x, y, sample_weight=None):
        x = _as_1d(x)
        y = _as_1d(y)
        w = np.ones_like(x) if sample_weight is None else _as_1d(sample_weight).copy()
        for _ in range(self.n_iter):
            self.iso.fit(x, y, sample_weight=w)
            resid = y - self.iso.predict(x)
            mad = np.median(np.abs(resid - np.median(resid))) or 1.0
            scaled = np.abs(resid) / (1.4826 * mad)
            # Peso Huber: 1 dentro del umbral, decae ~1/|r| fuera (des-enfatiza outliers)
            hub = np.where(scaled <= self.delta, 1.0, self.delta / np.maximum(scaled, 1e-9))
            w0 = np.ones_like(x) if sample_weight is None else _as_1d(sample_weight)
            w = w0 * hub
        self.iso.fit(x, y, sample_weight=w)
        return self

    def predict(self, x):
        return self.iso.predict(_as_1d(x))


class MotorLinealRobusto:
    """
    Recta robusta Theil-Sen en espacio (Precio Neto -> Delta), clampada a monotona.
    Candidato para campos con respuesta al precio APROXIMADAMENTE LINEAL y pocos puntos:
    Theil-Sen (mediana de pendientes por pares) resiste outliers mejor que OLS y no
    inventa escalones donde solo hay una tendencia. Si la pendiente estimada es <=0
    (sin respuesta real al precio) colapsa a constante = mediana de y (curva plana,
    honesta). Monotonia garantizada: pendiente >= 0 por construccion.
    """

    def __init__(self):
        self.slope = 0.0
        self.intercept = 0.0
        self._const = None

    def fit(self, x, y, sample_weight=None):
        from scipy import stats
        x = _as_1d(x)
        y = _as_1d(y)
        if len(np.unique(x)) < 2:
            self._const = float(np.median(y))
            return self
        slope, intercept, _, _ = stats.theilslopes(y, x)
        if slope <= 0:
            # Sin respuesta real al precio -> curva plana (no forzar pendiente espuria)
            self._const = float(np.median(y))
        else:
            self.slope, self.intercept, self._const = float(slope), float(intercept), None
        return self

    def predict(self, x):
        x = _as_1d(x)
        if self._const is not None:
            return np.full_like(x, self._const)
        return self.slope * x + self.intercept


class MotorFCPerfil:
    """
    Curva FINANCIERA PURA desde el perfil de salida del FC (BK_P10..BK_P90), SIN ajustar
    ML a los datos del deck. Candidato para campos donde el deck real es poco fiable o
    inexistente (gas/insensibles, SIN_MODELO con FC) pero SI hay flujo de caja certificado:
    el volumen economico a cada precio sale del FC, no de una regresion con N=9.

        delta(p) = -baseline * fraccion_de_volumen_que_ya_salio(p)

    interpolando el perfil (precio, fraccion) con extremos anclados en BK (0% salido =
    reservas completas) y Precio de Equilibrio (100% salido = abandono). Monotona por
    construccion (fraccion decrece con el precio). fit() es no-op: la curva no depende de y.
    Se construye con los parametros del FC del campo.
    """

    def __init__(self, perfil: dict, baseline: float, bk_sup: float, bk_inf: float):
        # perfil: {10: precio_P10, 25:..., ...}; bk_sup=BK (reservas completas),
        # bk_inf=Precio Equilibrio (abandono total).
        uniq = {round(float(bk_sup), 4): 0.0, round(float(bk_inf), 4): 1.0}
        for pct, pv in (perfil or {}).items():
            if bk_inf < pv < bk_sup:
                uniq[round(float(pv), 4)] = max(uniq.get(round(float(pv), 4), 0.0), pct / 100.0)
        self._xp = np.array(sorted(uniq))
        self._fp = np.array([uniq[k] for k in sorted(uniq)])
        self.baseline = float(baseline)

    def fit(self, x, y=None, sample_weight=None):
        return self  # curva parametrica del FC: no depende de los datos

    def predict(self, x):
        x = _as_1d(x)
        ex = np.clip(np.interp(x, self._xp, self._fp), 0.0, 1.0)
        return -ex * self.baseline


class MotorPlano:
    """Curva constante = mediana de los deltas de entrenamiento. Motor superior
    candidato del hibrido para campos donde la parte alta del deck es plana
    (insensible al precio por encima de la banda de breakevens): la constante es
    la prediccion honesta y no inventa pendiente. Monotona trivialmente."""

    def __init__(self):
        self._const = 0.0

    def fit(self, x, y, sample_weight=None):
        self._const = float(np.median(_as_1d(y)))
        return self

    def predict(self, x):
        return np.full_like(_as_1d(x), self._const)


class MotorHibrido:
    """
    Modelo hibrido por tramos (directriz usuario 2026-07-15): la parte BAJA de la
    curva conserva la escalera de breakevens (isotonica sobre sinteticos + reales,
    semantica financiera ratificable intacta) y la parte ALTA usa el motor candidato
    que mejor generaliza para ESE campo (Sigmoide, LinealRobusto, HuberIso, Suave,
    Plano) entrenado solo con la evidencia real.

    Esto resuelve la objecion que rechazo los swaps de motor completos (s8/s9,
    'cambia la FORMA de la curva'): aqui la forma financiera (escalera/hard-zero)
    la sigue gobernando la isotonica en [<= p_junta]; el candidato solo gobierna
    donde la evidencia es real (> p_junta), que es donde antes la isotonica
    interpolaba escalones espurios.

    p_junta la calcula el CALLER por campo (max(techo banda BK + transicion, fin de
    la evidencia sintetica)) — no es constante global porque la junta puede caer
    dentro del rango real en campos con BK alto (ej. CAÑO SUR ESTE).

    Garantias:
      - Continuidad C0: el tramo superior se desplaza (offset) para empatar
        exactamente el valor de la isotonica en la junta.
      - Monotonia global: ambos tramos son monotonos por construccion y el tramo
        superior arranca EN el valor de la junta con un piso (np.maximum) que
        impide caer por debajo — verificable en grid $40-120.
      - Asintota superior (C3 NORTE): el tramo superior se recorta al maximo delta
        observado en el entrenamiento (np.minimum), igual que la isotonica satura
        via out_of_bounds='clip'. Sin este techo, motores como LinealRobusto
        extrapolan crecimiento indefinido y violan la fisica de saturacion.
      - Compatible con volumen_anclado (C5 re-anclaje exacto y C6 hard-zero: el
        hibrido es un modelo 1D mas con la misma interfaz predict).
    """

    def __init__(self, motor_superior_factory, p_junta: float):
        self.p_junta = float(p_junta)
        self._factory = motor_superior_factory
        self._iso = MotorIsotonico()
        self._sup = None
        self._offset = 0.0
        self._y_junta = 0.0
        self._y_techo = None  # asintota: maximo delta visto en entrenamiento

    def fit(self, x, y, sample_weight=None):
        x = _as_1d(x)
        y = _as_1d(y)
        w = np.ones_like(x) if sample_weight is None else _as_1d(sample_weight)

        # Tramo inferior: isotonica sobre TODOS los puntos (sinteticos anclan la
        # escalera igual que en el champion de Produccion)
        self._iso.fit(x, y, sample_weight=w)
        self._y_junta = float(self._iso.predict([self.p_junta])[0])

        # Tramo superior: candidato entrenado SOLO con los puntos por encima de la
        # junta (por construccion de p_junta, alli solo hay evidencia real: los
        # sinteticos viven en la banda de breakevens, debajo de la junta)
        mask_sup = x >= self.p_junta
        if mask_sup.sum() >= 2 and np.ptp(x[mask_sup]) > 1e-6:
            self._sup = self._factory().fit(x[mask_sup], y[mask_sup],
                                            sample_weight=w[mask_sup])
            # Continuidad C0: desplazar el tramo superior para empatar la junta
            self._offset = self._y_junta - float(self._sup.predict([self.p_junta])[0])
            # Asintota (C3): el delta no puede superar el maximo historico del
            # campo — mismo comportamiento de saturacion que la isotonica con
            # out_of_bounds='clip'. Se toma el max de TODO el entrenamiento
            # (reales + sinteticos) porque ese es el techo de evidencia.
            self._y_techo = float(np.max(y))
        else:
            # Sin evidencia real suficiente arriba: el hibrido degenera a isotonica
            self._sup = None
            self._offset = 0.0
        return self

    def predict(self, x):
        x = _as_1d(x)
        base = self._iso.predict(x)
        if self._sup is None:
            return base
        sup = self._sup.predict(x) + self._offset
        # Piso en el valor de la junta: el tramo superior (monotono) nunca puede
        # caer bajo el nivel donde termina la escalera -> monotonia global
        sup = np.maximum(sup, self._y_junta)
        # Techo de asintota (C3): saturar en el maximo delta de entrenamiento,
        # igual que hace la isotonica al extrapolar (clip). Mantiene monotonia
        # (min de dos funciones no-decrecientes con constante) y la continuidad
        # C0 (en la junta sup = y_junta <= y_techo por construccion).
        if self._y_techo is not None:
            sup = np.minimum(sup, self._y_techo)
        return np.where(x <= self.p_junta, base, sup)


# Registro de candidatos del benchmark (nombre -> factory)
CANDIDATOS = {
    "Isotonica": MotorIsotonico,
    "XGBoost1D": MotorXGB1D,
    "Suave":     MotorSuave,
    "Sigmoide":  MotorSigmoide,
    "HuberIso":  MotorHuberIso,
    "LinealRobusto": MotorLinealRobusto,
}

# Motores candidatos para el TRAMO SUPERIOR del hibrido (todos monotonos)
CANDIDATOS_HIBRIDO = {
    "Sigmoide":      MotorSigmoide,
    "LinealRobusto": MotorLinealRobusto,
    "HuberIso":      MotorHuberIso,
    "Suave":         MotorSuave,
    "Plano":         MotorPlano,
}
