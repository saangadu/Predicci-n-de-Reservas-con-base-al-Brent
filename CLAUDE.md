# CLAUDE.md — Prediccion (Piloto Predicción Reservas 1P vs Brent)

## Propósito

Modelo sustituto (surrogate model) que estima la sensibilidad volumétrica de reservas 1P frente
al Precio Neto del crudo (Brent − Descuento Calidad − Descuento Transporte). Soporta decisiones
de CAPEX en la Gerencia de Desarrollo de Ecopetrol.

**NO reemplaza**: SEC / EcoFaro / ARIES / Planning Space. Es soporte analítico interno.

**Proyecto top-level independiente**: Pipeline Python/ML desligado del tablero `.pbix` Histórico BU.
Las reglas DAX y Power Query de ese proyecto no aplican aquí.

## Documento de referencia obligatorio

Leer `docs/MAESTRO.md` ANTES de modificar cualquier archivo de este subproyecto.
Ese documento contiene la fuente de verdad: inventario de datos, esquema del tablón único,
motor matemático, decisiones vigentes, supuestos y limitantes. Actualizar tras cada sesión.

## Alcance actual (Fase 2 — completada 2026-06-30)

- **Campos**: portafolio completo con homologación a UNIFICADO (34 familias fusionadas; clave final = DIM_CAMPO.xlsx columna UNIFICADO).
- **Gate dorado**: CASTILLA, CASTILLA NORTE, CASTILLA ESTE, RUBIALES (siempre validados).
- **Tests**: 7/7 gates PASS. `run_pipeline.py` exit 0 (7 fases, ~12 min). Corrida 2Q 2026 completada.
- **Arquitectura vigente**:
  - M2 solo-HIST: Theil-Sen `Aceite = α + β·Brent` entrenado únicamente con cierres reales (sin quarters Consolidado). R² mediana 0.963; fallback k·Brent para 157 campos sin historia (`ES_FALLBACK=True`).
  - M1 re-anclado: `Vol(p) = max(baseline + [f(p) − f(p_ref)], 0)`. `p_ref = M2(BRENT_REF=68.01 USD/bbl)`. Garantiza Vol(BRENT_REF) = 1P certificado más reciente. Hard-zero: `p_neto < BK_PDP → Vol=0`.
  - 3 matrices de output: `output_matriz_modelo1.csv` (M1 puro), `output_matriz_modelo2.csv` (M2 puro), `output_matriz_prediccion.csv` (cadena completa).
- **Aviso PBI**: `ESCENARIO_DESCUENTO` eliminado del output — actualizar DAX antes de publicar tablero.
- **Fase 3+**: VECM, Kalman, DCA, OOIP cap (backlog).

## Stack tecnológico

```
pandas>=2.1        openpyxl>=3.1      scikit-learn>=1.3
numpy>=1.24        xgboost>=2.0       joblib (incluido en scikit-learn)
matplotlib>=3.7    pyarrow>=14.0      scipy>=1.11
```

Instalar: `pip install -r requirements.txt`

## Estructura de carpetas

```
Prediccion/
├── CLAUDE.md                  ← este archivo
├── requirements.txt
├── 01_etl.py                  ← ingesta + tablón único
├── 02_synthetic.py            ← inyección puntos sintéticos breakeven
├── 03_modelo.py               ← Modelo 1: Isotónica (primario) + Suave/PCHIP + LOO-CV + plots
├── 03b_correlacion_brent.py   ← Modelo 2: Theil-Sen Aceite = α+β·Brent (solo HIST, sin escenarios)
├── 04_pbi_export.py           ← 3 matrices Brent→(M2)→Neto→(M1)→Volumen para Power BI
├── motores_modelo1.py         ← motores 1D candidatos (Isotónica, XGB-1D, Suave, Sigmoide)
├── benchmark_modelo1.py       ← (offline) benchmark LOO-CV de motores 1D
├── 06_comparativa_bk.py       ← (offline) anclaje BK ponderado vs clase de mayor incertidumbre
├── experimento_vigencia.py    ← (offline) experimento centrado por año (descartado, ver MAESTRO §12)
├── datos/
│   ├── raw/                   ← inputs originales (NO modificar)
│   │   ├── HIST 1P.xlsx
│   │   ├── Consolidado Bases de Datos Pronostico de Precios.xlsx
│   │   ├── Breakeven.xlsm
│   │   └── Codigo Breakeven
│   └── staging/               ← outputs intermedios del pipeline
│       ├── tablon_unico.parquet / .csv
│       ├── metricas.csv
│       ├── correlacion_brent.csv      ← coeficientes Modelo 2
│       ├── modelos/           ← {campo}_iso.joblib, {campo}_suave.joblib
│       └── plots/ , plots_correlacion/
├── docs/
│   ├── MAESTRO.md             ← fuente de verdad (leer siempre primero)
│   ├── NORTE.md               ← contrato de no-regresión (gate dorado)
│   ├── CHANGELOG_PREDICCIONES.md
│   ├── Breakeven Resumen Técnico
│   └── archivo/               ← docs superados (3D, Path D, Deep Research) — ver README
└── resultados/                ← output_matriz_modelo1/modelo2/prediccion.csv para Power BI
```

## Idioma y estilo de código

- Código y comentarios en **español**
- Comentarios PEP-8: explicar el POR QUÉ del negocio, no el qué del Python
- Sin abstracciones prematuras: 3 líneas repetidas > helper innecesario
- Bloque `if __name__ == "__main__":` en cada script

## Reglas del motor matemático

| Regla | Detalle |
|---|---|
| Monotonía obligatoria | Brent↑ → Reservas↑. Nunca violar (garantizada por construcción en M1 y M2). |
| Arquitectura 2 modelos (2026-06-12) | M1: Isotónica (primario) + Suave/PCHIP (validación), `Vol(p)=max(baseline+[f(p)−f(p_ref)],0)`. M2: Theil-Sen `Aceite=α+β·Brent` solo-HIST (β>0); fallback k·Brent taggeado. XGBoost 3D **retirado** (ver `docs/archivo/`). |
| Isotónica primaria | `increasing=True, out_of_bounds='clip'` siempre presente |
| RF / XGBoost descartados | RF suaviza step functions; XGBoost-1D fue el peor del benchmark → inviables como primario |
| Modelos por campo | NUNCA un modelo global. 1 motor M1 + 1 recta M2 por campo. |
| Anclaje sintético | Puntos (precio < breakeven, vol=0) siempre inyectados; BK_ANCLA_FIN = clase de mayor incertidumbre (PND→PNP→PDP) |
| Validación LOO-CV | `LeaveOneOut` por campo (N≈10 puntos → hold-out inestable); también en M2 (R2_LOO/MAE_LOO) |
| Métricas piloto | R²_LOO, MAE_LOO, RMSE_LOO, SKILL son REFERENCIALES, no KPIs de producción |

## Sanity checks funcionales (criterio de éxito)

1. `p_neto < BK_PDP` → Vol = 0 (hard-zero, C6)
2. `Vol(BRENT_REF) = baseline` del campo exacto (re-anclaje, C5, tolerancia < 0.5 MBPE)
3. Monotonía en plots (sin tramos decrecientes)
4. Asíntota superior visible (saturación cerca del máximo histórico)
5. Divergencia Isotónica ↔ Suave < 30% en banda histórica observada ($40–$80)
6. M2 `ES_FALLBACK=True` visible y taggeado en los 3 outputs para campos sin historia propia

## Notas críticas de parsing

| Archivo | Nota |
|---|---|
| `HIST 1P.xlsx / Reservas` | Decimal `,` ("0,000") → usar `.str.replace(',', '.').astype(float)` |
| `HIST 1P.xlsx / Precio` | Decimal `.` — lectura directa |
| `Sensibilidad...` | Header real en row 3, skiprows=2. Activo col 3, CAMPO col 4. |
| Breakeven campo | Extraer nombre con regex `^(.+?)_CF_SEC_.*` del nombre de archivo |
| Todos | `engine='openpyxl'` para preservar ñ/tildes |

## Glosario mínimo

| Término | Definición |
|---|---|
| 1P | Reservas Probadas = PDP + PNP + PND |
| PDP | Probadas Desarrolladas Produciendo |
| PNP | Probadas No Produciendo |
| PND | Probadas No Desarrolladas |
| PRB | Probables |
| PS | Posibles |
| MBPE | Miles de Barriles de Petróleo Equivalente |
| OOIP | Petróleo Original en Sitio (asíntota geológica máxima) |
| EUR | Reservas Recuperables Últimas |
| EL | Límite Económico (Economic Limit) — precio mínimo para que el campo sea rentable |
| Breakeven Operacional | Precio donde FC del **último año con aceite** (`_ult_oil`) = 0 (Límite Económico / precio de abandono). NO el acumulado. Aplica a clases PDP (capex hundido). Motor: `brentq` en `motor_breakeven.py`. |
| Breakeven Financiero | Precio donde VPN = 0 (Goal Seek Newton-Raphson) — más conservador |
| Precio Neto | Brent − Descuento Calidad − Descuento Transporte |
| Brent Flat | Precio Brent constante hipotético para escenarios |
| Netback | Ingreso neto por barril después de descuentos y costos de transporte |
| CAPEX | Capital Expenditure — inversión en desarrollo de reservas |
| OPEX | Operational Expenditure — costos de operación |
