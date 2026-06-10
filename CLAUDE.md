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

## Alcance actual (Fase 1 — completada 2026-06-05)

- **Campos**: 135 campos `Incluir=Si` del Consolidado (portafolio completo)
- **Gate dorado**: CASTILLA, CASTILLA NORTE, CASTILLA ESTE, RUBIALES (siempre validados)
- **Tests**: 88/88 verdes. `run_pipeline.py` exit 0 (6 fases, ~10 min). Listo para corrida 2Q 2026.
- **Fase 2+**: VECM, Kalman, DCA, OOIP cap (backlog).

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
├── 03_modelo.py               ← XGBoost + Isotónica + LOO-CV + plots
├── 04_pbi_export.py           ← meshgrid para Power BI
├── datos/
│   ├── raw/                   ← inputs originales (NO modificar)
│   │   ├── HIST 1P.xlsx
│   │   ├── Consolidado Bases de Datos Pronostico de Precios.xlsx
│   │   ├── Breakeven.xlsm
│   │   └── Codigo Breakeven
│   └── staging/               ← outputs intermedios del pipeline
│       ├── tablon_unico.parquet
│       ├── tablon_unico.csv
│       ├── metricas.csv
│       ├── modelos/           ← {campo}_xgb.joblib, {campo}_iso.joblib
│       └── plots/             ← {campo}.png
├── docs/
│   ├── MAESTRO.md             ← fuente de verdad (leer siempre primero)
│   ├── Modelos de Predicción de Reservas O&G.md   ← Deep Research (75 KB)
│   └── Breakeven Resumen Técnico
└── resultados/                ← output_matriz_prediccion.csv para Power BI
```

## Idioma y estilo de código

- Código y comentarios en **español**
- Comentarios PEP-8: explicar el POR QUÉ del negocio, no el qué del Python
- Sin abstracciones prematuras: 3 líneas repetidas > helper innecesario
- Bloque `if __name__ == "__main__":` en cada script

## Reglas del motor matemático

| Regla | Detalle |
|---|---|
| Monotonía obligatoria | Brent↑ → Reservas↑. Nunca violar. |
| XGBoost primario | `monotone_constraints='(1,-1,-1)'` siempre presente |
| Isotónica secundaria | `increasing=True, out_of_bounds='clip'` siempre presente |
| RF descartado | Bagging suaviza step functions → inviable para auditorías |
| Modelos por campo | NUNCA un modelo global. 1 XGBoost + 1 Isotónica por campo. |
| Anclaje sintético | Puntos (precio < breakeven_financiero, vol=0) siempre inyectados |
| Validación LOO-CV | `LeaveOneOut` por campo (N≈10 puntos → hold-out inestable) |
| Métricas piloto | R²_LOO, MAE_LOO, RMSE_LOO son REFERENCIALES, no KPIs de producción |

## Sanity checks funcionales (criterio de éxito del piloto)

1. `Precio Neto < Breakeven` → predicción ≈ 0 en ambos motores
2. Monotonía visible en plots (sin tramos decrecientes)
3. Asíntota superior visible (saturación cerca del máximo histórico)
4. Divergencia XGBoost ↔ Isotónica < 30% en banda histórica observada ($40–$80)

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
