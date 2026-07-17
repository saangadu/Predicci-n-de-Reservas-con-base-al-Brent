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

## Alcance actual (Fase 2 + Promoción total s14 — 2026-07-17)

- **s14 (2026-07-17) — Promoción total Calidad→Producción + bandas P25/P75 cuadratura (detalle: `docs/MAESTRO.md` §10 s14):**
  - `PRED_M2_SELECCION` ratificado (`FLAGS_RATIFICADOS`); registros M1 (23) y M2 (111) `ADOPTADO_CALIDAD→ADOPTADO`. Producción ≡ Calidad.
  - Bandas: cuantiles LOYO **P25/P75** por campo (`VOL_P25/P75_MBPE`); portafolio DAX en **cuadratura** de semianchos (antes suma → 84 MBPE irreales).
  - Consolidado actualizado: FLOREÑA serie deck completa; NARE UNIFICADO = fila propia del Excel (1P = Σ componentes ✓; precio del Excel ≠ ponderado por 1P, dif ~0.2 USD — anotado).
  - Taxonomía tablero: `DICCIONARIO.md §12` (Modelo/Rol/Método/Tipo de Curva/Ambiente); dim gana `METODO_REAL_PRIMARIO`; `[Método Real]` reemplaza `[Método del Motor]`.
  - Precios 2026_Q2 presentes sin reservas → Q2 sigue siendo el objetivo (no entrena).

## Alcance previo (Fase 2 + Expansión portafolio M2/híbrido s12 — 2026-07-15)

- **s12 (2026-07-15) — Portafolio completo + validación dinámica + umbral >5% (detalle: `docs/MAESTRO.md` §10 s12):**
  - Universo = campos con modelo (~184); DECK_PLANO excluido del benchmark híbrido.
  - Adopción Calidad: mejora MAE **>5%** → `ADOPTADO_CALIDAD` (sin EN_REVISION). `seleccion_reglas.py`.
  - Híbrido: ranking LOYO; **validación = 2.º mejor** (no Suave fija); `METODO_VALIDACION` en registro y pipeline.
  - M2: portafolio explícito + umbral >5% (antes ≥10% post-s11).
  - `PRED_M2_SELECCION` auto-ON si `PRED_TRACK=calidad` (`track.FLAGS_ON_EN_CALIDAD`); OFF en Producción; `--sin-flags` fuerza `"0"`.
  - Export: columnas `METODO_PRIMARIO` / `METODO_VALIDACION` / `METODO_REAL`.

## Alcance previo (Fase 2 + Rigor total Calidad + investigación M2/híbridos — 2026-07-15)

- **s11 (2026-07-15) — Calidad con gates completos + M2 por familias + híbridos M1 (detalle: `docs/MAESTRO.md` §10 s11):**
  - **Calidad = mismo rigor que Producción**: `run_pipeline.py --calidad` corre pytest por fase + gate NORTE + `tests_calidad/`. G1/G2 duros en ambos tracks; G3/G5 comparativos en Calidad → `resultados_calidad/norte_divergencias.csv` (warning, no aborta). Nuevo `rutas_track.py` (paths por `PRED_TRACK`); los 8 tests de `tests/` lo importan.
  - **M2 por familias** (`m2_familias.py` + `analisis_m2.py`, offline): THEILSEN / DESCOMPUESTO (reincorpora Descuento de Calidad como f(Brent)) / SEGMENTADA / HUBER / CUADRATICA_MONOTONA, evaluadas por campo × dataset (solo-HIST vs HIST+CONSOLIDADO) con LOO-CV + gate físico. 92 campos adoptados (mejora media MAE_LOO 32.5%) → `seleccion_metodos_m2.csv`. Dispatch en `03b` bajo flag `PRED_M2_SELECCION` (ON en Calidad); `neto_desde_brent` generalizada vía `M2_PARAMS` JSON. Reporte `m2_r2_bajo.csv` (materiales con R2_LOO<0.9).
  - **Híbridos M1** (`MotorHibrido` en `motores_modelo1.py` + `analisis_hibrido.py`): escalera isotónica de breakevens en `p ≤ p_junta` + motor candidato (Sigmoide/LinealRobusto/HuberIso/Suave/Plano) arriba, con C0, monotonía global y techo de asíntota (= máx delta de entrenamiento; sin él violaba G1 `asintota_iso`). 5 adopciones top-20: CASTILLA Sigmoide +28%, CASTILLA NORTE LinealRobusto +28%, PALAGUA Plano +27%, YARIGUI-CANTAGALLO LinealRobusto +19%, CHICHIMENE Suave +16%.
  - **Gobernanza `ADOPTADO_CALIDAD`**: Producción solo aplica `ADOPTADO`; Calidad aplica `ADOPTADO`+`ADOPTADO_CALIDAD` (ambos registros de selección). Promoción requiere ratificación de Finanzas.
  - A/B (`comparar_tracks.py`, mecanismos acumulativos `B_confound+C_hibrido+D_m2`): MAE_LOYO gate CASTILLA 1.77→1.27, CHICHIMENE 4.27→3.58, CASTILLA NORTE 1.69→1.21; dif volúmenes @Brent 60/70/80 < 2 MBPE. Producción intacta: 229 passed / 9 skipped, sin re-freeze.

## Alcance previo (Fase 2 + Promoción Calidad→Producción — 2026-07-15)

- **s10 (2026-07-15) — Promoción track Calidad a Producción + fixes tablero (detalle: `docs/MAESTRO.md` §10):**
  - Flags `PRED_PERFIL_SALIDA`/`PRED_SELECCION_METODO`/`PRED_BANDAS_LOYO` default ON en todo track (`track.py::FLAGS_RATIFICADOS`); Producción≡Calidad verificado bit a bit. `--sin-flags` = paridad legacy.
  - Bug corregido: `02_synthetic.py` perfil de salida daba volumen negativo en el cap por dato histórico (MACANA, 15 filas) — invisible porque Calidad omitía pytest. Hard-zero agregado.
  - DIM_CAMPO: 4 merges (REX NE, UNDERRIVER, TERECAY (COSECHA G), COSECHA) + fix casing CARACARA (resolvía campo "(Blank)" del tablero). 188→184 campos.
  - Rename `SOLO_GAS`→`INSENSIBLE_PRECIO` (enum + medida TMDL).
  - Reconciliación baseline vs cierre certificado 1685 MBPE: `reconciliacion_baseline.py` (nuevo, offline). Export 1703.5; gap +18.5 desglosado (carry-forward de campos sin reporte 2025 + ruido de redondeo).
  - DAX bandas P10/P90 corregido (COALESCE a predicción central evita que la banda agregada quede por debajo de la curva) + franjas visuales en 3 páginas del tablero.
  - NORTE re-freeze: ALTA 45→52, MEDIA 24→18, SIN_MODELO 57→53. 56 passed / 9 skipped.

## Alcance actual previo (Fase 2 + Normalización v2 + Agregación v3 + Coherencia BK↔deck — 2026-07-10)

- **s5 (2026-07-10) — Directriz escalera/deck (`docs/DIRECTRIZ_ESCALERA_DECK.md`, ratificación finanzas pendiente):**
  - **Flag BK↔deck** (`01_etl.py`): `ALERTA=BK_CONTRADICHO_POR_DECK` + `resultados/bk_revision_finanzas.csv` (15 campos materiales ≥5 MBPE: Caño Limón, Chichimene, Casabe-like). NO cambia curvas.
  - **Escalera suavizada** (`02_synthetic.py`): `TRANSICION_USD=6`, `CLIFF_FRAC=0.15`; escalón no cae >15%·baseline bajo el peor delta real junto a la banda (`ALERTA=ESCALON_SUAVIZADO`). Caño Limón sesgo-recup 36%→20%, recuperación Brent80 +34%→+18.7%.
  - **Modelo plano deck** (`03_modelo.py::es_deck_plano` + `04`): rango deltas < 2%·baseline en ≥2 vigencias (N≥4) → curva plana = baseline sobre BK_PDP, `TIPO_MODELO=PLANO_DECK` (13 campos). PAUTO: −40%@Brent60 → plano 108.5.
  - **Gates LOYO** (`04::clasificar_confianza`): `N_VIGENCIAS_LOYO≥2` → gate usa `MAE_LOYO`/`SKILL_LOYO`; LOO fallback. Motivo muestra `metrica=LOYO` + ambos skills.
  - **Cobertura total** (`04::emitir_cobertura_plana`): cierre 2025 sin modelo → línea plana `NIVEL=SIN_MODELO` (57); `SOLO_SINTETICO`→`SOLO_GAS` (5). Export 131→**188 campos** (filiales fuera). TMDL: `# Campos Solo Gas` + `# Campos Sin Modelo`.
  - **Peso de recencia RECHAZADO** (`experimento_recencia.py`): 2×/3× a la última vigencia EMPEORA MAE_LOYO (N=9 amplifica ruido). Peso uniforme 1.0 confirmado.
  - **TECA→AREA TECA-COCORNA** fusionado vía DIM (usuario): ahora modelo real ALTA.
  - NORTE re-freeze: ALTA 43→45, MEDIA 25→24, SOLO_GAS=5, SIN_MODELO=57, CONFOUND 19→14, SESGO 13→12. 223 passed / 9 skipped.

## Alcance previo (Fase 2 + Normalización v2 + Agregación v3 — 2026-07-09)

- **Campos**: portafolio completo con homologación a UNIFICADO (clave final = DIM_CAMPO.xlsx columna UNIFICADO). **Agregación v3 (s3)**: los componentes físicos de un UNIFICADO que coexisten (APIAY=APIAY+GAVAN+GUATIQUIA; LISAMA=LISAMA+NUTRIA+TESORO; etc.) **suman** su 1P en `01_etl.py` (antes coalesce `.first()`, que botaba volumen y falseaba la serie); precios ponderados por volumen (`_prom_ponderado`). PAUTO SUR fusionado en PAUTO; **CHICHIMENE SW re-fusionado en CHICHIMENE** vía `ALIAS_OVERRIDE` (serie continua 2023=149.7→2024=152.5→2025=174.3).
- **Gate dorado (pareto-9)**: RUBIALES, CASTILLA, CAÑO SUR ESTE, CASTILLA NORTE, AKACIAS, CHICHIMENE, LA CIRA, CUPIAGUA, YARIGUI-CANTAGALLO. PAUTO excluido (deck con sensibilidad=0 desde 2025_Q1, y 2026_Q1 aún 0). Excepciones G2 en `tests/test_norte.py::G2_EXCEPCIONES` (CHICHIMENE: solo SKILL, LOO negativo por salto de recertificación; MAE_REL pasa limpio).
- **Tests**: 224 passed + 9 skipped (exenciones visibles). `run_pipeline.py` exit 0 (7 fases, ~12 min).
- **Arquitectura vigente**:
  - **Normalización v2 (2026-07-09)**: `DELTA_SENS_MBPE = VOL_SENS − BASELINE_1P_MBPE` donde `BASELINE_1P_MBPE` = cierre OFICIAL del año ANTERIOR (A−1) — la base real del deck. `CHECKPOINT_1P_MBPE` (cierre del mismo año A) queda como referencia del confound. A/B: MAE_LOYO gate 13.1→1.5 MBPE; confound (tras v3) 16 flags. 2026_Q1 entrena (N=9).
  - M2 solo-HIST: Theil-Sen `Aceite = α + β·Brent` entrenado únicamente con cierres reales (sin quarters Consolidado). Fallback k·Brent para campos sin historia (`ES_FALLBACK=True`).
  - M1 re-anclado: `Vol(p) = max(baseline + [f(p) − f(p_ref)], 0)`. `p_ref = M2(BRENT_REF=68.64 USD/bbl = cierre oficial 2025)`. Garantiza Vol(BRENT_REF) = 1P certificado del cierre. Hard-zero: `p_neto < BK_PDP → Vol=0`.
  - 3 matrices de output: `output_matriz_modelo1.csv` (M1 puro), `output_matriz_modelo2.csv` (M2 puro), `output_matriz_prediccion.csv` (cadena completa).
- **Aviso PBI**: `ESCENARIO_DESCUENTO` eliminado del output — actualizar DAX antes de publicar tablero. Columnas del export estables tras normalización v2 (cambian los VALORES: curvas más planas).
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
├── motores_modelo1.py         ← motores 1D candidatos (Isotónica, XGB-1D, Suave, Sigmoide, Plano, Híbrido)
├── m2_familias.py             ← familias M2 (Theil-Sen, Descompuesto, Segmentada, Huber, Cuadrática) + física + LOO
├── rutas_track.py             ← paths STAGING/RESULTADOS parametrizados por PRED_TRACK
├── analisis_m2.py             ← (offline) benchmark familias M2 por campo × dataset → seleccion_metodos_m2.csv
├── analisis_hibrido.py        ← (offline) champion-challenger híbridos M1 top-20 → seleccion_metodos.csv
├── comparar_tracks.py         ← (offline) A/B Calidad vs Producción (comparativa_tracks.csv + plots)
├── benchmark_modelo1.py       ← (offline) benchmark LOO-CV de motores 1D
├── 06_comparativa_bk.py       ← (offline) anclaje BK ponderado vs clase de mayor incertidumbre
├── experimento_vigencia.py    ← (offline) experimento centrado por año (descartado, ver MAESTRO §12)
├── datos/
│   ├── raw/                   ← inputs originales (NO modificar)
│   │   ├── HIST 1P.xlsx
│   │   ├── Consolidado Bases de Datos.xlsx
│   │   ├── Breakeven.xlsm
│   │   └── Codigo Breakeven
│   ├── staging/               ← outputs intermedios del pipeline (Producción)
│   │   ├── tablon_unico.parquet / .csv
│   │   ├── metricas.csv
│   │   ├── correlacion_brent.csv      ← coeficientes Modelo 2
│   │   ├── modelos/           ← {campo}_iso.joblib, {campo}_suave.joblib
│   │   └── plots/ , plots_correlacion/
│   └── staging_calidad/       ← espejo del staging para el track Calidad (PRED_TRACK=calidad)
├── tests/                     ← gates por fase + NORTE (ambos tracks, via rutas_track.py)
├── tests_calidad/             ← gates de selección de método (M1 híbridos + M2 familias)
├── docs/
│   ├── MAESTRO.md             ← fuente de verdad (leer siempre primero)
│   ├── NORTE.md               ← contrato de no-regresión (gate dorado)
│   ├── CHANGELOG_PREDICCIONES.md
│   ├── Breakeven Resumen Técnico
│   └── archivo/               ← docs superados (3D, Path D, Deep Research) — ver README
├── resultados/                ← output_matriz_modelo1/modelo2/prediccion.csv para Power BI (Producción)
└── resultados_calidad/        ← outputs del track Calidad + seleccion_metodos*.csv + norte_divergencias.csv
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
| **Breakeven** *(antes "Breakeven Operacional"; renombre 2026-07-10)* | Precio mínimo para mantener el **Límite Económico** (y por ende las reservas) del FC analizado: FC del **último año con aceite** (`_ult_oil`) = 0. Piso SUPERIOR de la escalera. Columna `BREAKEVEN_USD_BBL`. Motor lo etiqueta internamente `"operacional"`. Ver `docs/DICCIONARIO.md` §3. |
| **Precio de Equilibrio** *(antes "Breakeven Financiero")* | Precio donde **VPN = 0**; por debajo no hay reservas (abandono total). Piso INFERIOR de la escalera. Columna `PRECIO_EQUILIBRIO_USD_BBL`. Motor lo etiqueta internamente `"financiero"`. |
| Precio Neto | Brent − Descuento Calidad − Descuento Transporte |
| Brent Flat | Precio Brent constante hipotético para escenarios |
| Netback | Ingreso neto por barril después de descuentos y costos de transporte |
| CAPEX | Capital Expenditure — inversión en desarrollo de reservas |
| OPEX | Operational Expenditure — costos de operación |
