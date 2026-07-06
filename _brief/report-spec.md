# Report Spec — Predicción de Reservas

## Report identity
- Report name: Predicción de Reservas 1P vs Brent
- Semantic model: Predicción de Reservas (live PBI Desktop; star schema, 24 measures)
- Audience: Gerencia de Desarrollo (decisión CAPEX) + analista técnico (drill)
- Primary purpose: leer el 1P del portafolio a un Brent de escenario, ver la sensibilidad al precio, y saber dónde el modelo es confiable
- Delivery target: PBIP local (sin publicación Fabric)

## User decisions and constraints
- Scope: 4 páginas (Exec + Drill) + 1 página oculta de diagnóstico drill-through
- Page count: 5 (4 visibles + 1 drill-through)
- Interactivity: slicer Brent sincronizado en páginas 1–3; slicer Campo/Motor solo en Explorador; drill-through a diagnóstico por campo
- Design direction: Brand-Forward Ecopetrol (verde volúmenes / amarillo precio / slate texto)
- Publishing: no
- Tooling: powerbi-report-author + powerbi-desktop CLIs; MCP para medidas
- Model edit permissions: sí (medidas de título ya creadas)
- Accessibility: WCAG AA; amarillo solo como fill con texto oscuro; alt text por visual
- Data caveats: 2 campos sin fila DIM (`ACAE-SAN MIGUEL`, `CARACARA SUR B Y C`) — corrección del usuario en Excel. Modelo actualmente Unprocessed en Desktop → refrescar antes de screenshots.

## Narrative
- Core story: "El 1P del portafolio responde de forma monótona al Brent; a $X el portafolio vale Y MBPE, y estos campos son los que mueven la aguja y los que están cerca de su breakeven."
- Audience promise: una cifra defendible de 1P por escenario de precio, con trazabilidad al motor por campo.
- Key questions: (1) ¿Cuánto 1P a este Brent y cuánto vs certificado? (2) ¿Qué campos son sensibles / están en riesgo por breakeven? (3) ¿Cómo responde ESTE campo? (4) ¿Dónde NO confiar en el modelo?

## Design identity
- Tone: Brand-Forward Ecopetrol (remix de Corporate Cool con paleta corporativa)
- Signature: banda de escenario superior — título dinámico a la izquierda + slicer Brent + KPI "Brent Escenario" en chip amarillo; se repite en páginas 1–3
- Color semantics: verde `#009639` = volúmenes/positivo; amarillo `#FFCD00` = precio/Brent (solo fill; línea de precio en oro oscuro `#A87900`); rojo `#C2410C` = alerta (no-viable, frágil, outlier); slate `#1E293B` = texto

## Page plan
1. **Resumen Ejecutivo** — Executive Summary, Variant B (KPI-strip; 5 KPIs sin héroe único, con curva de portafolio como evidencia)
2. **Sensibilidad del Portafolio** — Comparative Benchmark, Variant A (ranking + scatter de riesgo + detalle)
3. **Explorador de Campo** — Analytical Canvas, Variant A (filter-rail; curva por campo)
4. **Confiabilidad del Modelo** — Operational Monitor, Variant A (4-up status de calidad)
5. **Diagnóstico Campo** (oculta) — drillthrough desde Campo (curvas M1 + recta M2 + coeficientes)

## Model requirements
- Existing measures: 22 base (Vol 1P Predicho, Delta vs Base, # Campos Viables, Elasticidad Local, Colchón vs BK Ope/Fin, % Campos Fallback M2, # Campos Confianza *, % Vol Extrapolado, etc.)
- New measures (creadas vía MCP): `Título Escenario`, `Título Campo`
- New calculated columns: ninguna (orden de confianza se maneja con 4 medidas de conteo separadas)
- Relationship/sort: Motor ya ordenado por `Orden`; sin cambios

## Canonical design contract

```yaml
Design Brief:
  generated_by: powerbi-report-design
  contract_version: 1
  mode: greenfield
  design_identity:
    tone: Brand-Forward Ecopetrol (Corporate Cool remix — cool surface, corporate green/amber accents)
    signature: Top scenario band — dynamic title + Brent slicer + amber "Brent Escenario" chip, repeated on pages 1–3
  archetype: Executive Summary
  color_map:
    - measure: _Medidas[Vol 1P Predicho]
      color: "#009639"
      tint: "#DCF0E2"
    - measure: _Medidas[Vol 1P Baseline]
      color: "#94A3B8"
      tint: "#E2E8F0"
    - measure: _Medidas[Delta vs Base (MBPE)]
      color: "#009639"
      tint: "#DCF0E2"
    - measure: _Medidas[Brent Escenario (USD/bbl)]
      color: "#FFCD00"
      tint: "#FFF3C4"
    - measure: _Medidas[Elasticidad Local (MBPE/USD)]
      color: "#A87900"
      tint: "#FFF3C4"
  theme:
    base: assets/base.json adapted to Ecopetrol brand
    user_overrides: keep CY26SU05 base theme registered; layer Ecopetrol custom theme on top
  accessibility:
    alt_text_strategy: headline+trend per visual
    contrast_notes: amber (#FFCD00) used only as fill with slate (#1E293B) text; price line uses dark gold (#A87900) for AA on white
  interaction_pattern:
    drill_targets: [Diagnóstico Campo]
    cross_filter_rules:
      - source: Brent slicer
        target: portfolio curve (pg1) / field curve (pg3)
        rule: None   # slicer must NOT filter the price-axis curve
      - source: all other
        rule: Filter

  pages:
    - name: "Reservas 1P del portafolio por escenario de Brent"
      role: landing
      archetype: Executive Summary
      layout_variant: B
      variant_rationale: "5 KPIs de importancia comparable, sin un único héroe; la curva de portafolio es evidencia, no KPI."
      page_background: "#F7F9F7"
      layout_contract:
        canvas: { width: 1920, height: 1080, margin: 32, gutter: 24, snap: 8 }
        grid:
          columns: 12
          rows: 12
          regions:
            header:  [1, 1, 9, 2]
            filters: [9, 1, 13, 2]
            kpis:    [1, 2, 13, 4]
            hero:    [1, 4, 8, 10]
            movers:  [8, 4, 13, 10]
            context: [1, 10, 13, 13]
        placements:
          - id: page_title
            region: header
            kind: textbox
            text: "Reservas 1P del portafolio por escenario de Brent"
            purpose: "Título dinámico (medida Título Escenario) — enuncia el escenario antes de las cifras."
          - id: brent_slicer
            region: filters
            kind: slicer
            field_bindings: 'Precio Brent'[Brent (USD/bbl)]
            slicer_type: dropdown
            purpose: "Escenario de precio Brent (sincronizado pág. 1–3)."
          - id: kpi_vol_predicho
            region: kpis
            kind: cardVisual
            purpose: "¿Cuánto 1P al Brent del escenario?"
            field_bindings: _Medidas[Vol 1P Predicho]
            color_strategy: measure_match
            slot: 1
            of: 5
          - id: kpi_delta_mbpe
            region: kpis
            kind: cardVisual
            purpose: "¿Cuánto se mueve vs certificado (MBPE)?"
            field_bindings: _Medidas[Delta vs Base (MBPE)]
            color_strategy: semantic
            insight_basis: "Delta = predicho − baseline (derivado)."
            slot: 2
            of: 5
          - id: kpi_delta_pct
            region: kpis
            kind: cardVisual
            purpose: "¿Cuánto se mueve vs certificado (%)?"
            field_bindings: _Medidas[Delta vs Base %]
            color_strategy: semantic
            insight_basis: "Variación relativa vs baseline."
            slot: 3
            of: 5
          - id: kpi_campos_viables
            region: kpis
            kind: cardVisual
            purpose: "¿Cuántos campos siguen viables a este Brent?"
            field_bindings: _Medidas[# Campos Viables]
            color_strategy: measure_match
            insight_basis: "Conteo de campos con Precio Neto ≥ BK Operacional."
            slot: 4
            of: 5
          - id: kpi_brent_escenario
            region: kpis
            kind: cardVisual
            purpose: "¿Cuál es el Brent del escenario?"
            field_bindings: _Medidas[Brent Escenario (USD/bbl)]
            color_strategy: measure_match
            slot: 5
            of: 5
          - id: portfolio_curve
            region: hero
            kind: lineChart
            purpose: "¿Cómo escala el 1P del portafolio con el Brent?"
            field_bindings: { Category: 'Precio Brent'[Brent (USD/bbl)], Y: _Medidas[Vol 1P Predicho] }
            color_strategy: measure_match
            sort_policy: category_asc
            insight_basis: "Curva completa; línea de ref. en Brent 68.01 (oro) y marcador del escenario."
          - id: top_movers
            region: movers
            kind: barChart
            purpose: "¿Qué campos mueven más el 1P vs certificado a este Brent?"
            field_bindings: { Category: Campo[Campo], Y: _Medidas[Delta vs Base (MBPE)] }
            sort_policy: value_desc
            color_strategy: semantic
          - id: ctx_delta5
            region: context
            kind: cardVisual
            purpose: "¿Cuánto 1P aporta un escalón de +5 USD?"
            field_bindings: _Medidas[Δ Vol por +5 USD (MBPE)]
            color_strategy: measure_match
            insight_basis: "Sensibilidad al escalón de precio (derivado)."
            slot: 1
            of: 3
          - id: ctx_elasticidad
            region: context
            kind: cardVisual
            purpose: "¿Cuál es la pendiente local del portafolio?"
            field_bindings: _Medidas[Elasticidad Local (MBPE/USD)]
            color_strategy: measure_match
            insight_basis: "Diferencia central ±5 USD / 10."
            slot: 2
            of: 3
          - id: ctx_pct_viable
            region: context
            kind: cardVisual
            purpose: "¿Qué fracción del volumen es viable a este Brent?"
            field_bindings: _Medidas[% Volumen Viable]
            color_strategy: measure_match
            insight_basis: "Volumen viable / volumen total (derivado)."
            slot: 3
            of: 3
        space_audit:
          content_cell_count: 132
          placed_cell_count: 132
          empty_cell_pct: 0
          unplaced_regions: []
          largest_region: { name: hero, pct_of_content: 32 }
          balance_rationale: "Franja KPI de 2 filas, curva de portafolio y ranking de movers balanceados, y una franja de contexto derivado; sin banda muerta."

    - name: "Qué campos mueven el portafolio y cuáles están cerca de breakeven"
      role: detail
      archetype: Comparative
      layout_variant: A
      variant_rationale: "Ranking de sensibilidad por campo + matriz de riesgo (colchón vs volumen); comparación entre entidades."
      page_background: "#F7F9F7"
      layout_contract:
        canvas: { width: 1920, height: 1080, margin: 32, gutter: 24, snap: 8 }
        grid:
          columns: 12
          rows: 12
          regions:
            header:  [1, 1, 8, 2]
            filters: [8, 1, 13, 2]
            ranking: [1, 2, 7, 8]
            scatter: [7, 2, 13, 8]
            detail:  [1, 8, 13, 13]
        placements:
          - id: page_title
            region: header
            kind: textbox
            text: "Sensibilidad del portafolio al precio y riesgo de breakeven"
            purpose: "Enuncia las dos preguntas de la página."
          - id: brent_slicer
            region: filters
            kind: slicer
            field_bindings: 'Precio Brent'[Brent (USD/bbl)]
            slicer_type: dropdown
            slot: 1
            of: 2
          - id: gerencia_slicer
            region: filters
            kind: slicer
            field_bindings: Campo[Gerencia Desarrollo]
            slicer_type: dropdown
            slot: 2
            of: 2
          - id: ranking_elasticidad
            region: ranking
            kind: barChart
            purpose: "¿Qué campos son más sensibles al precio?"
            field_bindings: { Category: Campo[Campo], Y: _Medidas[Elasticidad Local (MBPE/USD)] }
            sort_policy: value_desc
            color_strategy: gradient
          - id: scatter_riesgo
            region: scatter
            kind: scatterChart
            purpose: "¿Qué volumen está cerca de su breakeven (colchón bajo)?"
            field_bindings: { X: _Medidas[Colchón vs BK Operacional (USD)], Y: _Medidas[Vol 1P Predicho], Size: _Medidas[Vol 1P Baseline], Details: Campo[Campo], Legend: 'Predicción'[Nivel Confianza] }
            color_strategy: semantic
            comparison_basis: "Breakeven operacional; línea de referencia en colchón = 0."
            insight_basis: "Cuadrante colchón<0 = volumen en riesgo."
          - id: detail_table
            region: detail
            kind: tableEx
            purpose: "¿Cuál es el detalle campo por campo?"
            field_bindings: [Campo[Campo], _Medidas[Vol 1P Baseline], _Medidas[Vol 1P Predicho], _Medidas[Delta vs Base (MBPE)], _Medidas[Elasticidad Local (MBPE/USD)], _Medidas[Colchón vs BK Operacional (USD)], _Medidas[Colchón vs BK Financiero (USD)]]
            sort_policy: value_desc
        space_audit:
          content_cell_count: 132
          placed_cell_count: 132
          empty_cell_pct: 0
          unplaced_regions: []
          largest_region: { name: detail, pct_of_content: 45 }
          balance_rationale: "Ranking y scatter comparten la fila superior en mitades; la tabla de detalle ocupa la base sin exceder 45% de celdas de contenido."

    - name: "Cómo responde este campo al precio"
      role: detail
      archetype: Analytical
      layout_variant: A
      variant_rationale: "Filter-rail: 3 slicers (Campo búsqueda 128 valores, Motor, Brent) + botón drill; densidad analítica por campo."
      page_background: "#F7F9F7"
      layout_contract:
        canvas: { width: 1920, height: 1080, margin: 32, gutter: 24, snap: 8 }
        grid:
          columns: 12
          rows: 12
          regions:
            header:    [1, 1, 13, 2]
            rail:      [1, 2, 3, 13]
            hero:      [3, 2, 10, 9]
            side_cards:[10, 2, 13, 9]
            detail:    [3, 9, 13, 13]
        placements:
          - id: page_title
            region: header
            kind: textbox
            text: "Respuesta al precio — Portafolio completo"
            purpose: "Título dinámico (medida Título Campo) con el campo seleccionado."
          - id: campo_slicer
            region: rail
            kind: slicer
            field_bindings: Campo[Campo]
            slicer_type: dropdown
            purpose: "Selección de campo (búsqueda activada, 128 valores)."
            slot: 1
            of: 4
          - id: motor_slicer
            region: rail
            kind: slicer
            field_bindings: Motor[Motor]
            slicer_type: list
            purpose: "Isotónica (primario) vs Suave (validación)."
            slot: 2
            of: 4
          - id: brent_slicer
            region: rail
            kind: slicer
            field_bindings: 'Precio Brent'[Brent (USD/bbl)]
            slicer_type: dropdown
            slot: 3
            of: 4
          - id: drill_button
            region: rail
            kind: actionButton
            purpose: "Ir al diagnóstico técnico del campo (M1/M2)."
            slot: 4
            of: 4
          - id: field_curve
            region: hero
            kind: lineChart
            purpose: "¿Cómo escala el 1P de este campo con el Brent, por motor?"
            field_bindings: { Category: 'Precio Brent'[Brent (USD/bbl)], Y: _Medidas[Vol 1P Predicho], Legend: Motor[Motor] }
            color_strategy: unique
            sort_policy: category_asc
            insight_basis: "Líneas de ref: Brent 68.01 (oro), baseline horizontal, BK ope/fin verticales."
          - id: side_card_baseline
            region: side_cards
            kind: cardVisual
            purpose: "1P certificado del campo."
            field_bindings: _Medidas[Vol 1P Baseline]
            color_strategy: measure_match
            slot: 1
            of: 6
          - id: side_card_vol
            region: side_cards
            kind: cardVisual
            purpose: "1P al Brent del escenario."
            field_bindings: _Medidas[Vol 1P Predicho]
            color_strategy: measure_match
            slot: 2
            of: 6
          - id: side_card_colchon_ope
            region: side_cards
            kind: cardVisual
            purpose: "Colchón vs breakeven operacional."
            field_bindings: _Medidas[Colchón vs BK Operacional (USD)]
            color_strategy: semantic
            insight_basis: "Precio Neto − BK operacional (derivado)."
            slot: 3
            of: 6
          - id: side_card_confianza
            region: side_cards
            kind: cardVisual
            purpose: "Nivel de confianza del campo."
            field_bindings: 'Predicción'[Nivel Confianza]
            color_strategy: semantic
            insight_basis: "ALTA/MEDIA/BAJA/SOLO_SINTETICO."
            slot: 4
            of: 6
          - id: side_card_r2
            region: side_cards
            kind: cardVisual
            purpose: "Bondad de ajuste M2 (R² LOO)."
            field_bindings: 'Predicción'[M2 R² LOO]
            color_strategy: none
            slot: 5
            of: 6
          - id: side_card_divergencia
            region: side_cards
            kind: cardVisual
            purpose: "Divergencia Isotónica vs Suave."
            field_bindings: 'Predicción'[Divergencia Motores (%)]
            color_strategy: semantic
            insight_basis: "Sanity check: divergencia < 30% en banda observada."
            slot: 6
            of: 6
          - id: detail_por_brent
            region: detail
            kind: tableEx
            purpose: "¿Cuál es el detalle por punto de Brent para este campo?"
            field_bindings: ['Precio Brent'[Brent (USD/bbl)], 'Predicción'[Precio Neto Efectivo (USD/bbl)], _Medidas[Vol 1P Predicho], 'Predicción'[Es Viable], 'Predicción'[Es Extrapolado]]
            sort_policy: category_asc
        space_audit:
          content_cell_count: 110
          placed_cell_count: 110
          empty_cell_pct: 0
          unplaced_regions: []
          largest_region: { name: hero, pct_of_content: 45 }
          balance_rationale: "Rail de filtros justificado (3 slicers + botón); héroe de curva domina sin superar 45% de contenido; cards de contexto y tabla por-Brent completan la base."

    - name: "Dónde no confiar en el modelo"
      role: detail
      archetype: Operational
      layout_variant: A
      variant_rationale: "4-up de calidad: KPIs de confianza + distribución + cola de excepciones + tabla frágil."
      page_background: "#F7F9F7"
      layout_contract:
        canvas: { width: 1920, height: 1080, margin: 32, gutter: 24, snap: 8 }
        grid:
          columns: 12
          rows: 12
          regions:
            header:     [1, 1, 13, 2]
            kpis:       [1, 2, 13, 4]
            dist:       [1, 4, 7, 9]
            exceptions: [7, 4, 13, 9]
            detail:     [1, 9, 13, 13]
        placements:
          - id: page_title
            region: header
            kind: textbox
            text: "Confiabilidad del modelo — dónde no confiar"
            purpose: "Enuncia el propósito de calidad."
          - id: kpi_fallback
            region: kpis
            kind: cardVisual
            purpose: "¿Qué fracción de campos usa beta de portafolio?"
            field_bindings: _Medidas[% Campos Fallback M2]
            color_strategy: semantic
            slot: 1
            of: 5
          - id: kpi_alta
            region: kpis
            kind: cardVisual
            purpose: "Campos de confianza ALTA."
            field_bindings: _Medidas[# Campos Confianza ALTA]
            color_strategy: measure_match
            slot: 2
            of: 5
          - id: kpi_media
            region: kpis
            kind: cardVisual
            purpose: "Campos de confianza MEDIA."
            field_bindings: _Medidas[# Campos Confianza MEDIA]
            color_strategy: none
            slot: 3
            of: 5
          - id: kpi_baja
            region: kpis
            kind: cardVisual
            purpose: "Campos de confianza BAJA."
            field_bindings: _Medidas[# Campos Confianza BAJA]
            color_strategy: semantic
            slot: 4
            of: 5
          - id: kpi_extrapolado
            region: kpis
            kind: cardVisual
            purpose: "¿Qué fracción del volumen está extrapolada?"
            field_bindings: _Medidas[% Vol Extrapolado]
            color_strategy: semantic
            slot: 5
            of: 5
          - id: dist_confianza
            region: dist
            kind: barChart
            purpose: "¿Cómo se distribuye el volumen por nivel de confianza?"
            field_bindings: { Category: 'Predicción'[Nivel Confianza], Y: _Medidas[Vol 1P Predicho] }
            sort_policy: value_desc
            color_strategy: semantic
          - id: exceptions_bar
            region: exceptions
            kind: barChart
            purpose: "¿Qué campos disparan alertas (frágil/outlier/no-identificada)?"
            field_bindings: { Category: Campo[Campo], Y: _Medidas[# Campos] }
            sort_policy: value_desc
            color_strategy: semantic
            insight_basis: "Filtrado a campos con M2 Frágil / Alerta LOO Outlier / Sensibilidad No Identificada."
          - id: fragil_table
            region: detail
            kind: tableEx
            purpose: "¿Cuál es el detalle de los campos frágiles?"
            field_bindings: [Campo[Campo], 'Predicción'[Nivel Confianza], 'Predicción'[Motivo Confianza], 'Predicción'[MAE Rel LOO], 'Predicción'[Divergencia Motores (%)], 'Predicción'[M2 N Puntos]]
            sort_policy: none
        space_audit:
          content_cell_count: 132
          placed_cell_count: 132
          empty_cell_pct: 0
          unplaced_regions: []
          largest_region: { name: detail, pct_of_content: 36 }
          balance_rationale: "Franja KPI, dos paneles de análisis balanceados (distribución + excepciones) y tabla de frágiles; sin banda muerta."

    - name: "Diagnóstico técnico del campo (M1 · M2)"
      role: drillthrough
      archetype: Analytical
      layout_variant: B
      variant_rationale: "Drill-through por campo: dos curvas de auditoría (M1 vs precio aceite, M2 recta Brent) + tira de coeficientes."
      page_background: "#F7F9F7"
      layout_contract:
        canvas: { width: 1920, height: 1080, margin: 32, gutter: 24, snap: 8 }
        grid:
          columns: 12
          rows: 12
          regions:
            header:  [1, 1, 11, 2]
            filters: [11, 1, 13, 2]
            m1:      [1, 2, 7, 8]
            m2:      [7, 2, 13, 8]
            coef:    [1, 8, 13, 13]
        placements:
          - id: page_title
            region: header
            kind: textbox
            text: "Diagnóstico técnico del campo (M1 · M2)"
            purpose: "Título dinámico (Título Campo) del campo drilled."
          - id: back_button
            region: filters
            kind: actionButton
            purpose: "Volver a la página de origen."
          - id: m1_curve
            region: m1
            kind: lineChart
            purpose: "M1: Volumen 1P vs Precio Aceite, por motor."
            field_bindings: { Category: 'Diagnóstico M1'[Precio Aceite (USD/bbl)], Y: 'Diagnóstico M1'[Volumen 1P Predicho (MBPE)], Legend: 'Diagnóstico M1'[Motor Clave] }
            color_strategy: unique
            sort_policy: category_asc
          - id: m2_fit
            region: m2
            kind: lineChart
            purpose: "M2: Precio Aceite = α + β·Brent (recta Theil-Sen)."
            field_bindings: { Category: 'Diagnóstico M2'[Brent Clave (USD/bbl)], Y: 'Diagnóstico M2'[Precio Aceite (USD/bbl)] }
            color_strategy: measure_match
            sort_policy: category_asc
          - id: coef_alpha
            region: coef
            kind: cardVisual
            purpose: "Intercepto α."
            field_bindings: 'Diagnóstico M2'[Alpha]
            color_strategy: none
            slot: 1
            of: 6
          - id: coef_beta
            region: coef
            kind: cardVisual
            purpose: "Sensibilidad β (aceite/Brent)."
            field_bindings: 'Diagnóstico M2'[Beta]
            color_strategy: none
            slot: 2
            of: 6
          - id: coef_r2
            region: coef
            kind: cardVisual
            purpose: "R²."
            field_bindings: 'Diagnóstico M2'[R²]
            color_strategy: none
            slot: 3
            of: 6
          - id: coef_r2loo
            region: coef
            kind: cardVisual
            purpose: "R² LOO (validación)."
            field_bindings: 'Diagnóstico M2'[R² LOO]
            color_strategy: none
            slot: 4
            of: 6
          - id: coef_maeloo
            region: coef
            kind: cardVisual
            purpose: "MAE LOO."
            field_bindings: 'Diagnóstico M2'[MAE LOO]
            color_strategy: none
            slot: 5
            of: 6
          - id: coef_npuntos
            region: coef
            kind: cardVisual
            purpose: "N puntos usados en el ajuste."
            field_bindings: 'Diagnóstico M2'[N Puntos]
            color_strategy: none
            slot: 6
            of: 6
        space_audit:
          content_cell_count: 132
          placed_cell_count: 132
          empty_cell_pct: 0
          unplaced_regions: []
          largest_region: { name: m1, pct_of_content: 27 }
          balance_rationale: "Dos curvas de auditoría en mitades superiores + tira de 6 coeficientes en la base; página de drill densa por diseño."
```

## Implementation notes
- Model changes: `Título Escenario` + `Título Campo` creadas (MCP). Refrescar modelo (Unprocessed) antes de screenshots.
- PBIR/report authoring: resize página existente a 1920×1080; renombrar a "Resumen Ejecutivo"; crear 4 páginas más. Slicer Brent en syncGroup "brent" (páginas 1–3). Editar interacción slicer Brent → curvas de precio = None. Drill-through por `Campo[Campo]`. Botón back.
- Validation: `powerbi-report-author validate` tras cada lote.
- Desktop verification: reload + screenshot-all; revisar monotonía, marcador de escenario, C5 (CASTILLA @68≈baseline), % fallback visible.
- Publishing boundary: ninguno (PBIP local).
- Risks: modelo Unprocessed → refrescar; 2 campos sin DIM; amarillo contraste (mitigado: fill + texto slate, línea oro oscuro).
