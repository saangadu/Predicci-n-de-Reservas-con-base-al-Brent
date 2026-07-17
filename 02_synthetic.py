"""
02_synthetic.py — Inyeccion de puntos sinteticos de ancla fisica (escalera multi-clase, Path D)

Genera puntos sinteticos con PRECIO_NETO < salida mas alta del libro para anclar el
modelo a la escalera de viabilidad economica consolidada a 1P (Path D, directrices
2026-06-11, docs/DISENO_1P_CONSOLIDADO.md):

  1P(p) = PDP·1[p >= abandono] + Σ_clase V_clase·1[p >= salida_clase]   (clase ∈ PNP, PND)

  TRAMO BAJO: [BK_ANCLA_PDP - RANGO_USD, BK_ANCLA_PDP)
    → vol = 0; delta = -BASELINE_1P (abandono total: bajo el VPN=0 de PDP nada vive)

  ESCALERA MULTI-CLASE: [BK_ANCLA_PDP, salida mas alta)
    → cada clase PNP/PND sale del libro debajo de su LIMITE ECONOMICO propio
      (BK_SALIDA_PNP/PND, post-swap FINANCIERO por clase). El orden de salida lo da
      el PRECIO, no la clase (RUBIALES 2025: PNP sale antes que PND).
    → degradacion (§4.5): clase sin limite economico propio sale en BK_ANCLA_FIN
      (ponderado PNP+PND, diseño anterior); sin ningun limite PNP/PND la escalera
      colapsa al diseño de 2 tramos; salida <= abandono → la clase se fusiona con
      el abandono (sin escalon propio).

  CAP DE MONOTONIA (obligatorio, §4.3): los volumenes certificados por clase ignoran
  el truncamiento por precio → un escalon podria quedar POR ENCIMA de un punto real
  a mayor precio. En espacio delta:
      delta_escalon = min(nivel - baseline, min(delta reales del campo))
  Filas capadas → ALERTA='ESCALON_CAPADO' (caso CASTILLA NORTE: 117.6 → 111.0 MBPE).

  Sobre la salida mas alta NO se inyecta ancla: la banda de datos reales gobierna
  (ningun ancla delta=0 — el breakeven financiero es umbral de contabilizacion del
  libro, no de invarianza de volumen; ver DISENO §3).

  BRENT_INSENSITIVE: campos donde el ingreso de gas/GLP fijo domina sobre el aceite.
    No se inyectan sinteticos para no crear un ancla falsa.

Supuestos:
  - RANGO_USD=5, PASO_USD=1 (validados con equipo financiero, 2026-06-09).
  - BRENT implicito (D5, 2026-06-11): reconstruido desde Precio Neto usando los
    descuentos CERTIFICADOS de la vigencia del breakeven (HIST Precio Net,
    CAMPO×AÑO=vigencia). Fallback: medianas historicas del campo. El FC del
    breakeven usa la economia de esa vigencia → la conversion debe usar la misma.

Ver docs/MAESTRO.md §8 y docs/DISENO_1P_CONSOLIDADO.md para la justificacion.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from track import sufijo_track, flag

_SUF = sufijo_track()   # '' Produccion; '_calidad' si PRED_TRACK=calidad
BASE_DIR = Path(__file__).parent
STAGING  = BASE_DIR / "datos" / f"staging{_SUF}"

RANGO_USD = 5    # puntos en Tramo BAJO: [bk_inf - RANGO, bk_inf)
PASO_USD  = 1
# Guard de banda real (2026-06-11): ningun sintetico se inyecta en/ sobre la banda
# de datos certificados del campo (margen de seguridad). Donde hay datos reales,
# los datos gobiernan — un ancla "el libro salio" dentro de la banda contradice
# los puntos certificados (el libro esta vivo ahi) y rompe la monotonia del
# entrenamiento. Caso tipico: anclas/salidas de breakevens extremos (GALEMBO
# VPN=0 a $285) o salidas de clase sobre la banda (BORANDA $142).
MARGEN_BANDA_USD = 1.0

# Suavizado de transición (directriz DIRECTRIZ_ESCALERA_DECK.md §3): junto a la banda de
# datos reales, el escalón sintético no puede sostener el precipicio completo — donde el
# deck real (misma economía que la predicción) muestra el libro vivo, un escalón de
# abandono profundo a <$5 de esos datos arrastra el ancla p_ref al fondo del valle y la
# curva "recupera" volumen al subir el Brent (artefacto Caño Limón +34%). En la franja
# TRANSICION_USD antes de la banda, el escalón se pisa a lo sumo CLIFF_FRAC·baseline por
# debajo del peor delta real observado. Cap SIMÉTRICO al de monotonía (§4.3): ambos
# impiden que un ancla sintética contradiga la evidencia real más cercana.
TRANSICION_USD = 6.0
CLIFF_FRAC     = 0.15


def calcular_medianas_precio(df: pd.DataFrame) -> pd.DataFrame:
    """Medianas de Calidad y Transporte por campo (filas reales con precio disponible)."""
    df_real = df[
        (~df["ES_SINTETICO"]) &
        df["BRENT_FLAT_USD_BBL"].notna() &
        df["DESCUENTO_CALIDAD_USD_BBL"].notna() &
        df["DESCUENTO_TRANSPORTE_USD_BBL"].notna()
    ]
    return (df_real
            .groupby("CAMPO")[["BRENT_FLAT_USD_BBL",
                                "DESCUENTO_CALIDAD_USD_BBL",
                                "DESCUENTO_TRANSPORTE_USD_BBL"]]
            .median()
            .reset_index()
            .rename(columns={
                "BRENT_FLAT_USD_BBL":             "MED_BRENT",
                "DESCUENTO_CALIDAD_USD_BBL":      "MED_CALIDAD",
                "DESCUENTO_TRANSPORTE_USD_BBL":   "MED_TRANSPORTE",
            }))


def calcular_baselines_latest(df: pd.DataFrame) -> dict:
    """
    Ultimo VOLUMEN_1P_OFICIAL_MBPE certificado por campo.
    Es el BASELINE total contra el que se calcula el DELTA_SENS en espacio delta.
    """
    df_base = df[(df["ESCENARIO"] == "BASE") & df["VOLUMEN_1P_OFICIAL_MBPE"].notna()]
    return (df_base.sort_values("AÑO")
            .groupby("CAMPO")["VOLUMEN_1P_OFICIAL_MBPE"]
            .last()
            .to_dict())


def calcular_baselines_por_clase(df: pd.DataFrame) -> dict:
    """
    Ultimo VOLUMEN certificado por clase (PDP / PNP / PND) por campo.
    Alturas de los escalones de la escalera multi-clase.
    Retorna dict: {campo: {"pdp": float, "pnp": float, "pnd": float}}
    """
    df_base = df[(df["ESCENARIO"] == "BASE")]
    result = {}
    for campo, sub in df_base.groupby("CAMPO"):
        sub = sub.sort_values("AÑO")
        vols = {}
        for clave, col in [("pdp", "VOLUMEN_PDP_MBPE"),
                           ("pnp", "VOLUMEN_PNP_MBPE"),
                           ("pnd", "VOLUMEN_PND_MBPE")]:
            vals = sub[col].dropna().values
            vols[clave] = float(vals[-1]) if len(vals) > 0 else 0.0
        result[campo] = vols
    return result


def calcular_descuentos_cert(df: pd.DataFrame) -> dict:
    """
    Descuentos CERTIFICADOS por CAMPO×AÑO (filas BASE de HIST Precio Net).
    Usados para convertir Precio Neto ↔ Brent en la vigencia del breakeven (D5):
    el FC del breakeven usa la economia de su vigencia, no la mediana historica.
    Retorna dict: {(campo, "2025"): (cal, tra)}
    """
    df_base = df[(df["ESCENARIO"] == "BASE") &
                 df["DESCUENTO_CALIDAD_USD_BBL"].notna() &
                 df["DESCUENTO_TRANSPORTE_USD_BBL"].notna()]
    return {(r["CAMPO"], str(int(r["AÑO"]))):
            (float(r["DESCUENTO_CALIDAD_USD_BBL"]),
             float(r["DESCUENTO_TRANSPORTE_USD_BBL"]))
            for _, r in df_base.iterrows()}


def calcular_min_delta_real(df: pd.DataFrame) -> dict:
    """
    Peor delta certificado por campo (puntos reales del Consolidado).
    Cota del cap de monotonia (§4.3): ningun escalon sintetico puede quedar por
    encima del peor punto real observado a mayor precio.
    """
    df_real = df[(~df["ES_SINTETICO"]) & df["DELTA_SENS_MBPE"].notna()]
    return df_real.groupby("CAMPO")["DELTA_SENS_MBPE"].min().to_dict()


def calcular_banda_real_lo(df: pd.DataFrame) -> dict:
    """
    Borde inferior (Precio Neto) de la banda de datos certificados por campo.
    Guard de inyeccion: ningun sintetico en/ sobre este precio (los datos reales
    gobiernan su propia banda). Campos sin reales → sin restriccion (inf).
    """
    df_real = df[(~df["ES_SINTETICO"]) & df["DELTA_SENS_MBPE"].notna()]
    return df_real.groupby("CAMPO")["PRECIO_NETO_USD_BBL"].min().to_dict()


def _fila_sintetica(campo, pneto, med_cal, med_tra, vbk_v, bk_fin, bk_op_val,
                   vol_1p, delta, baseline, alerta="",
                   sal_pnp=np.nan, sal_pnd=np.nan) -> dict:
    """Construye una fila sintetica con todos los campos canonicos del tablon."""
    brent_impl = pneto - med_cal - med_tra
    return {
        "CAMPO":                          campo,
        "CAMPO_ORIGEN_RAW":               campo,
        "AÑO":                            9999,
        "VIGENCIA":                       "SINTETICO",
        "ESCENARIO":                      "SINTETICO",
        "ES_BASELINE":                    False,
        "ES_SINTETICO":                   True,
        "NIVEL_DEFINICIONAL":             "",
        "BRENT_FLAT_USD_BBL":             round(brent_impl, 4),
        "DESCUENTO_CALIDAD_USD_BBL":      med_cal,
        "DESCUENTO_TRANSPORTE_USD_BBL":   med_tra,
        "PRECIO_NETO_USD_BBL":            round(pneto, 4),
        "VOLUMEN_PDP_MBPE":               np.nan,
        "VOLUMEN_PNP_MBPE":               np.nan,
        "VOLUMEN_PND_MBPE":               np.nan,
        "VOLUMEN_1P_OFICIAL_MBPE":        np.nan,
        # Los sinteticos se refieren SIEMPRE al ultimo libro certificado (baseline_latest):
        # el floor fisico (vol=0 bajo abandono) es una propiedad del libro VIGENTE, no de
        # una vigencia del deck. No llevan CHECKPOINT (no tienen año A).
        "CHECKPOINT_1P_MBPE":             np.nan,
        "BASELINE_1P_MBPE":               baseline,
        "VOLUMEN_1P_SENSIBILIDAD_MBPE":   vol_1p,
        "DELTA_SENS_MBPE":                delta,
        "BREAKEVEN_USD_BBL":              bk_fin,
        "PRECIO_EQUILIBRIO_USD_BBL":     bk_op_val,
        # Redondear a 4dp para que coincida con PRECIO_NETO_USD_BBL y las comparaciones
        # >= BK_ANCLA_PDP / < BK_ANCLA_FIN no fallen por error de float en los bordes
        # de los tramos BAJO/ESCALERA.
        "BK_ANCLA_FIN_USD_BBL":           round(bk_fin, 4),
        "BK_ANCLA_PDP_USD_BBL":           round(bk_op_val, 4) if pd.notna(bk_op_val) else bk_op_val,
        # Salidas por clase USADAS en la escalera (trazabilidad Path D): la salida
        # propia puede superar BK_ANCLA_FIN — sin estas columnas, la auditoria del
        # techo de inyeccion contra el tablon es imposible.
        "BK_SALIDA_PNP_USD_BBL":          round(sal_pnp, 4) if pd.notna(sal_pnp) else np.nan,
        "BK_SALIDA_PND_USD_BBL":          round(sal_pnd, 4) if pd.notna(sal_pnd) else np.nan,
        "BRENT_INSENSITIVE":              False,
        "VIGENCIA_BREAKEVEN":             vbk_v,
        "PRED_XGBOOST_MBPE":              np.nan,
        "PRED_ISOTONICA_MBPE":            np.nan,
        "DELTA_XGBOOST_VS_OFICIAL":       np.nan,
        "DELTA_ISOTONICA_VS_OFICIAL":     np.nan,
        "ALERTA":                         alerta,
        "HOMOLOG_FLAG":                   "OK",
    }


def generar_sinteticos(df: pd.DataFrame, medianas: pd.DataFrame,
                       baselines: dict, baselines_clase: dict,
                       descuentos_cert: dict, min_delta_real: dict,
                       banda_real_lo: dict = None) -> pd.DataFrame:
    """
    Por cada campo genera la ESCALERA MULTI-CLASE (Path D, ver docstring del modulo):

      Tramo BAJO: [BK_ANCLA_PDP - RANGO, BK_ANCLA_PDP)  → vol = 0 (abandono total)
      Escalera:   [BK_ANCLA_PDP, salida mas alta)        → en cada precio p,
                  vol = PDP + Σ V_clase·1[p >= salida_clase], capado en espacio delta
                  al peor delta real del campo (§4.3, ALERTA=ESCALON_CAPADO).

    Salida de clase: BK_SALIDA_PNP/PND (limite economico propio); degradacion §4.5:
    sin limite propio → BK_ANCLA_FIN; salida <= abandono → se fusiona con el abandono.
    Sobre la salida mas alta no se inyecta ancla (la banda real gobierna).
    Campos BRENT_INSENSITIVE o sin breakeven: se omiten con advertencia explicita.
    """
    banda_real_lo = banda_real_lo or {}
    filas = []
    n_capados = n_en_banda = n_suavizados = n_perfil = 0
    usar_perfil = flag("PRED_PERFIL_SALIDA")
    mbk_perfil_pcts = (10, 25, 50, 75, 90)
    if usar_perfil:
        print("  [CALIDAD] Perfil de salida ON: escalera de clase -> curva gradual "
              "del FC (ALERTA=PERFIL_SALIDA, directriz 4bis.6)")
    for campo in df["CAMPO"].unique():
        sub = df[df["CAMPO"] == campo]
        # Guard de banda: techo absoluto de inyeccion para este campo
        tope_banda = banda_real_lo.get(campo, np.inf) - MARGEN_BANDA_USD

        # Verificar si el campo es Brent-insensible
        insens_vals = sub["BRENT_INSENSITIVE"].dropna().values
        if len(insens_vals) > 0 and bool(insens_vals[0]):
            print(f"  [WARN] {campo}: Brent-insensible, sin ancla sintetica")
            continue

        # Leer anclas del tablon (post-swap: FIN=piso superior, PDP=piso inferior).
        # FIN/PDP/salidas deben venir de la MISMA vigencia (calcular_breakeven_ponderado
        # los calcula en pareja por CAMPO×VIGENCIA). Se usa la vigencia mas reciente.
        anclas_sub = sub.dropna(subset=["BK_ANCLA_FIN_USD_BBL", "BK_ANCLA_PDP_USD_BBL"], how="all")

        # Si no hay ningun ancla, omitir
        if anclas_sub.empty:
            print(f"  [WARN] {campo}: sin breakeven disponible, omitiendo sinteticos")
            continue

        fila_ancla = anclas_sub.sort_values("VIGENCIA_BREAKEVEN", ascending=False).iloc[0]
        bk_sup = float(fila_ancla["BK_ANCLA_FIN_USD_BBL"]) if pd.notna(fila_ancla["BK_ANCLA_FIN_USD_BBL"]) else np.nan
        bk_inf = float(fila_ancla["BK_ANCLA_PDP_USD_BBL"]) if pd.notna(fila_ancla["BK_ANCLA_PDP_USD_BBL"]) else np.nan

        # Si solo uno esta disponible, usar el mismo para ambos (tramo unico)
        if np.isnan(bk_sup):
            bk_sup = bk_inf
        if np.isnan(bk_inf):
            bk_inf = bk_sup

        med_row = medianas[medianas["CAMPO"] == campo]
        if med_row.empty:
            print(f"  [WARN] {campo}: sin medianas de precio, omitiendo sinteticos")
            continue

        # Vigencia de la MISMA fila de anclas: los sinteticos quedan etiquetados
        # con la vigencia cuyos pisos usan.
        vbk_v = str(fila_ancla["VIGENCIA_BREAKEVEN"]) \
            if pd.notna(fila_ancla.get("VIGENCIA_BREAKEVEN")) else "2024"

        # D5: descuentos certificados de la vigencia del breakeven; fallback medianas
        if (campo, vbk_v) in descuentos_cert:
            med_cal, med_tra = descuentos_cert[(campo, vbk_v)]
        else:
            med_cal = float(med_row["MED_CALIDAD"].values[0])
            med_tra = float(med_row["MED_TRANSPORTE"].values[0])

        baseline_total = baselines.get(campo, np.nan)
        bl_clase       = baselines_clase.get(campo, {"pdp": 0.0, "pnp": 0.0, "pnd": 0.0})

        # ── Salidas por clase (Path D + degradacion §4.5) ─────────────────────
        # Sin limite economico propio → la clase sale en BK_ANCLA_FIN (ponderado).
        # Salida <= abandono → la clase se fusiona con el abandono (vive en toda
        # la escalera, sin escalon propio).
        salidas = {}
        for clave, col in [("pnp", "BK_SALIDA_PNP_USD_BBL"),
                           ("pnd", "BK_SALIDA_PND_USD_BBL")]:
            vol = float(bl_clase.get(clave, 0.0) or 0.0)
            if vol <= 0.0:
                continue   # clase sin volumen certificado: no aporta escalon
            s = fila_ancla.get(col, np.nan) if col in fila_ancla.index else np.nan
            # La salida propia PUEDE superar BK_ANCLA_FIN (CASTILLA: PND 43.4 vs
            # ancla 37.6) — el ponderado promedia, la salida es de la clase.
            salidas[clave] = float(s) if pd.notna(s) else bk_sup
        sal_pnp = salidas.get("pnp", np.nan)
        sal_pnd = salidas.get("pnd", np.nan)

        # ── Tramo BAJO: [bk_inf - RANGO, bk_inf) ──────────────────────────────
        # Bajo el piso de abandono (VPN=0 de PDP): ninguna reserva 1P es economica.
        delta_total = (-float(baseline_total)) if pd.notna(baseline_total) else np.nan
        for pneto in np.arange(bk_inf - RANGO_USD, bk_inf, PASO_USD):
            if pneto >= tope_banda:
                n_en_banda += 1
                continue   # los datos certificados gobiernan su banda
            filas.append(_fila_sintetica(campo, pneto, med_cal, med_tra, vbk_v,
                                         bk_sup, bk_inf, 0.0, delta_total,
                                         baseline_total,
                                         sal_pnp=sal_pnp, sal_pnd=sal_pnd))

        if pd.isna(baseline_total):
            continue   # sin baseline no hay espacio delta para la escalera

        # ── P4: perfil de salida volumetrico (track Calidad, directriz §4bis.6) ──
        # Reemplaza el acantilado de clase por la curva gradual "% de volumen que
        # deja de ser economico" vs precio, tomada del MISMO FC (columnas BK_P*).
        # Entre Precio Equilibrio (bk_inf, vol=0) y BK (bk_sup, reservas completas)
        # el volumen recupera siguiendo el perfil. Mata el artefacto de recuperacion
        # (Caño Limon +34%) porque la curva es gradual como el deck real.
        perfil = {}
        if usar_perfil:
            for pct in mbk_perfil_pcts:
                col = f"BK_P{pct}"
                if col in fila_ancla.index and pd.notna(fila_ancla[col]):
                    perfil[pct] = float(fila_ancla[col])
        if perfil:
            techo = min(bk_sup, tope_banda)
            if techo - bk_inf <= PASO_USD:
                continue
            # Puntos (precio, fraccion de volumen ya salida); endpoints anclados
            uniq = {round(bk_sup, 4): 0.0, round(bk_inf, 4): 1.0}
            for pct, pv in perfil.items():
                if bk_inf < pv < bk_sup:
                    uniq[round(pv, 4)] = max(uniq.get(round(pv, 4), 0.0), pct / 100.0)
            xp = np.array(sorted(uniq))
            fp = np.array([uniq[k] for k in sorted(uniq)])
            cap_real = min_delta_real.get(campo, np.nan)
            for pneto in np.arange(bk_inf, techo, PASO_USD):
                ex    = float(np.clip(np.interp(pneto, xp, fp), 0.0, 1.0))
                delta = -ex * float(baseline_total)
                nivel = float(baseline_total) + delta
                alerta = "PERFIL_SALIDA"
                if pd.notna(cap_real) and delta > float(cap_real):
                    delta  = float(cap_real)
                    nivel  = float(baseline_total) + delta
                    alerta = "PERFIL_SALIDA|ESCALON_CAPADO"
                    n_capados += 1
                # Hard-zero (invariante M1: Vol=max(baseline+delta,0)): el cap_real
                # (peor delta historico real del campo) puede superar -baseline_total
                # cuando el baseline de referencia (cierre A-1) es menor al volumen
                # certificado en la vigencia real mas mala — nunca reservas negativas.
                if nivel < 0.0:
                    nivel = 0.0
                    delta = -float(baseline_total)
                filas.append(_fila_sintetica(campo, pneto, med_cal, med_tra, vbk_v,
                                             bk_sup, bk_inf, nivel, delta,
                                             baseline_total, alerta,
                                             sal_pnp=sal_pnp, sal_pnd=sal_pnd))
            n_perfil += 1
            continue   # perfil reemplaza la escalera de clases para este campo

        techo = max([bk_inf] + list(salidas.values()))
        # Guard de banda: la escalera no invade la banda de datos certificados
        techo = min(techo, tope_banda)
        if techo - bk_inf <= PASO_USD:
            continue   # escalera degenerada: solo tramo BAJO (diseño de un piso)

        # ── Escalera: [bk_inf, techo) — nivel = clases vivas a cada precio ────
        vol_pdp  = float(bl_clase.get("pdp", 0.0) or 0.0)
        cap_real = min_delta_real.get(campo, np.nan)   # peor delta certificado
        # Piso de transición (directriz §3): a lo sumo CLIFF_FRAC·baseline por debajo del
        # peor delta real; solo aplica en la franja TRANSICION_USD antes de la banda.
        piso_trans = (float(cap_real) - CLIFF_FRAC * float(baseline_total)
                      if pd.notna(cap_real) else np.nan)
        inicio_trans = techo - TRANSICION_USD
        for pneto in np.arange(bk_inf, techo, PASO_USD):
            nivel = vol_pdp + sum(float(bl_clase[c]) for c, s in salidas.items()
                                  if pneto >= s)
            delta = nivel - float(baseline_total)
            alerta = ""
            # Cap de monotonia (§4.3): el escalon nunca por encima del peor real
            if pd.notna(cap_real) and delta > float(cap_real):
                delta  = float(cap_real)
                nivel  = float(baseline_total) + delta
                alerta = "ESCALON_CAPADO"
                n_capados += 1
            # Suavizado de transición (§3): junto a la banda, el escalón no cae por debajo
            # del piso de transición (evita el precipicio pegado a datos reales vivos).
            elif pd.notna(piso_trans) and pneto >= inicio_trans and delta < piso_trans:
                delta  = piso_trans
                nivel  = float(baseline_total) + delta
                alerta = "ESCALON_SUAVIZADO"
                n_suavizados += 1
            filas.append(_fila_sintetica(campo, pneto, med_cal, med_tra, vbk_v,
                                         bk_sup, bk_inf, nivel, delta,
                                         baseline_total, alerta,
                                         sal_pnp=sal_pnp, sal_pnd=sal_pnd))

    if n_perfil > 0:
        print(f"  [INFO] {n_perfil} campos con perfil de salida volumetrico "
              f"(ALERTA=PERFIL_SALIDA, directriz §4bis.6)")
    if n_suavizados > 0:
        print(f"  [INFO] {n_suavizados} puntos sinteticos suavizados junto a la banda "
              f"real (ALERTA=ESCALON_SUAVIZADO, directriz §3)")
    if n_capados > 0:
        print(f"  [INFO] {n_capados} puntos sinteticos capados por monotonia "
              f"(ALERTA=ESCALON_CAPADO)")
    if n_en_banda > 0:
        print(f"  [INFO] {n_en_banda} puntos BAJO omitidos por invadir la banda "
              f"real (anclas sobre la banda — revisar breakevens extremos)")
    return pd.DataFrame(filas)


def validar_sinteticos(df_sint: pd.DataFrame) -> None:
    sep = "-" * 70
    print(f"\n{sep}\n  Resumen sinteticos generados (escalera multi-clase)\n{sep}")
    print(f"  Total filas sinteticas : {len(df_sint)}")
    for campo in sorted(df_sint["CAMPO"].unique()):
        sub     = df_sint[df_sint["CAMPO"] == campo]
        bajo    = sub[sub["VOLUMEN_1P_SENSIBILIDAD_MBPE"] == 0.0]
        esc     = sub[sub["VOLUMEN_1P_SENSIBILIDAD_MBPE"] != 0.0]
        # Numero de niveles distintos de la escalera (sin contar vol=0)
        niveles = esc["VOLUMEN_1P_SENSIBILIDAD_MBPE"].round(2).nunique()
        capados = int((sub["ALERTA"] == "ESCALON_CAPADO").sum())
        bk_f    = sub["BK_ANCLA_FIN_USD_BBL"].values[0]
        bk_p    = sub["BK_ANCLA_PDP_USD_BBL"].values[0]
        print(f"  {campo:<20} | BAJO={len(bajo):3d}  ESC={len(esc):3d} pts "
              f"({niveles} niveles{', cap=' + str(capados) if capados else ''}) "
              f"| BK_FIN={bk_f:.2f}  BK_PDP={bk_p:.2f}")
    print(sep + "\n")


if __name__ == "__main__":
    print("=== 02_synthetic.py — Inyeccion de sinteticos (escalera financiero/operacional) ===\n")

    ruta = STAGING / "tablon_unico.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"Ejecutar 01_etl.py primero. No encontrado: {ruta}")

    df = pd.read_parquet(ruta)

    # Eliminar sinteticos previos (idempotente)
    n_prev = int(df["ES_SINTETICO"].sum())
    if n_prev > 0:
        print(f"  [INFO] Eliminando {n_prev} sinteticos previos")
        df = df[~df["ES_SINTETICO"]].copy()

    print("[1/4] Calculando medianas de precio por campo...")
    medianas = calcular_medianas_precio(df)
    for _, r in medianas.iterrows():
        print(f"  {r['CAMPO']:<20} | med_brent={r['MED_BRENT']:.2f} | "
              f"cal={r['MED_CALIDAD']:.2f} | tra={r['MED_TRANSPORTE']:.2f}")

    print("\n[2/4] Calculando baselines totales (ultimo OFICIAL por campo)...")
    baselines = calcular_baselines_latest(df)
    for campo, bl in sorted(baselines.items()):
        print(f"  {campo:<20} | baseline_total={bl:.2f} MBPE")

    print("\n[3/4] Calculando baselines por clase (PDP / PNP / PND)...")
    baselines_clase = calcular_baselines_por_clase(df)
    for campo, bl in sorted(baselines_clase.items()):
        print(f"  {campo:<20} | PDP={bl['pdp']:.2f}  PNP={bl['pnp']:.2f}  "
              f"PND={bl['pnd']:.2f} MBPE")

    descuentos_cert = calcular_descuentos_cert(df)
    min_delta_real  = calcular_min_delta_real(df)
    banda_real_lo   = calcular_banda_real_lo(df)

    print(f"\n[4/4] Generando sinteticos (BAJO: rango BK_PDP-{RANGO_USD}; "
          f"escalera multi-clase hasta la salida mas alta, sin invadir banda real)...")
    df_sint = generar_sinteticos(df, medianas, baselines, baselines_clase,
                                 descuentos_cert, min_delta_real, banda_real_lo)

    validar_sinteticos(df_sint)

    df_completo = pd.concat([df, df_sint], ignore_index=True)
    df_completo = df_completo.sort_values(
        ["CAMPO", "ESCENARIO", "PRECIO_NETO_USD_BBL"]).reset_index(drop=True)

    df_completo.to_parquet(STAGING / "tablon_unico.parquet", index=False)
    df_completo.to_csv(STAGING / "tablon_unico.csv", index=False, encoding="utf-8-sig")

    total_sint = int(df_completo["ES_SINTETICO"].sum())
    print(f"  Tablon actualizado: {len(df_completo)} filas ({total_sint} sinteticas, "
          f"{len(df)} reales)")
    print("\n=== 02_synthetic.py — Completado ===")
