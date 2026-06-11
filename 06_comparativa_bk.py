"""
06_comparativa_bk.py — Comparativa de criterios de anclaje del breakeven (analisis offline)

NO forma parte de run_pipeline.py ni altera las anclas del modelo. Su unico objetivo
es validar, lado a lado, dos formas de fijar el PISO SUPERIOR del anclaje (BK_ANCLA_FIN,
el precio bajo el cual PNP/PND dejan de ser contabilizables):

  A) PONDERADO (vigente en el pipeline): promedio del breakeven financiero de las clases
     PNP y PND ponderado por su volumen certificado. Es el que usa 01_etl::calcular_breakeven_ponderado.
     Riesgo senalado por el usuario: una clase grande con breakeven bajo "arrastra" el piso
     hacia abajo y enmascara que la reserva mas incierta sale del libro a un precio mucho mayor.

  B) MAYOR INCERTIDUMBRE (alternativa conservadora): breakeven financiero de la clase mas
     incierta PRESENTE en el campo, en orden PND > PNP > PDP. La reserva con mayor riesgo
     de no materializarse fija el limite — vision mas exigente para soporte CAPEX.

Tambien reporta el "consolidado" (ponderado de las 3 clases, cifra de trazabilidad de PBI)
para contexto. Se calcula para CADA vigencia disponible (2024 y 2025) — de paso documenta
el drift inter-vigencia del ancla.

Convencion de etiquetas: identica al pipeline (post-swap _swap_breakeven_labels). Se importan
las funciones de 01_etl para no duplicar la logica del swap ni del ponderado.

Salidas:
  resultados/comparativa_bk_ancla.csv
  resultados/plots_analisis/comparativa_bk_ancla.png
"""

import importlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR   = Path(__file__).parent
STAGING    = BASE_DIR / "datos" / "staging"
RESULTADOS = BASE_DIR / "resultados"
PLOTS_DIR  = RESULTADOS / "plots_analisis"
RESULTADOS.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# 01_etl.py no es un identificador Python valido (empieza por digito): import dinamico
etl = importlib.import_module("01_etl")

# Orden de incertidumbre: la clase MAS incierta primero. PND (perforacion nueva, sin
# pozo) es la mas riesgosa; PNP (reactivacion/workover) intermedia; PDP (produciendo) la
# mas cierta. La clase mas incierta presente con volumen fija el piso alternativo.
ORDEN_INCERTIDUMBRE = [("D-PND", "VOLUMEN_PND_MBPE"),
                       ("D-PNP", "VOLUMEN_PNP_MBPE"),
                       ("D-PDP", "VOLUMEN_PDP_MBPE")]


def volumenes_ultimo_anio(df_res: pd.DataFrame) -> pd.DataFrame:
    """Volumen certificado por clase del ULTIMO año disponible de cada campo (HIST).
    Mismo criterio de seleccion que usa calcular_breakeven_ponderado para ponderar."""
    ultimo = (df_res.groupby("CAMPO")["AÑO"].max().reset_index()
              .rename(columns={"AÑO": "AÑO_MAX"}))
    df_ult = df_res.merge(ultimo, on="CAMPO")
    df_ult = df_ult[df_ult["AÑO"] == df_ult["AÑO_MAX"]]
    cols = ["CAMPO", "VOLUMEN_PDP_MBPE", "VOLUMEN_PNP_MBPE", "VOLUMEN_PND_MBPE"]
    cols = [c for c in cols if c in df_ult.columns]
    return df_ult[cols].drop_duplicates("CAMPO").reset_index(drop=True)


def bk_clase_mas_incierta(bk_campo_vig: pd.DataFrame, vols: dict) -> tuple:
    """
    Breakeven financiero (post-swap, piso superior) de la clase MAS incierta presente
    con volumen > 0, recorriendo PND -> PNP -> PDP.

    Solo se acepta una clase si su breakeven financiero es valido (SOLVER_S1_FALLO=False
    y no NaN) — mismo gate que el ponderado. Retorna (bk_alt, clase_usada).
    """
    for clase, col_vol in ORDEN_INCERTIDUMBRE:
        vol = float(vols.get(col_vol, 0.0) or 0.0)
        if vol <= 0.0:
            continue   # la clase no aporta reserva: no puede fijar el piso
        fila = bk_campo_vig[bk_campo_vig["CLASE_RESERVA"] == clase]
        if fila.empty:
            continue
        if bool(fila["SOLVER_S1_FALLO"].values[0]):
            continue   # breakeven financiero no confiable para esta clase
        bk = fila["BREAKEVEN_FINANCIERO_USD_BBL"].values[0]
        if pd.notna(bk):
            return float(bk), clase
    return np.nan, None


def construir_comparativa(df_bk: pd.DataFrame, df_pond: pd.DataFrame,
                          df_vols: pd.DataFrame) -> pd.DataFrame:
    """Una fila por CAMPO × VIGENCIA con ambos criterios de piso superior y su diferencia."""
    vols_por_campo = df_vols.set_index("CAMPO").to_dict(orient="index")
    filas = []
    for (campo, vig), bk_cv in df_bk.groupby(["CAMPO", "VIGENCIA_BREAKEVEN"]):
        # Criterio A (vigente): BK_ANCLA_FIN ponderado PNP+PND del pipeline
        pond_row = df_pond[(df_pond["CAMPO"] == campo) &
                           (df_pond["VIGENCIA_BREAKEVEN"] == vig)]
        if pond_row.empty:
            continue
        bk_ponderado   = float(pond_row["BK_ANCLA_FIN_USD_BBL"].values[0]) \
            if pd.notna(pond_row["BK_ANCLA_FIN_USD_BBL"].values[0]) else np.nan
        bk_consolidado = float(pond_row["BREAKEVEN_FINANCIERO_USD_BBL"].values[0]) \
            if pd.notna(pond_row["BREAKEVEN_FINANCIERO_USD_BBL"].values[0]) else np.nan
        brent_insens   = bool(pond_row["BRENT_INSENSITIVE"].values[0])

        # Criterio B (alternativo): clase mas incierta presente
        vols = vols_por_campo.get(campo, {})
        bk_alt, clase_alt = bk_clase_mas_incierta(bk_cv, vols)

        delta = bk_alt - bk_ponderado if (pd.notna(bk_alt) and pd.notna(bk_ponderado)) else np.nan
        delta_pct = (delta / bk_ponderado * 100) if (pd.notna(delta) and bk_ponderado not in (0, np.nan)) else np.nan

        filas.append({
            "CAMPO":                      campo,
            "VIGENCIA":                   vig,
            "BK_PONDERADO_PNP_PND":       round(bk_ponderado, 2) if pd.notna(bk_ponderado) else None,
            "BK_MAYOR_INCERTIDUMBRE":     round(bk_alt, 2) if pd.notna(bk_alt) else None,
            "CLASE_INCERTIDUMBRE":        clase_alt,
            "BK_CONSOLIDADO_1P":          round(bk_consolidado, 2) if pd.notna(bk_consolidado) else None,
            "DELTA_ALT_VS_POND":          round(delta, 2) if pd.notna(delta) else None,
            "DELTA_PCT":                  round(delta_pct, 1) if pd.notna(delta_pct) else None,
            "BRENT_INSENSITIVE":          brent_insens,
        })
    return pd.DataFrame(filas).sort_values(["CAMPO", "VIGENCIA"]).reset_index(drop=True)


def generar_grafica(df_cmp: pd.DataFrame) -> None:
    """3 paneles: scatter 45° (ambos criterios), histograma de diferencias, top-15 |delta|."""
    valido = df_cmp.dropna(subset=["BK_PONDERADO_PNP_PND", "BK_MAYOR_INCERTIDUMBRE"])
    if valido.empty:
        print("  [WARN] Sin datos validos para graficar.")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: scatter ponderado vs alternativa con linea 45° (igualdad)
    for vig, color in [("2024", "steelblue"), ("2025", "tomato")]:
        sub = valido[valido["VIGENCIA"] == vig]
        if not sub.empty:
            ax1.scatter(sub["BK_PONDERADO_PNP_PND"], sub["BK_MAYOR_INCERTIDUMBRE"],
                        s=40, alpha=0.6, color=color, label=f"Vigencia {vig}")
    lims = [0, max(valido["BK_PONDERADO_PNP_PND"].max(),
                   valido["BK_MAYOR_INCERTIDUMBRE"].max()) * 1.05]
    ax1.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Igualdad (45°)")
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_xlabel("BK ponderado PNP+PND (vigente) [USD/bbl]")
    ax1.set_ylabel("BK clase mas incierta (alternativo) [USD/bbl]")
    ax1.set_title("Piso superior: ponderado vs mayor incertidumbre")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # Panel 2: histograma de la diferencia (alt - ponderado). >0 => alternativa mas exigente
    difs = valido["DELTA_ALT_VS_POND"].dropna()
    ax2.hist(difs, bins=30, color="mediumpurple", alpha=0.8, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=1)
    ax2.axvline(float(difs.median()), color="firebrick", linestyle="--",
                label=f"Mediana={difs.median():.1f}")
    ax2.set_xlabel("Δ (alternativo − ponderado) [USD/bbl]")
    ax2.set_ylabel("N° campos×vigencia")
    ax2.set_title("Cuanto sube el piso con el criterio conservador")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    # Panel 3: top-15 campos por |delta| (vigencia mas reciente por campo)
    recientes = (valido.sort_values("VIGENCIA")
                 .groupby("CAMPO").tail(1))
    top = recientes.reindex(
        recientes["DELTA_ALT_VS_POND"].abs().sort_values(ascending=False).index).head(15)
    y = np.arange(len(top))
    ax3.barh(y - 0.2, top["BK_PONDERADO_PNP_PND"], height=0.4,
             color="steelblue", label="Ponderado")
    ax3.barh(y + 0.2, top["BK_MAYOR_INCERTIDUMBRE"], height=0.4,
             color="tomato", label="Mayor incertidumbre")
    ax3.set_yticks(y)
    ax3.set_yticklabels(top["CAMPO"], fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel("BK piso superior [USD/bbl]")
    ax3.set_title("Top-15 campos por |Δ| (vigencia reciente)")
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Comparativa de criterios de anclaje del breakeven (piso superior)",
                 fontsize=14)
    plt.tight_layout()
    ruta = PLOTS_DIR / "comparativa_bk_ancla.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafica guardada: {ruta}")


if __name__ == "__main__":
    print("=== 06_comparativa_bk.py — Comparativa de anclaje de breakeven ===\n")

    # Reutilizar los lectores del ETL: garantiza misma convencion de swap y mismo ponderado
    print("[1/4] Leyendo breakeven (post-swap, mismas etiquetas que el pipeline)...")
    df_bk = etl.leer_breakeven()

    print("[2/4] Leyendo volumenes por clase (HIST, ultimo año por campo)...")
    df_res = etl.leer_hist1p_reservas()
    df_vols = volumenes_ultimo_anio(df_res)

    print("[3/4] Calculando ponderado vigente (BK_ANCLA_FIN PNP+PND)...")
    df_pond = etl.calcular_breakeven_ponderado(df_bk, df_res)

    print("[4/4] Construyendo comparativa y grafica...")
    df_cmp = construir_comparativa(df_bk, df_pond, df_vols)

    ruta_csv = RESULTADOS / "comparativa_bk_ancla.csv"
    df_cmp.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    print(f"\n  Tabla comparativa: {ruta_csv}  ({len(df_cmp)} filas campo×vigencia)")

    generar_grafica(df_cmp)

    # Resumen ejecutivo
    val = df_cmp.dropna(subset=["DELTA_ALT_VS_POND"])
    if not val.empty:
        print(f"\n{'-'*60}\n  Resumen\n{'-'*60}")
        print(f"  Campos x vigencia comparados : {len(val)}")
        print(f"  Delta mediana (alt - pond)   : {val['DELTA_ALT_VS_POND'].median():.2f} USD/bbl")
        print(f"  Campos donde el alternativo es MAS exigente (delta>0): "
              f"{(val['DELTA_ALT_VS_POND'] > 0).sum()} de {len(val)}")
        print("\n  Top-10 mayor diferencia |delta| (vigencia reciente):")
        recientes = val.sort_values("VIGENCIA").groupby("CAMPO").tail(1)
        top = recientes.reindex(
            recientes["DELTA_ALT_VS_POND"].abs().sort_values(ascending=False).index).head(10)
        for _, r in top.iterrows():
            print(f"    {r['CAMPO']:<22} [{r['VIGENCIA']}] pond={r['BK_PONDERADO_PNP_PND']:.1f}  "
                  f"alt={r['BK_MAYOR_INCERTIDUMBRE']:.1f} ({r['CLASE_INCERTIDUMBRE']})  "
                  f"delta={r['DELTA_ALT_VS_POND']:+.1f}")
    print("\n=== 06_comparativa_bk.py — Completado ===")
