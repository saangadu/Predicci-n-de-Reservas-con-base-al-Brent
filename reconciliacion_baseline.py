"""reconciliacion_baseline.py — (offline) Reconciliación Vol 1P Baseline export vs cierre
certificado ECP S.A.

POR QUÉ: el tablero agrega VOLUMEN_1P_BASELINE_MBPE = 1703.5 MBPE, el cierre 2025
certificado para ECP S.A. es 1685 MBPE (+18.5, ~1.1%). Este script homologa HIST 1P.xlsx
(cierre 2025 crudo, por CAMPO granular) contra DIM_CAMPO/UNIFICADO y compara contra el
export del pipeline para localizar el origen exacto de la diferencia, en vez de asumir
un solo culpable (duplicados / gas / cobertura SIN_MODELO / alcance sin_consolidado).

No modifica ningun modelo ni dato — solo diagnostico. Sin gate pytest (offline).
"""
from pathlib import Path

import pandas as pd

from homologacion import Homologador

BASE_DIR = Path(__file__).parent
RAW = BASE_DIR / "datos" / "raw"
RESULTADOS = BASE_DIR / "resultados"

AÑO_CIERRE = 2025
CIERRE_CERTIFICADO_ECP_SA = 1685.0


def cargar_hist_homologado(anio: int = AÑO_CIERRE) -> pd.DataFrame:
    """HIST 1P.xlsx, año `anio`, 1P homologado a UNIFICADO. Suma componentes
    físicos que coexisten bajo el mismo UNIFICADO (agregación v3, igual que 01_etl).

    `anio` parametrizable (default AÑO_CIERRE=2025) para que 04_pbi_export.py derive
    el cierre de comparación del año correcto en vigencias futuras (A-1 del deck),
    sin hardcodear el año ni el número (1685 es un resultado, no una constante)."""
    df = pd.read_excel(RAW / "HIST 1P.xlsx", sheet_name="Reservas (MBPE)", engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    col_anio = [c for c in df.columns if c.strip().lower() in ("año", "ano", "año")][0]
    df["1P"] = df["1P"].astype(str).str.replace(",", ".", regex=False).astype(float)
    cierre = df[df[col_anio] == anio][["CAMPO", "1P"]].copy()

    hom = Homologador()
    res = hom.homologar_serie(cierre["CAMPO"])
    cierre["UNIFICADO"] = res["CAMPO"]
    cierre["HOMOLOG_FLAG"] = res["HOMOLOG_FLAG"]
    return cierre


def clasificar_scope(unificado: str, hom: Homologador) -> str:
    attrs = hom.atributos(unificado)
    gerencia = str(attrs.get("NEW GERENCIA", "") or "")
    if "FILIAL" in gerencia.upper():
        return "FILIAL"
    return "PORTAFOLIO"


def main():
    hist = cargar_hist_homologado()
    hom = Homologador()
    hist["SCOPE"] = hist["UNIFICADO"].apply(lambda u: clasificar_scope(u, hom))

    no_hom = hist[hist["HOMOLOG_FLAG"] == "NO_HOMOLOGADO"]
    if len(no_hom):
        print(f"[AVISO] {len(no_hom)} filas de HIST no homologadas (excluidas del "
              f"agregado por UNIFICADO): {sorted(no_hom['CAMPO'].unique())[:10]}...")

    # Agrega por UNIFICADO (suma componentes físicos, igual que 01_etl agregación v3)
    hist_agg = (hist[hist["HOMOLOG_FLAG"] == "OK"]
                .groupby(["UNIFICADO", "SCOPE"], as_index=False)["1P"].sum())

    total_portafolio = hist_agg.loc[hist_agg["SCOPE"] == "PORTAFOLIO", "1P"].sum()
    total_filial = hist_agg.loc[hist_agg["SCOPE"] == "FILIAL", "1P"].sum()

    print(f"\n{'='*70}\n  RECONCILIACIÓN CIERRE {AÑO_CIERRE} — HIST 1P.xlsx homologado\n{'='*70}")
    print(f"  Certificado ECP S.A. (dato usuario)      : {CIERRE_CERTIFICADO_ECP_SA:10.2f} MBPE")
    print(f"  HIST homologado, portafolio (sin filial) : {total_portafolio:10.2f} MBPE"
          f"  (delta {total_portafolio - CIERRE_CERTIFICADO_ECP_SA:+.2f})")
    print(f"  HIST homologado, filiales                : {total_filial:10.2f} MBPE")
    print(f"  HIST homologado, TOTAL (portaf.+filial)   : {total_portafolio + total_filial:10.2f} MBPE"
          f"  (delta {total_portafolio + total_filial - CIERRE_CERTIFICADO_ECP_SA:+.2f})")

    # ── Cruce contra el export del pipeline ──────────────────────────────
    export = pd.read_csv(RESULTADOS / "output_matriz_prediccion.csv").drop_duplicates("CAMPO")
    export_total = export["VOLUMEN_1P_BASELINE_MBPE"].sum()
    print(f"\n  Export pipeline (VOLUMEN_1P_BASELINE_MBPE, {len(export)} campos): "
          f"{export_total:10.2f} MBPE  (delta vs certificado {export_total - CIERRE_CERTIFICADO_ECP_SA:+.2f})")

    hist_portaf = hist_agg[hist_agg["SCOPE"] == "PORTAFOLIO"].set_index("UNIFICADO")["1P"]
    exp_base = export.set_index("CAMPO")["VOLUMEN_1P_BASELINE_MBPE"]

    solo_hist = sorted(set(hist_portaf.index) - set(exp_base.index))
    solo_export = sorted(set(exp_base.index) - set(hist_portaf.index))
    comunes = sorted(set(hist_portaf.index) & set(exp_base.index))

    print(f"\n  Campos en HIST pero AUSENTES del export : {len(solo_hist)} "
          f"(suma {hist_portaf.reindex(solo_hist).sum():.2f} MBPE)")
    if solo_hist:
        top_ausentes = hist_portaf.reindex(solo_hist).sort_values(ascending=False).head(15)
        print(top_ausentes.to_string())
    print(f"\n  Campos en export pero AUSENTES de HIST {AÑO_CIERRE}: {len(solo_export)} "
          f"(suma {exp_base.reindex(solo_export).sum():.2f} MBPE)")
    if solo_export:
        print(sorted(solo_export))

    cmp = pd.DataFrame({"HIST": hist_portaf.reindex(comunes), "EXPORT": exp_base.reindex(comunes)})
    cmp["DIF"] = cmp["EXPORT"] - cmp["HIST"]
    dif_material = cmp[cmp["DIF"].abs() > 0.05].sort_values("DIF", key=abs, ascending=False)
    print(f"\n  Campos comunes con DIF material (|dif|>0.05 MBPE): {len(dif_material)} "
          f"de {len(comunes)} — suma DIF total campos comunes: {cmp['DIF'].sum():.2f} MBPE")
    if len(dif_material):
        print(dif_material.head(25).to_string())

    # ── Bucket INSENSIBLE_PRECIO (gas) dentro del export ─────────────────
    gas = export[export["NIVEL_CONFIANZA"] == "INSENSIBLE_PRECIO"]
    print(f"\n  Bucket INSENSIBLE_PRECIO (gas) en export: {len(gas)} campos, "
          f"{gas['VOLUMEN_1P_BASELINE_MBPE'].sum():.2f} MBPE")

    # ── Resumen de buckets del gap ────────────────────────────────────────
    gap_total = export_total - CIERRE_CERTIFICADO_ECP_SA
    gap_ausentes_hist = -hist_portaf.reindex(solo_hist).sum()  # campos en HIST no exportados: restan
    gap_solo_export = exp_base.reindex(solo_export).sum()      # campos exportados sin HIST: suman
    gap_dif_comunes = cmp["DIF"].sum()
    print(f"\n{'='*70}\n  RESUMEN DEL GAP ({gap_total:+.2f} MBPE export vs certificado)\n{'='*70}")
    print(f"  Campos en HIST no exportados (restan del cierre pero no del export): {gap_ausentes_hist:+.2f}")
    print(f"  Campos exportados sin HIST {AÑO_CIERRE} (ej. sin_consolidado inverso): {gap_solo_export:+.2f}")
    print(f"  Diferencia en campos comunes (redondeo / vigencia / criterio)      : {gap_dif_comunes:+.2f}")
    print(f"  Suma de los tres buckets                                          : "
          f"{gap_ausentes_hist + gap_solo_export + gap_dif_comunes:+.2f}  (deberia == {gap_total:+.2f})")

    out = RESULTADOS / "reconciliacion_baseline_2025.csv"
    cmp.reset_index().rename(columns={"index": "CAMPO"}).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  Detalle guardado en {out}")


if __name__ == "__main__":
    main()
