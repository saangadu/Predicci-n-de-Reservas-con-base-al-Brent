"""comparar_tracks.py — A/B track Calidad vs Produccion (offline).

Compara resultados/ (Produccion) vs resultados_calidad/ (Calidad) y emite:
  resultados_calidad/comparativa_tracks.csv   — una fila por campo, orden por materialidad
  resultados_calidad/plots_tracks/{campo}.png — curva Vol vs Brent antes/despues

Mecanismos de divergencia que etiqueta (columna MECANISMO, acumulativos con '+'):
  B_confound : re-anclaje selectivo por confound (flag PRED_REANCLAJE_CONFOUND)
  C_hibrido  : M1 hibrido adoptado en Calidad (seleccion_metodos.csv, ADOPTADO_CALIDAD)
  D_m2       : familia M2 no lineal adoptada (correlacion_brent.csv, ALERTA=M2_SELECCION)
  A_perfil   : divergencia >5% del baseline sin mecanismo explicito (perfil de salida u otro)

Protocolo normalizacion v2: el A/B se juzga por MAE_LOYO (no por SKILL entre semanticas).
No corre el pipeline; asume que ambos tracks ya se corrieron.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR   = Path(__file__).parent
PROD       = BASE_DIR / "resultados"
CAL        = BASE_DIR / "resultados_calidad"
PLOTS      = CAL / "plots_tracks"
PLOTS.mkdir(parents=True, exist_ok=True)

BRENTS_KEY = [60.0, 70.0, 80.0]


def _vol_en(sub: pd.DataFrame, brent: float) -> float:
    """Volumen ISO predicho al Brent mas cercano de la grilla."""
    if sub.empty:
        return np.nan
    i = (sub["BRENT_USD_BBL"] - brent).abs().idxmin()
    return float(sub.loc[i, "VOLUMEN_1P_PREDICHO_MBPE"])


def _sesgo(met: pd.DataFrame) -> dict:
    """|DELTA_REF_ISO|/BASELINE_LATEST por campo (metrica de sesgo-recuperacion)."""
    out = {}
    for _, r in met.iterrows():
        bl = r.get("BASELINE_LATEST")
        dr = r.get("DELTA_REF_ISO")
        if pd.notna(bl) and bl > 0 and pd.notna(dr):
            out[r["CAMPO"]] = abs(float(dr)) / float(bl)
    return out


if __name__ == "__main__":
    mp = pd.read_csv(PROD / "output_matriz_prediccion.csv")
    mc = pd.read_csv(CAL / "output_matriz_prediccion.csv")
    metp = pd.read_csv(PROD / "metricas.csv")
    metc = pd.read_csv(CAL / "metricas.csv")

    mp = mp[mp["MOTOR"] == "Isotonica"]
    mc = mc[mc["MOTOR"] == "Isotonica"]

    mae_p = dict(zip(metp["CAMPO"], metp.get("MAE_LOYO_ISO")))
    mae_c = dict(zip(metc["CAMPO"], metc.get("MAE_LOYO_ISO")))
    reanc = dict(zip(metc["CAMPO"], metc.get("REANCLADO_CONFOUND", pd.Series(dtype=bool))))
    sesgo_p, sesgo_c = _sesgo(metp), _sesgo(metc)

    # Mecanismo C: hibridos M1 adoptados en Calidad (registro de seleccion)
    hibridos = {}
    sel_path = CAL / "seleccion_metodos.csv"
    if sel_path.exists():
        sel = pd.read_csv(sel_path)
        sel = sel[sel["ESTADO"].isin(["ADOPTADO", "ADOPTADO_CALIDAD"])]
        hibridos = dict(zip(sel["CAMPO"], sel["METODO"]))

    # Mecanismo D: familias M2 no lineales aplicadas (correlacion de Calidad)
    m2_sel = {}
    corr_path = CAL / "correlacion_brent.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        if "ALERTA" in corr.columns and "METODO" in corr.columns:
            apl = corr[corr["ALERTA"].fillna("") == "M2_SELECCION"]
            m2_sel = dict(zip(apl["CAMPO"], apl["METODO"]))

    base = mp.groupby("CAMPO")["VOLUMEN_1P_BASELINE_MBPE"].first()
    niv_p = mp.groupby("CAMPO")["NIVEL_CONFIANZA"].first()
    niv_c = mc.groupby("CAMPO")["NIVEL_CONFIANZA"].first()

    filas = []
    for campo in sorted(set(mp["CAMPO"]) & set(mc["CAMPO"])):
        sp = mp[mp["CAMPO"] == campo]
        sc = mc[mc["CAMPO"] == campo]
        bl = float(base.get(campo, np.nan))
        vp = {b: _vol_en(sp, b) for b in BRENTS_KEY}
        vc = {b: _vol_en(sc, b) for b in BRENTS_KEY}
        pares = [(vp[b], vc[b]) for b in BRENTS_KEY if pd.notna(vp[b]) and pd.notna(vc[b])]
        dif_max = max(abs(c - p) for p, c in pares) if pares else np.nan

        # Los mecanismos son acumulativos: un campo puede tener hibrido M1 + M2
        # no lineal + confound a la vez; se listan todos separados por '+'
        mecs = []
        if bool(reanc.get(campo, False)):
            mecs.append("B_confound")
        if campo in hibridos:
            mecs.append(f"C_{hibridos[campo]}")
        if campo in m2_sel:
            mecs.append(f"D_m2:{m2_sel[campo]}")
        if not mecs and pd.notna(dif_max) and dif_max > 0.05 * (bl if bl > 0 else 1):
            mecs.append("A_perfil")
        mecan = "+".join(mecs)

        filas.append({
            "CAMPO": campo,
            "BASELINE_MBPE": round(bl, 2) if pd.notna(bl) else None,
            "NIVEL_PROD": niv_p.get(campo), "NIVEL_CAL": niv_c.get(campo),
            "MAE_LOYO_PROD": mae_p.get(campo), "MAE_LOYO_CAL": mae_c.get(campo),
            "SESGO_PROD": round(sesgo_p.get(campo, np.nan), 3) if campo in sesgo_p else None,
            "SESGO_CAL": round(sesgo_c.get(campo, np.nan), 3) if campo in sesgo_c else None,
            "VOL60_PROD": round(vp[60.0], 2) if pd.notna(vp[60.0]) else None,
            "VOL60_CAL": round(vc[60.0], 2) if pd.notna(vc[60.0]) else None,
            "VOL70_PROD": round(vp[70.0], 2) if pd.notna(vp[70.0]) else None,
            "VOL70_CAL": round(vc[70.0], 2) if pd.notna(vc[70.0]) else None,
            "VOL80_PROD": round(vp[80.0], 2) if pd.notna(vp[80.0]) else None,
            "VOL80_CAL": round(vc[80.0], 2) if pd.notna(vc[80.0]) else None,
            "DIF_MAX_MBPE": round(dif_max, 2) if pd.notna(dif_max) else None,
            "MECANISMO": mecan,
        })

    df = pd.DataFrame(filas).sort_values("BASELINE_MBPE", ascending=False)
    df.to_csv(CAL / "comparativa_tracks.csv", index=False, encoding="utf-8-sig")
    print(f"  Comparativa: {CAL / 'comparativa_tracks.csv'} ({len(df)} campos)")

    # Plots antes/despues para campos materiales que cambiaron
    cambiados = df[(df["BASELINE_MBPE"] >= 5) & (df["MECANISMO"] != "")]
    for _, r in cambiados.iterrows():
        campo = r["CAMPO"]
        sp = mp[mp["CAMPO"] == campo].sort_values("BRENT_USD_BBL")
        sc = mc[mc["CAMPO"] == campo].sort_values("BRENT_USD_BBL")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(sp["BRENT_USD_BBL"], sp["VOLUMEN_1P_PREDICHO_MBPE"],
                "o-", color="#B0413E", ms=3, label="Produccion")
        ax.plot(sc["BRENT_USD_BBL"], sc["VOLUMEN_1P_PREDICHO_MBPE"],
                "s-", color="#1B6535", ms=3, label=f"Calidad ({r['MECANISMO']})")
        if pd.notna(r["BASELINE_MBPE"]):
            ax.axhline(r["BASELINE_MBPE"], ls="--", color="gray", lw=0.8, label="Baseline 1P")
        ax.set_title(f"{campo}  (base={r['BASELINE_MBPE']:.1f} MBPE)")
        ax.set_xlabel("Brent (USD/bbl)"); ax.set_ylabel("Vol 1P predicho (MBPE)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS / f"{campo.replace(' ', '_')}.png", dpi=110)
        plt.close(fig)
    print(f"  Plots antes/despues: {PLOTS} ({len(cambiados)} campos materiales cambiados)")
    print("\n  Top-15 por materialidad:")
    print(df.head(15).to_string(index=False))
