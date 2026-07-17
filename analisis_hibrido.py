"""
analisis_hibrido.py — Champion-challenger del modelo HIBRIDO M1 por campo (OFFLINE)

Directriz usuario 2026-07-15; expansion s12 (2026-07-15):
  - Universo = portafolio completo con modelo (excluye DECK_PLANO e INSENSIBLE_PRECIO)
  - Adopcion = mejora MAE > 5% vs champion isotonica + checks fisicos
  - Validacion = 2.o mejor del ranking LOYO (dinamica: Suave, otro hibrido, champion…)

PROTOCOLO por campo:
  Pool         = champion (isotonica) + hibrido:* (5 motores superiores)
                 + Suave/Plano standalone (curva completa, distinta del hibrido)
  Ranking      = MAE LOYO (leave-one-year-out) si N_VIG>=2; LOO fallback
  Adopcion     = mejor hibrido:* con mejora >5% vs champion + monotonia/C5/C6
                 + divergencia vs 2.o del ranking < 30%
  Validacion   = 2.o lugar del ranking completo (METODO_VALIDACION)
  p_junta      = max(BK_ANCLA_FIN + TRANSICION_USD, ultimo precio sintetico)

SALIDAS:
  resultados_calidad/benchmark_hibrido.csv
  resultados_calidad/benchmark_hibrido_resumen.csv
  resultados_calidad/seleccion_metodos.csv  += ADOPTADO_CALIDAD (hibrido:*)
  resultados_calidad/plots_hibrido/<campo>.png
  docs/informe_hibridos.html
"""
import importlib
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import motores_modelo1 as M
import seleccion_reglas as R

m3 = importlib.import_module("03_modelo")
FEATURE, TARGET = m3.FEATURE, m3.TARGET

BASE  = Path(__file__).parent
CAL   = BASE / "resultados_calidad"
PLOTS = CAL / "plots_hibrido"
PLOTS.mkdir(parents=True, exist_ok=True)

MEJORA_MIN     = R.MEJORA_ADOPT      # >5% vs champion (s12)
TRANSICION_USD = 6.0
DIVERGENCIA_MAX = R.DIVERGENCIA_MAX


def _datos_campo(tab: pd.DataFrame, campo: str) -> tuple:
    """(reales con ANIO, sinteticos) del campo, con TARGET y FEATURE disponibles."""
    sub = tab[(tab["CAMPO"] == campo) & tab[TARGET].notna() & tab[FEATURE].notna()]
    reales = sub[~sub["ES_SINTETICO"]].copy()
    reales["ANIO"] = reales["VIGENCIA"].astype(str).str.slice(0, 4)
    return reales.reset_index(drop=True), sub[sub["ES_SINTETICO"]].reset_index(drop=True)


def _p_junta_campo(tab: pd.DataFrame, campo: str, sint: pd.DataFrame) -> float:
    """p_junta por campo: fin del gobierno de la escalera de breakevens."""
    sub = tab[tab["CAMPO"] == campo].dropna(subset=["BK_ANCLA_FIN_USD_BBL"])
    bk_sup = (float(sub.sort_values("VIGENCIA_BREAKEVEN", ascending=False)
                    .iloc[0]["BK_ANCLA_FIN_USD_BBL"]) if len(sub) else np.nan)
    p_sint_max = float(sint[FEATURE].max()) if len(sint) else np.nan
    candidatos = [v for v in (bk_sup + TRANSICION_USD if pd.notna(bk_sup) else np.nan,
                              p_sint_max) if pd.notna(v)]
    return max(candidatos) if candidatos else np.nan


def _validacion(reales: pd.DataFrame, sint: pd.DataFrame, factory) -> tuple:
    """MAE LOYO (>=2 años) o LOO; pesos sinteticos por nivel como 03_modelo."""
    n_real = len(reales)
    if n_real < 2:
        return np.nan, "N/A"
    x_s = sint[FEATURE].values if len(sint) else np.empty(0)
    y_s = sint[TARGET].values if len(sint) else np.empty(0)
    w_s = m3.pesos_sinteticos_tramo(sint, n_real)[0] if len(sint) else np.empty(0)

    anios = reales["ANIO"].values
    uni = sorted(set(anios))
    if len(uni) >= 2:
        folds = [(anios != a, anios == a) for a in uni]
        metrica = "LOYO"
    else:
        folds = [(np.arange(n_real) != i, np.arange(n_real) == i)
                 for i in range(n_real)]
        metrica = "LOO"

    errs = []
    for tr, te in folds:
        if tr.sum() < 2:
            continue
        x_t = np.concatenate([x_s, reales.loc[tr, FEATURE].values]) if len(x_s) \
            else reales.loc[tr, FEATURE].values
        y_t = np.concatenate([y_s, reales.loc[tr, TARGET].values]) if len(y_s) \
            else reales.loc[tr, TARGET].values
        w_t = np.concatenate([w_s, np.ones(int(tr.sum()))]) if len(w_s) \
            else np.ones(int(tr.sum()))
        mo = factory().fit(x_t, y_t, sample_weight=w_t)
        y_h = mo.predict(reales.loc[te, FEATURE].values)
        errs.extend(np.abs(np.asarray(y_h) - reales.loc[te, TARGET].values).tolist())
    return (float(np.mean(errs)) if errs else np.nan), metrica


def _xyw(reales, sint):
    """Matrices finales (reales + sinteticos) con pesos por nivel."""
    x = np.concatenate([sint[FEATURE].values, reales[FEATURE].values]) \
        if len(sint) else reales[FEATURE].values
    y = np.concatenate([sint[TARGET].values, reales[TARGET].values]) \
        if len(sint) else reales[TARGET].values
    w = np.concatenate([m3.pesos_sinteticos_tramo(sint, len(reales))[0],
                        np.ones(len(reales))]) if len(sint) else np.ones(len(reales))
    return x, y, w


def _checks_base(modelo, baseline, p_ref, bk_pdp) -> dict:
    """Monotonia + C5/C6 del re-anclaje (independiente del segundo motor)."""
    out = {}
    grid = np.linspace(40.0, 120.0, 161)
    out["monotonia"] = bool(np.all(np.diff(modelo.predict(grid)) >= -1e-9))
    if pd.notna(p_ref) and pd.notna(baseline):
        d_ref = float(modelo.predict([p_ref])[0])
        v_ref = float(M.volumen_anclado(modelo, [p_ref], baseline, d_ref,
                                        bk_pdp, p_ref=p_ref)[0])
        out["c5"] = abs(v_ref - max(baseline, 0.0)) < 1e-6
        piso = M.piso_efectivo(bk_pdp, p_ref, baseline)
        if piso is not None:
            v0 = float(M.volumen_anclado(modelo, [piso - 5.0], baseline, d_ref,
                                         bk_pdp, p_ref=p_ref)[0])
            out["c6"] = v0 == 0.0
        else:
            out["c6"] = True
    else:
        out["c5"] = out["c6"] = True
    return out


def _plot(campo, reales, sint, champ_m, prim_m, valid_m, p_junta,
          nom_prim, nom_valid, mae_ch, mae_p, mae_v, metrica):
    """3 curvas: champion isotonica, primario adoptado, 2.o (validacion)."""
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.scatter(reales[FEATURE], reales[TARGET], color="steelblue", s=55, zorder=5,
               label=f"Reales (n={len(reales)})")
    if len(sint):
        ax.scatter(sint[FEATURE], sint[TARGET], color="tomato", s=22, alpha=0.5,
                   marker="x", zorder=4, label=f"Sinteticos (n={len(sint)})")
    lo = float(min(sint[FEATURE].min() if len(sint) else reales[FEATURE].min(),
                   reales[FEATURE].min())) - 5
    hi = float(reales[FEATURE].max()) + 12
    grid = np.linspace(lo, hi, 250)
    ax.plot(grid, champ_m.predict(grid), color="darkorange", lw=1.8,
            label=f"Champion isotonica ({metrica}={mae_ch:.2f})")
    ax.plot(grid, prim_m.predict(grid), color="seagreen", lw=2, ls="--",
            label=f"Primario {nom_prim} ({metrica}={mae_p:.2f})")
    ax.plot(grid, valid_m.predict(grid), color="royalblue", lw=1.5, ls=":",
            label=f"Validacion {nom_valid} ({metrica}={mae_v:.2f})")
    if pd.notna(p_junta):
        ax.axvline(p_junta, color="purple", lw=1.2, ls=":",
                   label=f"Junta p={p_junta:.1f}")
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.set_xlabel("Precio Neto (USD/bbl)"); ax.set_ylabel("Δ Reservas (MBPE)")
    ax.set_title(f"{campo} — primario vs validacion (2.o LOYO)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS / f"{campo.replace(' ', '_').replace('/', '_')}.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)


def _informe_html(dfb: pd.DataFrame, resumen: pd.DataFrame, n_adop: int) -> None:
    ad = (resumen[resumen["ESTADO"] == "ADOPTADO_CALIDAD"]
          .to_html(index=False) if n_adop else
          "<p>(ningun hibrido supero umbral + checks)</p>")
    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<title>Modelo hibrido M1 — portafolio completo (s12)</title>
<style>
 body {{ font-family: Segoe UI, sans-serif; margin: 2em auto; max-width: 1100px; color: #222; }}
 table {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
 th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
 th {{ background: #f0f3f5; }} td:first-child, th:first-child {{ text-align: left; }}
 h1 {{ font-size: 1.4em; }} h2 {{ font-size: 1.1em; margin-top: 2em; }}
 .nota {{ background: #fff8e0; padding: 0.8em 1em; border-left: 4px solid #d4a017; }}
</style></head><body>
<h1>Modelo hibrido M1 — portafolio + validacion dinamica ({date.today()})</h1>
<p class="nota">Adopcion (s12): mejora MAE (LOYO/LOO) &gt; {MEJORA_MIN:.0%} vs champion
isotonica + monotonia $40–120 + C5/C6 + divergencia primario↔2.o LOYO
&lt; {DIVERGENCIA_MAX:.0%}. Validacion = 2.o del ranking (no Suave fija).
DECK_PLANO e INSENSIBLE_PRECIO excluidos. ESTADO=ADOPTADO_CALIDAD (solo Calidad).</p>
<h2>Adopciones ({n_adop})</h2>
{ad}
<h2>Resumen por campo</h2>
{resumen.to_html(index=False)}
<h2>Evidencia completa</h2>
{dfb.to_html(index=False)}
</body></html>"""
    ruta = BASE / "docs" / "informe_hibridos.html"
    ruta.write_text(html, encoding="utf-8")
    print(f"Informe: {ruta}")


def _pool_candidatos(p_junta: float) -> dict:
    """{id_metodo: factory} — champion + hibridos + Suave/Plano standalone."""
    pool = {"champion": M.MotorIsotonico}
    for nombre, factory in M.CANDIDATOS_HIBRIDO.items():
        # Captura por valor para que el lambda no se cierre sobre la ultima iteracion
        pool[f"hibrido:{nombre}"] = (
            lambda f=factory, pj=p_junta: M.MotorHibrido(f, pj))
    pool["Suave"] = M.MotorSuave
    pool["Plano"] = M.MotorPlano
    return pool


if __name__ == "__main__":
    print("=== analisis_hibrido.py — portafolio + ranking LOYO (s12) ===\n")
    tab = pd.read_parquet(BASE / "datos" / "staging" / "tablon_unico.parquet")
    met = pd.read_csv(BASE / "resultados" / "metricas.csv").set_index("CAMPO")
    mtx = pd.read_csv(BASE / "resultados" / "output_matriz_prediccion.csv")
    exp = mtx.drop_duplicates("CAMPO")
    nivel_map = dict(zip(exp["CAMPO"], exp["NIVEL_CONFIANZA"])) \
        if "NIVEL_CONFIANZA" in exp.columns else {}
    base_map = exp.set_index("CAMPO")["VOLUMEN_1P_BASELINE_MBPE"]

    portafolio = R.campos_portafolio_con_modelo(BASE)
    print(f"Portafolio con modelo: {len(portafolio)} campos\n")

    bench, resumen_filas, adopciones = [], [], []
    for campo in portafolio:
        motivo = R.excluir_hibrido(campo, met, nivel_map)
        baseline = float(base_map.get(campo, np.nan))
        if motivo:
            resumen_filas.append({
                "CAMPO": campo, "BASELINE_MBPE": round(baseline, 1) if pd.notna(baseline) else None,
                "MOTOR_PRIMARIO": None, "MOTOR_VALIDACION": None,
                "MAE_PRIMARIO": None, "MAE_VALID": None, "DIVERGENCIA_PCT": None,
                "ESTADO": "SKIP", "MOTIVO_SKIP": motivo,
            })
            bench.append({"CAMPO": campo, "BASELINE_MBPE": round(baseline, 1)
                          if pd.notna(baseline) else None,
                          "MOTOR": "SKIP", "MAE": None, "METRICA": "N/A",
                          "MEJORA_PCT": None, "RANK": None,
                          "ES_PRIMARIO": False, "ES_VALIDACION": False,
                          "MOTIVO_SKIP": motivo})
            continue

        reales, sint = _datos_campo(tab, campo)
        if len(reales) < R.N_MIN_REAL_HIBRIDO:
            resumen_filas.append({
                "CAMPO": campo, "BASELINE_MBPE": round(baseline, 1),
                "MOTOR_PRIMARIO": None, "MOTOR_VALIDACION": None,
                "MAE_PRIMARIO": None, "MAE_VALID": None, "DIVERGENCIA_PCT": None,
                "ESTADO": "SKIP", "MOTIVO_SKIP": "pocos_reales",
            })
            continue

        p_ref = (float(met.loc[campo, "P_REF_USD_BBL"])
                 if campo in met.index and pd.notna(met.loc[campo, "P_REF_USD_BBL"])
                 else np.nan)
        bk_fin, bk_pdp, _ = m3._anclas_campo(tab[tab["CAMPO"] == campo])
        p_junta = _p_junta_campo(tab, campo, sint)
        if pd.isna(p_junta):
            resumen_filas.append({
                "CAMPO": campo, "BASELINE_MBPE": round(baseline, 1),
                "MOTOR_PRIMARIO": None, "MOTOR_VALIDACION": None,
                "MAE_PRIMARIO": None, "MAE_VALID": None, "DIVERGENCIA_PCT": None,
                "ESTADO": "SKIP", "MOTIVO_SKIP": "junta_invalida",
            })
            continue
        n_arriba = int((reales[FEATURE] >= p_junta).sum())
        if n_arriba < 2:
            resumen_filas.append({
                "CAMPO": campo, "BASELINE_MBPE": round(baseline, 1),
                "MOTOR_PRIMARIO": None, "MOTOR_VALIDACION": None,
                "MAE_PRIMARIO": None, "MAE_VALID": None, "DIVERGENCIA_PCT": None,
                "ESTADO": "SKIP", "MOTIVO_SKIP": "junta_invalida",
            })
            bench.append({"CAMPO": campo, "BASELINE_MBPE": round(baseline, 1),
                          "MOTOR": "SKIP", "MAE": None, "METRICA": "N/A",
                          "MEJORA_PCT": None, "RANK": None,
                          "ES_PRIMARIO": False, "ES_VALIDACION": False,
                          "MOTIVO_SKIP": f"junta_{p_junta:.1f}_n_arriba={n_arriba}"})
            continue

        # ── Ranking LOYO de todos los candidatos ──────────────────────────────
        pool = _pool_candidatos(p_junta)
        scores = {}   # id -> (mae, metrica, modelo_fit)
        x, y, w = _xyw(reales, sint)
        metrica_campo = "LOYO"
        for mid, factory in pool.items():
            mae, metrica = _validacion(reales, sint, factory)
            if pd.isna(mae):
                continue
            metrica_campo = metrica
            modelo = factory().fit(x, y, sample_weight=w)
            scores[mid] = (mae, metrica, modelo)

        if "champion" not in scores:
            print(f"  [SKIP] {campo}: champion no evaluable")
            continue
        mae_ch = scores["champion"][0]
        ranking = sorted(scores.keys(), key=lambda k: scores[k][0])
        for rank_i, mid in enumerate(ranking, start=1):
            mae = scores[mid][0]
            mejora = (mae_ch - mae) / mae_ch if mae_ch > 0 else np.nan
            bench.append({
                "CAMPO": campo, "BASELINE_MBPE": round(baseline, 1),
                "MOTOR": mid, "MAE": round(mae, 3), "METRICA": metrica_campo,
                "MEJORA_PCT": round(mejora * 100, 1) if pd.notna(mejora) else None,
                "RANK": rank_i,
                "ES_PRIMARIO": False, "ES_VALIDACION": False, "MOTIVO_SKIP": None,
            })

        # Validacion = 2.o del ranking completo
        if len(ranking) < 2:
            resumen_filas.append({
                "CAMPO": campo, "BASELINE_MBPE": round(baseline, 1),
                "MOTOR_PRIMARIO": None, "MOTOR_VALIDACION": None,
                "MAE_PRIMARIO": None, "MAE_VALID": None, "DIVERGENCIA_PCT": None,
                "ESTADO": "SKIP", "MOTIVO_SKIP": "sin_segundo",
            })
            continue
        valid_id = ranking[1]
        mae_valid = scores[valid_id][0]
        modelo_valid = scores[valid_id][2]

        # Candidato a adopcion = mejor hibrido:* por MAE (no standalone)
        hibridos = {k: v for k, v in scores.items() if k.startswith("hibrido:")}
        estado = None
        prim_id = div = None
        if hibridos:
            mejor_h = min(hibridos, key=lambda k: hibridos[k][0])
            mae_h = hibridos[mejor_h][0]
            mejora_h = (mae_ch - mae_h) / mae_ch if mae_ch > 0 else np.nan
            modelo_h = hibridos[mejor_h][2]
            chk = _checks_base(modelo_h, baseline, p_ref, bk_pdp)
            # Divergencia del hibrido adoptado vs 2.o del ranking (validacion dinamica)
            div = R.divergencia_volumen(modelo_h, modelo_valid, baseline,
                                        reales[FEATURE].values)
            div_ok = (pd.isna(div) or div < DIVERGENCIA_MAX)
            checks_ok = bool(chk["monotonia"] and chk["c5"] and chk["c6"] and div_ok)
            estado = R.estado_por_mejora(mejora_h, checks_ok)
            if estado == "ADOPTADO_CALIDAD":
                prim_id = mejor_h
                # Marcar ES_PRIMARIO / ES_VALIDACION en el bench
                for row in bench:
                    if row["CAMPO"] == campo and row["MOTOR"] == prim_id:
                        row["ES_PRIMARIO"] = True
                    if row["CAMPO"] == campo and row["MOTOR"] == valid_id:
                        row["ES_VALIDACION"] = True
                # Si validacion == primario (empate raro), tomar el siguiente del ranking
                if valid_id == prim_id and len(ranking) >= 3:
                    valid_id = ranking[2]
                    mae_valid = scores[valid_id][0]
                    modelo_valid = scores[valid_id][2]
                    div = R.divergencia_volumen(modelo_h, modelo_valid, baseline,
                                                reales[FEATURE].values)
                    for row in bench:
                        if row["CAMPO"] == campo:
                            row["ES_VALIDACION"] = (row["MOTOR"] == valid_id)

                adopciones.append({
                    "CAMPO": campo, "METODO": prim_id,
                    "METODO_VALIDACION": valid_id,
                    "ARQUETIPO": "hibrido-escalera+motor-superior",
                    "BASELINE_MBPE": round(baseline, 1),
                    "MAE_LOYO_CHAMPION": round(mae_ch, 3),
                    "MAE_LOYO_METODO": round(mae_h, 3),
                    "MEJORA_PCT": round(mejora_h * 100, 1),
                    "DIVERGENCIA_PCT": round(div * 100, 2) if pd.notna(div) else None,
                    "HIPOTESIS": (
                        f"Hibrido s12: escalera intacta hasta p_junta={p_junta:.1f}; "
                        f"arriba {prim_id.split(':',1)[1]}. Validacion dinamica = "
                        f"{valid_id} (2.o LOYO). {metrica_campo} {mae_ch:.2f}->{mae_h:.2f} "
                        f"({mejora_h:+.0%}). Div={div:.2f}." if pd.notna(div) else
                        f"Hibrido s12: {prim_id}; validacion={valid_id}."
                    ),
                    "ESTADO": "ADOPTADO_CALIDAD", "FECHA": str(date.today()),
                })
                _plot(campo, reales, sint, scores["champion"][2], modelo_h,
                      modelo_valid, p_junta, prim_id, valid_id,
                      mae_ch, mae_h, mae_valid, metrica_campo)
                print(f"  [ADOPTAR] {campo:<22} {prim_id:<20} valid={valid_id:<16} "
                      f"{metrica_campo} {mae_ch:.2f}->{mae_h:.2f} ({mejora_h:+.0%})")
            else:
                motivo_no = ("checks_fail" if not checks_ok else
                             f"mejora={mejora_h:+.1%}" if pd.notna(mejora_h) else "n/a")
                print(f"  [NO-ADOPTA] {campo:<20} mejor={mejor_h} ({motivo_no})")

        resumen_filas.append({
            "CAMPO": campo, "BASELINE_MBPE": round(baseline, 1),
            "MOTOR_PRIMARIO": prim_id,
            "MOTOR_VALIDACION": valid_id if prim_id else ranking[1],
            "MAE_PRIMARIO": round(scores[prim_id][0], 3) if prim_id else None,
            "MAE_VALID": round(mae_valid, 3),
            "DIVERGENCIA_PCT": round(div * 100, 2) if div is not None and pd.notna(div) else None,
            "ESTADO": estado or "SIN_ADOPCION",
            "MOTIVO_SKIP": None,
        })

    dfb = pd.DataFrame(bench)
    dfb.to_csv(CAL / "benchmark_hibrido.csv", index=False, encoding="utf-8-sig")
    print(f"\nBenchmark: {CAL / 'benchmark_hibrido.csv'} ({len(dfb)} filas)")

    dfr = pd.DataFrame(resumen_filas)
    dfr.to_csv(CAL / "benchmark_hibrido_resumen.csv", index=False, encoding="utf-8-sig")
    print(f"Resumen:  {CAL / 'benchmark_hibrido_resumen.csv'} ({len(dfr)} campos)")

    # Registro: regenerar hibridos ADOPTADO_CALIDAD; conservar historico s8/s9
    ruta_sel = CAL / "seleccion_metodos.csv"
    sel = pd.read_csv(ruta_sel) if ruta_sel.exists() else pd.DataFrame()
    if len(sel):
        sel = sel[~(sel["METODO"].astype(str).str.startswith("hibrido:")
                    & (sel["ESTADO"] == "ADOPTADO_CALIDAD"))]
        # Asegurar columna METODO_VALIDACION sin romper filas historicas
        if "METODO_VALIDACION" not in sel.columns:
            sel["METODO_VALIDACION"] = np.nan
        if "DIVERGENCIA_PCT" not in sel.columns:
            sel["DIVERGENCIA_PCT"] = np.nan
    sel = pd.concat([sel, pd.DataFrame(adopciones)], ignore_index=True)
    sel.to_csv(ruta_sel, index=False, encoding="utf-8-sig")
    print(f"Seleccion: {ruta_sel} (+{len(adopciones)} hibridos ADOPTADO_CALIDAD)")

    _informe_html(dfb, dfr, n_adop=len(adopciones))
