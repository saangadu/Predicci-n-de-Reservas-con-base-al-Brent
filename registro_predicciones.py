"""
registro_predicciones.py — registro estructurado de corridas de prediccion.

Complementa (no reemplaza) resultados/historico_predicciones/ (snapshots CSV
completos, se conservan para auditoria) con una base SQLite que permite
comparar corridas sin volver a parsear 16+ CSV de ~4-7 MB cada uno. Pensado
para la comparacion pre/post-Q2 (tarea diferida, ver docs_public/DIFFICULTIES.md)
y para que backtest_g4.py deje de re-escanear archivos por nombre.

Tablas:
  corridas      — 1 fila por corrida del pipeline (fecha, Q_OBJETIVO, git sha, ...)
  predicciones  — 1 fila por corrida x campo x motor x Brent-ancla
  reales        — 1 fila por campo x quarter, dato real cuando llega (NORTE G4)

No es input de entrenamiento: el reentrenamiento usa exclusivamente datos
reales auditados (regla G4.4, docs/NORTE.md) — este registro es de lectura.

Uso:
  python registro_predicciones.py --backfill   # importa los 16 snapshots existentes
  python registro_predicciones.py --resumen    # lista corridas registradas
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import subprocess
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
RESULTADOS = BASE_DIR / "resultados"
HISTORICO = RESULTADOS / "historico_predicciones"
DB_PATH = RESULTADOS / "registro_predicciones.db"

# Brent ancla: no la grilla completa (eso es lo que hace pesados los snapshots CSV)
BRENT_ANCLAS = [55, 60, 65, 70, 75, 80, 82.9, 85, 90, 95]

DDL = """
CREATE TABLE IF NOT EXISTS corridas (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha           TEXT NOT NULL,
    q_objetivo      TEXT,
    vigencia_base   TEXT,
    brent_ref       REAL,
    n_campos        INTEGER,
    git_sha         TEXT,
    track           TEXT DEFAULT 'produccion',
    fuente_snapshot TEXT,
    UNIQUE(fecha, q_objetivo, fuente_snapshot)
);
CREATE TABLE IF NOT EXISTS predicciones (
    run_id       INTEGER NOT NULL REFERENCES corridas(run_id),
    campo        TEXT NOT NULL,
    motor        TEXT NOT NULL,
    brent_ancla  REAL NOT NULL,
    vol_mbpe     REAL,
    vol_p25_mbpe REAL,
    vol_p75_mbpe REAL,
    nivel_confianza TEXT,
    tipo_modelo  TEXT,
    PRIMARY KEY (run_id, campo, motor, brent_ancla)
);
CREATE TABLE IF NOT EXISTS reales (
    campo      TEXT NOT NULL,
    quarter    TEXT NOT NULL,
    vol_real_mbpe REAL,
    brent_real    REAL,
    PRIMARY KEY (campo, quarter)
);
"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(DDL)
    return con


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                               cwd=BASE_DIR, capture_output=True, text=True,
                               check=True).stdout.strip()
    except Exception:
        return None


def _interpolar_anclas(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce un export completo (grilla paso $1) a las BRENT_ANCLAS via
    interpolacion lineal, igual que prediccion_puntual.py. Tolerante a esquemas
    viejos: snapshots de sesiones anteriores a s14 no traen VOL_P25/P75_MBPE
    ni TIPO_MODELO (bandas/taxonomia se agregaron despues)."""
    cols_estaticas = [c for c in ("NIVEL_CONFIANZA", "TIPO_MODELO") if c in df.columns]
    cols_interp = [c for c in ("VOLUMEN_1P_PREDICHO_MBPE", "VOL_P25_MBPE", "VOL_P75_MBPE")
                   if c in df.columns]
    grid = sorted(df["BRENT_USD_BBL"].unique())
    filas = []
    for b in BRENT_ANCLAS:
        if b < grid[0] or b > grid[-1]:
            continue
        exactos = [g for g in grid if abs(g - b) < 1e-6]
        if exactos:
            sub = df[df["BRENT_USD_BBL"].round(6) == round(exactos[0], 6)].copy()
            sub["_brent_ancla"] = b
        else:
            lo = max(g for g in grid if g < b)
            hi = min(g for g in grid if g > b)
            w = (b - lo) / (hi - lo)
            dl = df[df["BRENT_USD_BBL"] == lo].set_index(["CAMPO", "MOTOR"])
            dh = df[df["BRENT_USD_BBL"] == hi].set_index(["CAMPO", "MOTOR"])
            idx = dl.index.intersection(dh.index)
            sub = dl.loc[idx, cols_estaticas].copy() if cols_estaticas else \
                pd.DataFrame(index=idx)
            for c in cols_interp:
                sub[c] = dl.loc[idx, c] * (1 - w) + dh.loc[idx, c] * w
            sub = sub.reset_index()
            sub["_brent_ancla"] = b
        filas.append(sub)
    out = pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()
    for c in ("NIVEL_CONFIANZA", "TIPO_MODELO", "VOL_P25_MBPE", "VOL_P75_MBPE"):
        if c not in out.columns:
            out[c] = None
    return out


def registrar_corrida(df_out: pd.DataFrame, q_objetivo: str, vigencia_base: str,
                       brent_ref: float, fecha: str, fuente_snapshot: str,
                       track: str = "produccion", resultados_dir: Path | None = None) -> int:
    """Hook llamado desde 04_pbi_export.py tras escribir el snapshot fechado.
    Additivo: no cambia ningun output existente, solo inserta en el registro.
    resultados_dir permite separar el registro por track (Produccion vs Calidad);
    por defecto usa resultados/ (Produccion)."""
    global DB_PATH
    if resultados_dir is not None:
        resultados_dir.mkdir(exist_ok=True)
        DB_PATH = resultados_dir / "registro_predicciones.db"
    con = _conn()
    try:
        cur = con.execute(
            "INSERT OR IGNORE INTO corridas "
            "(fecha, q_objetivo, vigencia_base, brent_ref, n_campos, git_sha, "
            " track, fuente_snapshot) VALUES (?,?,?,?,?,?,?,?)",
            (fecha, q_objetivo, vigencia_base, brent_ref,
             int(df_out["CAMPO"].nunique()), _git_sha(), track, fuente_snapshot))
        con.commit()
        row = con.execute(
            "SELECT run_id FROM corridas WHERE fecha=? AND q_objetivo=? AND fuente_snapshot=?",
            (fecha, q_objetivo, fuente_snapshot)).fetchone()
        run_id = row[0]

        anclas = _interpolar_anclas(df_out)
        if not anclas.empty:
            registros = [
                (run_id, r["CAMPO"], r["MOTOR"], r["_brent_ancla"],
                 r.get("VOLUMEN_1P_PREDICHO_MBPE"), r.get("VOL_P25_MBPE"),
                 r.get("VOL_P75_MBPE"), r.get("NIVEL_CONFIANZA"), r.get("TIPO_MODELO"))
                for _, r in anclas.iterrows()
            ]
            con.executemany(
                "INSERT OR REPLACE INTO predicciones "
                "(run_id, campo, motor, brent_ancla, vol_mbpe, vol_p25_mbpe, "
                " vol_p75_mbpe, nivel_confianza, tipo_modelo) VALUES (?,?,?,?,?,?,?,?,?)",
                registros)
            con.commit()
        return run_id
    finally:
        con.close()


def backfill() -> None:
    """Importa los snapshots de historico_predicciones/ ya existentes."""
    con = _conn()
    n = 0
    for p in sorted(HISTORICO.glob("prediccion_*_generada_*.csv")):
        m = re.match(r"prediccion_(\d{4}_Q\d)_generada_(\d{4}-\d{2}-\d{2})", p.name)
        if not m:
            continue
        q, fecha = m.group(1), m.group(2)
        existe = con.execute(
            "SELECT 1 FROM corridas WHERE fecha=? AND q_objetivo=? AND fuente_snapshot=?",
            (fecha, q, p.name)).fetchone()
        if existe:
            continue
        df = pd.read_csv(p, encoding="utf-8-sig")
        df_iso = df[df["MOTOR"] == "Isotonica"]
        brent_ref = df_iso["BRENT_REF_USD_BBL"].dropna().iloc[0] \
            if "BRENT_REF_USD_BBL" in df_iso.columns and df_iso["BRENT_REF_USD_BBL"].notna().any() \
            else None
        vig = df_iso["VIGENCIA_BASE"].dropna().iloc[0] \
            if "VIGENCIA_BASE" in df_iso.columns and df_iso["VIGENCIA_BASE"].notna().any() else None
        con.close()
        registrar_corrida(df, q, vig, brent_ref, fecha, p.name)
        con = _conn()
        n += 1
        print(f"  Importado: {p.name}")
    con.close()
    print(f"\nBackfill completo: {n} corridas nuevas registradas en {DB_PATH}")


def resumen() -> None:
    con = _conn()
    df = pd.read_sql("SELECT run_id, fecha, q_objetivo, vigencia_base, brent_ref, "
                      "n_campos, git_sha, track FROM corridas ORDER BY fecha", con)
    con.close()
    if df.empty:
        print("Registro vacio. Correr con --backfill primero.")
        return
    print(df.to_string(index=False))
    print(f"\n{len(df)} corridas registradas.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backfill", action="store_true",
                     help="Importa los snapshots existentes en historico_predicciones/")
    ap.add_argument("--resumen", action="store_true", help="Lista las corridas registradas")
    args = ap.parse_args()

    if args.backfill:
        backfill()
    elif args.resumen:
        resumen()
    else:
        ap.print_help()
