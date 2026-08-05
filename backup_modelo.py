"""
backup_modelo.py — copia de seguridad del proyecto (codigo + docs, sin datos).

Empaqueta lo que git rastrea + el propio historial .git + docs/ (que esta
gitignored porque contiene cifras reales) en un unico zip FUERA del repo,
sobreescribiendo la copia anterior. docs/ es la carga critica: ~430 KB de
markdown (MAESTRO.md, CHANGELOG_PREDICCIONES.md, etc.) sin ningun historial
de versiones en otro lugar.

NO incluye datos/ ni resultados/ (datos corporativos Ecopetrol, ~240 MB) —
ver directriz de sesion 2026-08-05.

Uso: python backup_modelo.py
"""
from __future__ import annotations

import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).parent.resolve()
DEST_DIR = REPO_DIR.parent  # fuera del repo: GALLEGO/
DEST_ZIP = DEST_DIR / "Prediccion_backup.zip"

# Carpetas/patrones a incluir ademas de lo rastreado por git
INCLUIR_EXTRA = ["docs", ".git"]

# Exclusiones dentro de lo incluido (datos corporativos, cachés, temporales)
EXCLUIR_PREFIJOS = (
    "datos/", "resultados/",
    "__pycache__/", ".pytest_cache/",
)
EXCLUIR_SUFIJOS = (".pbi/cache.abf", ".pbi/localSettings.json", ".pbi/editorSettings.json")
EXCLUIR_PATRONES = ("~$",)  # locks de Excel


def _git(cmd: list[str]) -> str:
    return subprocess.run(["git"] + cmd, cwd=REPO_DIR, capture_output=True,
                           text=True, check=True).stdout.strip()


def _archivos_git_tracked() -> list[str]:
    """Lista de rutas relativas rastreadas por git (respeta .gitignore)."""
    salida = _git(["ls-files"])
    return [l for l in salida.splitlines() if l]


def _archivos_docs() -> list[str]:
    """docs/ completo (gitignored) — recorrido manual del filesystem."""
    docs_dir = REPO_DIR / "docs"
    if not docs_dir.exists():
        return []
    return [str(p.relative_to(REPO_DIR)).replace("\\", "/")
            for p in docs_dir.rglob("*") if p.is_file()]


def _excluido(ruta_rel: str) -> bool:
    if any(ruta_rel.startswith(p) for p in EXCLUIR_PREFIJOS):
        return True
    if any(ruta_rel.endswith(s) for s in EXCLUIR_SUFIJOS):
        return True
    nombre = ruta_rel.rsplit("/", 1)[-1]
    if any(nombre.startswith(p) for p in EXCLUIR_PATRONES):
        return True
    return False


def _manifest() -> str:
    try:
        commit = _git(["rev-parse", "HEAD"])
        rama = _git(["rev-parse", "--abbrev-ref", "HEAD"])
        sucios = len([l for l in _git(["status", "--porcelain"]).splitlines() if l])
    except subprocess.CalledProcessError:
        commit, rama, sucios = "SIN_COMMITS", "?", -1
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return (
        f"Backup generado: {ts}\n"
        f"Commit HEAD: {commit}\n"
        f"Rama: {rama}\n"
        f"Archivos modificados sin commitear al momento del backup: {sucios}\n"
        f"Contenido: archivos rastreados por git + .git/ (historial) + docs/\n"
        f"Excluido: datos/, resultados/, __pycache__, caches PBI locales\n"
    )


def main() -> None:
    rutas = set(_archivos_git_tracked()) | set(_archivos_docs())
    rutas = sorted(r for r in rutas if not _excluido(r))

    # eliminar backup anterior (sobreescribir, no acumular)
    if DEST_ZIP.exists():
        DEST_ZIP.unlink()

    tmp_zip = DEST_ZIP.with_suffix(".zip.tmp")
    n_bytes = 0
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in rutas:
            ruta_abs = REPO_DIR / rel
            if ruta_abs.is_file():
                zf.write(ruta_abs, arcname=rel)
                n_bytes += ruta_abs.stat().st_size
        # .git/ completo (no esta en git ls-files ni en docs/)
        git_dir = REPO_DIR / ".git"
        n_git = 0
        for p in git_dir.rglob("*"):
            if p.is_file():
                arcname = str(p.relative_to(REPO_DIR)).replace("\\", "/")
                zf.write(p, arcname=arcname)
                n_bytes += p.stat().st_size
                n_git += 1
        zf.writestr("MANIFEST.txt", _manifest())

    tmp_zip.replace(DEST_ZIP)

    mb = n_bytes / (1024 * 1024)
    print(f"Backup escrito: {DEST_ZIP}")
    print(f"  Archivos rastreados+docs: {len(rutas)} | Archivos .git: {n_git}")
    print(f"  Tamaño sin comprimir: {mb:.1f} MB | Zip final: "
          f"{DEST_ZIP.stat().st_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
