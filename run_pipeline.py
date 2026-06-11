"""
run_pipeline.py — Orquestador del pipeline de prediccion 1P

Secuencia:
  Fase 0:   homologacion (gate solo, sin script de datos)
  Fase 0.5: 05_breakeven.py  → tests/test_05_breakeven.py
  Fase 1:   01_etl.py        → tests/test_01_etl.py
  Fase 2:   02_synthetic.py  → tests/test_02_synthetic.py
  Fase 3:   03_modelo.py     → tests/test_03_modelo.py        (Modelo 1: Neto→Delta)
  Fase 3b:  03b_correlacion_brent.py → tests/test_03b_correlacion.py (Modelo 2: Brent→Neto)
  Fase 4:   04_pbi_export.py → tests/test_04_pbi_export.py    (composicion + meshgrid)

Si cualquier gate falla → pipeline aborta con exit code 1.
100% verde → artefactos validos.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

# ANSI colors (desactiva en CI sin TTY)
VERDE  = "\033[92m" if sys.stdout.isatty() else ""
ROJO   = "\033[91m" if sys.stdout.isatty() else ""
CYAN   = "\033[96m" if sys.stdout.isatty() else ""
NEGRITA = "\033[1m" if sys.stdout.isatty() else ""
RESET  = "\033[0m"  if sys.stdout.isatty() else ""

PYTHON = sys.executable


def correr(cmd: list[str], desc: str) -> tuple[int, float]:
    print(f"\n{CYAN}>> {desc}{RESET}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    return result.returncode, elapsed


FASES = [
    {
        "nombre": "Fase 0 — Homologación",
        "script": None,
        "gate":   "tests/test_00_homologacion.py",
    },
    {
        "nombre": "Fase 0.5 — Breakeven",
        "script": "05_breakeven.py",
        "gate":   "tests/test_05_breakeven.py",
    },
    {
        "nombre": "Fase 1 — ETL",
        "script": "01_etl.py",
        "gate":   "tests/test_01_etl.py",
    },
    {
        "nombre": "Fase 2 — Sintéticos",
        "script": "02_synthetic.py",
        "gate":   "tests/test_02_synthetic.py",
    },
    {
        "nombre": "Fase 3 — Modelo 1 (Neto->Delta)",
        "script": "03_modelo.py",
        "gate":   "tests/test_03_modelo.py",
    },
    {
        "nombre": "Fase 3b — Modelo 2 (Brent->Neto)",
        "script": "03b_correlacion_brent.py",
        "gate":   "tests/test_03b_correlacion.py",
    },
    {
        "nombre": "Fase 4 — Export PBI",
        "script": "04_pbi_export.py",
        "gate":   "tests/test_04_pbi_export.py",
    },
]


def main():
    print(f"\n{NEGRITA}{'='*60}")
    print("  PIPELINE PREDICCIÓN 1P — Castilla/Rubiales")
    print(f"{'='*60}{RESET}")

    resultados = []

    for fase in FASES:
        nombre  = fase["nombre"]
        script  = fase["script"]
        gate    = fase["gate"]

        print(f"\n{NEGRITA}{'-'*50}")
        print(f"  {nombre}")
        print(f"{'-'*50}{RESET}")

        # Correr script de datos (si existe para esta fase)
        script_ok = True
        script_t  = 0.0
        if script:
            rc, script_t = correr([PYTHON, script], f"Corriendo {script}")
            if rc != 0:
                print(f"{ROJO}FAIL: {script} termino con error (rc={rc}){RESET}")
                resultados.append({"fase": nombre, "tests": "N/A (script falló)", "estado": "FAIL"})
                _imprimir_tabla(resultados)
                sys.exit(1)
            script_ok = True

        # Correr gate pytest
        rc_gate, gate_t = correr(
            [PYTHON, "-m", "pytest", gate, "-v", "--tb=short"],
            f"Gate: {gate}",
        )

        if rc_gate == 0:
            estado = "PASS"
            color  = VERDE
        else:
            estado = "FAIL"
            color  = ROJO

        tiempo_total = script_t + gate_t
        resultados.append({
            "fase":   nombre,
            "tests":  gate.split("/")[-1],
            "estado": estado,
            "tiempo": f"{tiempo_total:.1f}s",
        })

        print(f"\n{color}{NEGRITA}  {estado} - {nombre} ({tiempo_total:.1f}s){RESET}")

        if rc_gate != 0:
            print(f"\n{ROJO}Pipeline abortado: {nombre} no superó el gate.{RESET}")
            _imprimir_tabla(resultados)
            sys.exit(1)

    _imprimir_tabla(resultados)
    print(f"\n{VERDE}{NEGRITA}[OK] Pipeline completo -- todos los gates verdes.{RESET}\n")


def _imprimir_tabla(resultados: list[dict]):
    print(f"\n{NEGRITA}{'-'*60}")
    print(f"  {'FASE':<28} {'GATE':<28} {'TIEMPO':>6}  ESTADO")
    print(f"{'-'*60}{RESET}")
    for r in resultados:
        estado = r["estado"]
        color  = VERDE if estado == "PASS" else ROJO
        tiempo = r.get("tiempo", "—")
        print(f"  {r['fase']:<28} {r['tests']:<28} {tiempo:>6}  {color}{estado}{RESET}")
    print(f"{NEGRITA}{'-'*60}{RESET}")


if __name__ == "__main__":
    main()
