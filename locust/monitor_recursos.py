"""Registra el consumo de CPU y memoria de los contenedores del sistema mientras corre Locust.

Toma una muestra de `docker stats --no-stream` cada N segundos y la agrega a un CSV.
Se detiene con Ctrl+C o al cumplirse la duración indicada.

Uso (desde la raíz del repo, en otra terminal mientras corre Locust):
    python locust/monitor_recursos.py --salida reports/locust/u50_recursos.csv --duracion 200
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess
import time

CONTENEDORES = ["mlflow", "api-p1", "api-p2", "api-p3"]
FORMATO = "{{.Name}};{{.CPUPerc}};{{.MemUsage}};{{.MemPerc}};{{.NetIO}}"
UNIDADES = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "kB": 1e3, "MB": 1e6, "GB": 1e9}


def a_mib(texto: str) -> float:
    """Convierte '123.4MiB' o '1.2GiB' a MiB."""
    texto = texto.strip()
    for unidad in sorted(UNIDADES, key=len, reverse=True):
        if texto.endswith(unidad):
            return float(texto[: -len(unidad)]) * UNIDADES[unidad] / 1024**2
    return float("nan")


def muestrear(contenedores: list[str]) -> list[dict]:
    salida = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", FORMATO, *contenedores],
        capture_output=True, text=True, check=True,
    ).stdout
    marca = datetime.now().isoformat(timespec="seconds")
    filas = []
    for linea in salida.strip().splitlines():
        nombre, cpu, mem_uso, mem_pct, net = linea.split(";")
        usado, limite = mem_uso.split("/")
        filas.append({
            "timestamp": marca,
            "contenedor": nombre,
            "cpu_pct": float(cpu.rstrip("%")),
            "mem_mib": round(a_mib(usado), 1),
            "mem_limite_mib": round(a_mib(limite), 1),
            "mem_pct": float(mem_pct.rstrip("%")),
            "net_io": net.strip(),
        })
    return filas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--salida", type=Path, required=True, help="CSV de salida")
    parser.add_argument("--intervalo", type=float, default=2.0, help="Segundos entre muestras")
    parser.add_argument("--duracion", type=float, default=0, help="Segundos totales (0 = hasta Ctrl+C)")
    parser.add_argument("--contenedores", nargs="+", default=CONTENEDORES)
    args = parser.parse_args()

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    inicio = time.monotonic()
    n = 0
    with args.salida.open("w", newline="", encoding="utf-8") as f:
        escritor = None
        try:
            while not args.duracion or time.monotonic() - inicio < args.duracion:
                filas = muestrear(args.contenedores)
                if escritor is None:
                    escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
                    escritor.writeheader()
                escritor.writerows(filas)
                f.flush()
                n += 1
                time.sleep(args.intervalo)
        except KeyboardInterrupt:
            pass
    print(f"{n} muestras guardadas en {args.salida}")


if __name__ == "__main__":
    main()
