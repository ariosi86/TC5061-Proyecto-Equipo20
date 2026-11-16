#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Conjuntos de P3 para el monitoreo con Evidently y el A/B testing (Entregable 8).

P3 trabaja con una fila por estación (perfil funcional de uso), así que cada
conjunto es el perfil de las estaciones calculado sobre un periodo de viajes:

  referencia:            2025, el periodo con el que se entrenó el champion.
  actual:                octubre 2026, el primer mes que ninguna partición de P3 usó.
  referencia_octubre:    octubre 2025, mismo mes un año antes (control de temporada).
  reciente:              octubre 2025 a septiembre 2026, los doce meses previos al
                         periodo actual; con ellos se entrena el challenger.

Todos se construyen con las mismas clases de features_p3.py, así que el drift se
mide sobre las mismas ocho variables que recibe el modelo.
"""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from proyecto_mibici.config import (
    P3_DURACION_MAX_MIN,
    P3_MIN_VIAJES_ESTACION,
    P3_TRAIN_FIN,
    P3_TRAIN_INICIO,
    P3_VARIABLES,
    PROCESSED_DATA_DIR,
    STATIONS_PROCESSED_PATH,
    TRIPS_PROCESSED_PATH,
)
from proyecto_mibici.features import construir_perfil_estaciones, limpiar_viajes_p3

# Periodos [inicio, fin) sobre la fecha de inicio del viaje
PERIODOS_P3 = {
    "referencia": (P3_TRAIN_INICIO, P3_TRAIN_FIN),
    "actual": ("2026-10-01", "2026-11-01"),
    "referencia_octubre": ("2025-10-01", "2025-11-01"),
    "reciente": ("2025-10-01", "2026-10-01"),
}

SALIDA = PROCESSED_DATA_DIR.parent / "monitoring"
COLUMNAS_VIAJES = ["viaje_id", "inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"]

app = typer.Typer(help="Construye los perfiles de estaciones de P3 para monitoreo.")


def ruta_conjunto(nombre: str, salida: Path = SALIDA) -> Path:
    return salida / f"p3_{nombre}.parquet"


def construir_perfiles(periodos: dict = PERIODOS_P3) -> dict:
    """Perfil funcional de las estaciones para cada periodo, con la limpieza de P3."""
    viajes = pd.read_parquet(TRIPS_PROCESSED_PATH, columns=COLUMNAS_VIAJES)
    estaciones = pd.read_parquet(STATIONS_PROCESSED_PATH)
    limpios = limpiar_viajes_p3(viajes, estaciones, duracion_max_min=P3_DURACION_MAX_MIN)

    perfiles = {}
    for nombre, (inicio, fin) in periodos.items():
        en_periodo = limpios["inicio_del_viaje"].between(
            pd.Timestamp(inicio), pd.Timestamp(fin), inclusive="left"
        )
        perfil = construir_perfil_estaciones(limpios[en_periodo], min_viajes=P3_MIN_VIAJES_ESTACION)
        perfil = perfil[P3_VARIABLES].rename_axis("estacion_id")
        perfil.attrs["n_viajes"] = int(en_periodo.sum())
        perfiles[nombre] = perfil
    return perfiles


def cargar_conjunto(nombre: str, salida: Path = SALIDA) -> pd.DataFrame:
    ruta = ruta_conjunto(nombre, salida)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta primero: python -m proyecto_mibici.monitoring.datos_p3"
        )
    return pd.read_parquet(ruta)


@app.command()
def main(salida: Path = typer.Option(SALIDA, "--salida", help="Carpeta de salida de los parquet.")):
    perfiles = construir_perfiles()
    salida.mkdir(parents=True, exist_ok=True)

    for nombre, perfil in perfiles.items():
        inicio, fin = PERIODOS_P3[nombre]
        perfil.to_parquet(ruta_conjunto(nombre, salida))
        logger.info(
            f"{nombre:<19} [{inicio}, {fin}) -> {len(perfil)} estaciones | "
            f"{perfil.attrs['n_viajes']:,} viajes"
        )

    ref = set(perfiles["referencia"].index)
    act = set(perfiles["actual"].index)
    logger.info(
        f"Estaciones en ambos: {len(ref & act)} | solo en actual (nuevas): {len(act - ref)} | "
        f"solo en referencia: {len(ref - act)}"
    )
    logger.success(f"Conjuntos guardados en {salida}")


if __name__ == "__main__":
    app()
