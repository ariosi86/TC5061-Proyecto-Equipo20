"""Conjuntos de referencia y actual de P2 para el monitoreo con Evidently.

Referencia: filas de entrenamiento del champion (hora objetivo en 2025).
Actual: julio a septiembre de 2026, un periodo que ninguna versión del modelo vio.
Ambos se construyen con las mismas clases de features_p2.py, para medir el drift
sobre las mismas variables que recibe el modelo.
"""

import argparse
from pathlib import Path

import pandas as pd

from proyecto_mibici.config import (
    P2_INICIO_VALIDACION,
    P2_OBJETIVO,
    P2_VARIABLES,
    STATIONS_PROCESSED_PATH,
    TRIPS_PROCESSED_PATH,
)
from proyecto_mibici.features_p2 import ConstructorSerieRiesgoP2, LimpiadorFlujoP2

INICIO_ACTUAL = "2026-07-01 00:00"
FIN_ACTUAL = "2026-09-30 23:00"
SALIDA = Path(TRIPS_PROCESSED_PATH).parent.parent / "monitoring"

COLUMNAS_VIAJES = ["inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id", "periodo"]


def construir_serie(fin_periodo: str) -> pd.DataFrame:
    """La misma tabla de modelado de P2, pero cortada en fin_periodo."""
    viajes = pd.read_parquet(TRIPS_PROCESSED_PATH, columns=COLUMNAS_VIAJES)
    estaciones = pd.read_parquet(STATIONS_PROCESSED_PATH)

    flujo = LimpiadorFlujoP2(fin_periodo=fin_periodo).fit_transform(viajes)
    constructor = ConstructorSerieRiesgoP2(fin_periodo=fin_periodo)
    constructor.fit(flujo, df_trips=viajes, df_stations=estaciones)
    return constructor.transform(flujo, df_stations=estaciones)


def separar(serie: pd.DataFrame, inicio_actual: str, fin_actual: str):
    """Referencia = entrenamiento (2025); actual = periodo indicado, sin estaciones nuevas."""
    columnas = ["estacion_id", "hora", "hora_objetivo"] + list(P2_VARIABLES) + [P2_OBJETIVO]

    es_referencia = serie["hora_objetivo"] < pd.Timestamp(P2_INICIO_VALIDACION)
    en_actual = serie["hora_objetivo"].between(pd.Timestamp(inicio_actual), pd.Timestamp(fin_actual))

    nuevas = serie.loc[en_actual & serie["estacion_nueva"], "estacion_id"].nunique()
    referencia = serie.loc[es_referencia, columnas].reset_index(drop=True)
    actual = serie.loc[en_actual & ~serie["estacion_nueva"], columnas].reset_index(drop=True)
    return referencia, actual, nuevas


def resumir(nombre: str, tabla: pd.DataFrame) -> None:
    print(
        f"{nombre}: {len(tabla):,} filas | {tabla['estacion_id'].nunique()} estaciones | "
        f"{tabla['hora_objetivo'].min()} a {tabla['hora_objetivo'].max()} | "
        f"tasa de riesgo {tabla[P2_OBJETIVO].mean():.4f}"
    )


def main(inicio_actual: str, fin_actual: str, salida: Path) -> None:
    serie = construir_serie(fin_periodo=fin_actual)
    referencia, actual, nuevas = separar(serie, inicio_actual, fin_actual)

    salida.mkdir(parents=True, exist_ok=True)
    referencia.to_parquet(salida / "p2_referencia.parquet", index=False)
    actual.to_parquet(salida / "p2_actual.parquet", index=False)

    print("Variables:", list(P2_VARIABLES))
    resumir("Referencia", referencia)
    resumir("Actual    ", actual)
    print(f"Estaciones nuevas excluidas del periodo actual: {nuevas}")
    print(f"Guardado en {salida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conjuntos de referencia y actual de P2.")
    parser.add_argument("--inicio-actual", default=INICIO_ACTUAL)
    parser.add_argument("--fin-actual", default=FIN_ACTUAL)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    args = parser.parse_args()
    main(args.inicio_actual, args.fin_actual, args.salida)