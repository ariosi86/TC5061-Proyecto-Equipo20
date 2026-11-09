#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de inferencia para P3: Segmentación Funcional de Estaciones.
Carga el Pipeline serializado (ClusterKMeansP3) y asigna:
  - Grupo asignado por KMeans K=4 (columna 'grupo' y 'grupo_kmeans_k4')
  - Grupo estimado por demarcación geográfica (columna 'baseline_municipio')
Opera exclusivamente con almacenamiento Parquet.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import typer
from loguru import logger

from proyecto_mibici.config import (
    MODELS_DIR,
    P3_DURACION_MAX_MIN,
    P3_MIN_VIAJES_ESTACION,
    P3_TEST_FIN,
    P3_TEST_INICIO,
    P3_VARIABLES,
    STATIONS_PROCESSED_PATH,
    TRIPS_PROCESSED_PATH,
)
from proyecto_mibici.features import construir_perfil_estaciones, limpiar_viajes_p3
from proyecto_mibici.modeling.model_p3 import BaselineMunicipioP3, ClusterKMeansP3

app = typer.Typer(help="CLI para generar inferencia sobre clusters funcionales (P3).")


def cargar_pipeline_p3(ruta_modelo: Optional[Path] = None) -> ClusterKMeansP3:
    """Carga la instancia serializada de ClusterKMeansP3 desde models/."""
    directorio = MODELS_DIR
    if ruta_modelo:
        ruta = Path(ruta_modelo)
    else:
        ruta = (
            directorio / "p3_kmeans_model.pkl"
            if (directorio / "p3_kmeans_model.pkl").exists()
            else directorio / "p3_kmeans_pipeline.pkl"
        )

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el pipeline de clustering en {ruta}. Ejecuta primero train_p3.py."
        )
    return ClusterKMeansP3.load(ruta)


def predecir_todos_los_modelos(
    df: pd.DataFrame,
    df_stations: Optional[pd.DataFrame] = None,
    ruta_modelo: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Genera asignaciones de grupo para el baseline municipal y KMeans K=4."""
    preds = {}

    # Asegurar catálogo de estaciones para el baseline geográfico
    if df_stations is None and STATIONS_PROCESSED_PATH.exists():
        df_stations = pd.read_parquet(STATIONS_PROCESSED_PATH)

    # 1. Baseline por Municipio
    col_id = "origen_id" if "origen_id" in df.columns else "id"
    if df_stations is not None and col_id in df.columns:
        mun_map = df_stations.set_index("id")["municipality_code"].to_dict()
        mun_series = df[col_id].map(mun_map)
        preds["baseline_municipio"] = (
            mun_series.astype("category").cat.codes.to_numpy()
        )
    elif "municipality_code" in df.columns:
        preds["baseline_municipio"] = (
            df["municipality_code"].astype("category").cat.codes.to_numpy()
        )
    else:
        logger.warning("No se encontró catálogo ni columna 'municipality_code' para baseline.")
        preds["baseline_municipio"] = np.zeros(len(df), dtype=int)

    # 2. Modelo KMeans K=4 Candidato
    pipeline = cargar_pipeline_p3(ruta_modelo)
    cols_faltantes = set(P3_VARIABLES) - set(df.columns)
    if cols_faltantes:
        raise ValueError(
            f"El archivo de entrada no contiene las características requeridas por P3: {cols_faltantes}."
        )

    grupos_k = pipeline.predict(df[P3_VARIABLES])
    preds["grupo_kmeans_k4"] = grupos_k
    preds["grupo"] = grupos_k  # Compatibilidad

    return preds


def generar_reporte_inferencia(
    df: pd.DataFrame,
    df_stations: Optional[pd.DataFrame] = None,
    ruta_modelo: Optional[Path] = None,
) -> pd.DataFrame:
    """Retorna una copia del DataFrame anexando los grupos asignados."""
    df_out = df.copy()
    dict_preds = predecir_todos_los_modelos(
        df, df_stations=df_stations, ruta_modelo=ruta_modelo
    )
    for col_name, valores in dict_preds.items():
        df_out[col_name] = valores
    return df_out


@app.command()
def main(
    input_path: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="Ruta al archivo Parquet con perfiles de estaciones. Si se omite, usa la muestra de prueba.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Ruta para guardar el archivo Parquet con clusters asignados.",
    ),
):
    """Ejecuta inferencia de clustering de estaciones (P3) desde la terminal en Parquet."""
    try:
        df_stations = pd.read_parquet(STATIONS_PROCESSED_PATH)

        if input_path is None:
            logger.info("No se proporcionó --input. Construyendo perfiles de prueba desde Parquet...")
            cols_viajes = ["viaje_id", "inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"]
            df_trips = pd.read_parquet(TRIPS_PROCESSED_PATH, columns=cols_viajes)
            df_clean = limpiar_viajes_p3(df_trips, df_stations, duracion_max_min=P3_DURACION_MAX_MIN)
            df_test_p3 = df_clean[
                (df_clean["inicio_del_viaje"] >= P3_TEST_INICIO)
                & (df_clean["inicio_del_viaje"] < P3_TEST_FIN)
            ]
            perfil = construir_perfil_estaciones(df_test_p3, min_viajes=P3_MIN_VIAJES_ESTACION)
            df = perfil.reset_index().rename(columns={"index": "origen_id"})
        else:
            input_path = Path(input_path)
            logger.info(f"Cargando perfiles Parquet desde {input_path}...")
            df = pd.read_parquet(input_path)

        logger.info("Asignando clusters a estaciones...")
        resultado = generar_reporte_inferencia(df, df_stations=df_stations)

        distribucion = resultado["grupo"].value_counts().to_dict()
        logger.success(
            f"Inferencia P3 completada: {len(resultado)} estaciones asignadas | Distribución: {distribucion}"
        )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.suffix == ".parquet":
                resultado.to_parquet(output_path, index=False)
                logger.info(f"Resultados guardados exitosamente en Parquet: {output_path}")
            else:
                # pipeline.py solicita .csv: guardamos en CSV para su lectura inmediata
                resultado.to_csv(output_path, index=False)
                # y guardamos el .parquet para mantener la persistencia oficial
                parquet_path = output_path.with_suffix(".parquet")
                resultado.to_parquet(parquet_path, index=False)
                logger.info(f"Resultados guardados en CSV: {output_path} y Parquet: {parquet_path}")
        else:
            cols_mostrar = [
                c
                for c in ["origen_id", "id", "baseline_municipio", "grupo", "viajes_dia_log"]
                if c in resultado.columns
            ]
            if not cols_mostrar:
                cols_mostrar = resultado.columns[:4]
            logger.info("\n" + resultado[cols_mostrar].head(10).to_string(index=False))

        return resultado

    except Exception as e:
        logger.exception(f"Error en inferencia P3: {e}")
        raise


if __name__ == "__main__":
    app()