#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de inferencia para P3: Segmentación Funcional de Estaciones.
Carga el Pipeline serializado (ClusterKMeansP3) y asigna:
  - Grupo asignado por KMeans K=4 (columna 'grupo')
  - Grupo estimado por demarcación geográfica (columna 'baseline_municipio')
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
    ruta = ruta_modelo or (MODELS_DIR / "p3_kmeans_pipeline.pkl")
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el pipeline de clustering en {ruta}. Ejecuta primero train_p3.py."
        )
    return ClusterKMeansP3.load(ruta)


def predecir_todos_los_modelos(
    df: pd.DataFrame, ruta_modelo: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """Genera asignaciones de grupo para el baseline municipal y KMeans K=4."""
    preds = {}

    # 1. Baseline por Municipio
    modelo_base = BaselineMunicipioP3()
    modelo_base.fit(df)
    preds["baseline_municipio"] = modelo_base.predict(df)

    # 2. Modelo KMeans K=4 Campeón
    pipeline = cargar_pipeline_p3(ruta_modelo)
    cols_faltantes = set(P3_VARIABLES) - set(df.columns)
    if cols_faltantes:
        raise ValueError(
            f"El archivo de entrada no contiene las características requeridas por P3: {cols_faltantes}."
        )

    grupos_k = pipeline.predict(df[P3_VARIABLES])
    preds["grupo_kmeans_k4"] = grupos_k
    preds["grupo"] = grupos_k  # Compatibilidad con pipeline.py y contratos previos

    return preds


def generar_reporte_inferencia(
    df: pd.DataFrame, ruta_modelo: Optional[Path] = None
) -> pd.DataFrame:
    """Retorna una copia del DataFrame anexando los grupos asignados."""
    df_out = df.copy()
    dict_preds = predecir_todos_los_modelos(df, ruta_modelo=ruta_modelo)
    for col_name, valores in dict_preds.items():
        df_out[col_name] = valores
    return df_out


@app.command()
def main(
    input_path: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="Ruta al archivo con perfiles de estaciones. Si se omite, usa la muestra de prueba.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Ruta para guardar el archivo con clusters asignados (CSV o Parquet).",
    ),
):
    """Ejecuta inferencia de clustering de estaciones (P3) desde la terminal."""
    try:
        if input_path is None:
            logger.info("No se proporcionó --input. Construyendo perfiles de estaciones de prueba...")
            cols_viajes = ["viaje_id", "inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"]
            df_trips = pd.read_parquet(TRIPS_PROCESSED_PATH, columns=cols_viajes)
            df_stations = pd.read_parquet(STATIONS_PROCESSED_PATH)
            df_clean = limpiar_viajes_p3(df_trips, df_stations, duracion_max_min=P3_DURACION_MAX_MIN)
            df_test_p3 = df_clean[
                (df_clean["inicio_del_viaje"] >= P3_TEST_INICIO)
                & (df_clean["inicio_del_viaje"] < P3_TEST_FIN)
            ]
            perfil = construir_perfil_estaciones(df_test_p3, min_viajes=P3_MIN_VIAJES_ESTACION)
            df = perfil.reset_index()
        else:
            input_path = Path(input_path)
            logger.info(f"Cargando perfiles de estaciones desde {input_path}...")
            if input_path.suffix == ".parquet":
                df = pd.read_parquet(input_path)
            else:
                df = pd.read_csv(input_path)

        logger.info("Asignando clusters a estaciones...")
        resultado = generar_reporte_inferencia(df)

        distribucion = resultado["grupo"].value_counts().to_dict()
        logger.success(
            f"Inferencia P3 completada: {len(resultado)} estaciones asignadas | Distribución: {distribucion}"
        )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.suffix == ".parquet":
                resultado.to_parquet(output_path, index=False)
            else:
                resultado.to_csv(output_path, index=False)
            logger.info(f"Resultados guardados exitosamente en {output_path}")
        else:
            cols_mostrar = [
                c for c in ["origen_id", "id", "baseline_municipio", "grupo", "viajes_dia_log"]
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