#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de inferencia para P1: Predicción de Demanda Horaria de Viajes.
Calcula y anexa las predicciones de los 6 modelos de P1 en formato Parquet:
  - 4 baselines ingenuos (persistencia, lag 24h, lag 168h, promedio 24h/168h)
  - v1 Regresión Lineal (lags + variables de calendario) cargado desde p1_v1_model.pkl
  - v2 Regresión Lineal (v1 + rolling means) cargado desde p1_v2_model.pkl
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import typer
from loguru import logger

from proyecto_mibici.config import MODELS_DIR, P1_FEATURES_V1, P1_FEATURES_V2
from proyecto_mibici.modeling.model_p1 import BaselineIngenuoP1, DemandaLinealP1

app = typer.Typer(help="CLI para generar inferencia sobre los 6 modelos de demanda horaria (P1).")


def _resolver_directorio_modelos(models_dir: Optional[Union[Path, str]]) -> Path:
    """Resuelve de forma segura el directorio de modelos, protegiendo contra OptionInfo de Typer."""
    if isinstance(models_dir, (str, Path)):
        return Path(models_dir)
    return Path(MODELS_DIR)


def cargar_modelo(nombre_archivo: str, ruta_directorio: Optional[Union[Path, str]] = None) -> DemandaLinealP1:
    """Carga un modelo entrenado específico desde el directorio de modelos."""
    directorio = _resolver_directorio_modelos(ruta_directorio)
    ruta = directorio / nombre_archivo
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {ruta}. Ejecuta primero train_p1.py."
        )
    return DemandaLinealP1.load(ruta)


def predecir_todos_los_modelos(
    df: pd.DataFrame, models_dir: Optional[Union[Path, str]] = None
) -> Dict[str, np.ndarray]:
    """Genera las predicciones independientes de los 6 modelos de P1."""
    directorio = _resolver_directorio_modelos(models_dir)
    preds = {}

    # 1. Cuatro baselines ingenuos
    estrategias = [
        ("pred_naive_persistencia", "persistencia"),
        ("pred_naive_lag_24h", "lag_24h"),
        ("pred_naive_lag_168h", "lag_168h"),
        ("pred_naive_promedio_24h_168h", "promedio_24h_168h"),
    ]
    for col_name, estrategia in estrategias:
        modelo_ing = BaselineIngenuoP1(estrategia=estrategia)
        preds[col_name] = np.round(modelo_ing.predict(df), 2)

    # 2. Modelo v1 Lineal (lags + componentes temporales)
    modelo_v1 = cargar_modelo("p1_v1_model.pkl", directorio)
    preds["pred_v1_lineal"] = np.round(modelo_v1.predict(df[P1_FEATURES_V1]), 2)

    # 3. Modelo v2 Lineal Rolling (lags + calendario + rolling means)
    archivo_v2 = "p1_v2_model.pkl" if (directorio / "p1_v2_model.pkl").exists() else "p1_linear_model.pkl"
    modelo_v2 = cargar_modelo(archivo_v2, directorio)
    pred_v2 = np.round(modelo_v2.predict(df[P1_FEATURES_V2]), 2)
    preds["pred_v2_lineal_rolling"] = pred_v2

    # =========================================================================
    # ALIAS CRÍTICO: pipeline.py espera explícitamente 'demanda_predicha'
    # =========================================================================
    preds["demanda_predicha"] = pred_v2

    return preds

def generar_reporte_inferencia(
    df: pd.DataFrame, models_dir: Optional[Union[Path, str]] = None
) -> pd.DataFrame:
    """Retorna una copia del DataFrame original con las columnas de predicción de los 6 modelos."""
    df_out = df.copy()
    dict_preds = predecir_todos_los_modelos(df, models_dir=models_dir)
    for col_pred, valores in dict_preds.items():
        df_out[col_pred] = valores
    return df_out


@app.command()
def main(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="Ruta al archivo Parquet con las features de P1."
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Ruta para guardar el archivo Parquet con predicciones."
    ),
    models_dir: Optional[Path] = typer.Option(
        None, "--models-dir", "-m", help="Directorio opcional donde residen los artefactos .pkl."
    ),
):
    """Ejecuta inferencia de demanda horaria para los 6 modelos desde terminal."""
    # Desempaquetar si viene como parámetro de Typer al llamarlo directamente
    input_p = Path(input_path) if isinstance(input_path, (str, Path)) else input_path
    output_p = Path(output_path) if isinstance(output_path, (str, Path)) else None
    dir_modelos = _resolver_directorio_modelos(models_dir)

    logger.info(f"Cargando datos para inferencia desde {input_p}...")
    df = pd.read_parquet(input_p)

    logger.info("Generando predicciones para los 6 modelos...")
    resultado = generar_reporte_inferencia(df, models_dir=dir_modelos)

    logger.info(f"Total de registros procesados: {len(resultado):,}")
    logger.info("Demanda horaria promedio estimada por modelo:")
    logger.info(f" - Persistencia:            {resultado['pred_naive_persistencia'].mean():.2f}")
    logger.info(f" - Lag 24h:                 {resultado['pred_naive_lag_24h'].mean():.2f}")
    logger.info(f" - Lag 168h:                {resultado['pred_naive_lag_168h'].mean():.2f}")
    logger.info(f" - Promedio 24h-168h:       {resultado['pred_naive_promedio_24h_168h'].mean():.2f}")
    logger.info(f" - v1 Regresión Lineal:     {resultado['pred_v1_lineal'].mean():.2f}")
    logger.info(f" - v2 Lineal + Rolling:     {resultado['pred_v2_lineal_rolling'].mean():.2f}")

    if output_p:
        output_p.parent.mkdir(parents=True, exist_ok=True)
        if output_p.suffix == ".parquet":
            resultado.to_parquet(output_p, index=False)
        else:
            # Compatibilidad si pipeline.py pasa una ruta con .csv u otra extensión
            resultado.to_parquet(output_p.with_suffix(".parquet"), index=False)
            resultado.to_csv(output_p, index=False)
        logger.info(f"Predicciones guardadas exitosamente en {output_p}")

    else:
        cols_id = [c for c in ["origen_id", "target_hour", "hour_window"] if c in resultado.columns]
        cols_modelos = [
            "pred_naive_persistencia",
            "pred_naive_lag_24h",
            "pred_naive_lag_168h",
            "pred_naive_promedio_24h_168h",
            "pred_v1_lineal",
            "pred_v2_lineal_rolling",
        ]
        logger.info("\n" + resultado[cols_id + cols_modelos].head(10).to_string())

    return resultado


if __name__ == "__main__":
    app()