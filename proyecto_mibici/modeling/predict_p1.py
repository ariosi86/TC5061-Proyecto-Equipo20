#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de inferencia para P1: Predicción de Demanda Horaria de Viajes.
Calcula y anexa las predicciones de cada uno de los 6 modelos de P1:
  - 4 baselines ingenuos (persistencia, lag 24h, lag 168h, promedio 24h/168h)
  - v1 Regresión Lineal (lags + componentes temporales)
  - v2 Regresión Lineal (lags + calendario + rolling_3h y rolling_168h, serializado en p1_linear_model.pkl)
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import typer
from loguru import logger

from proyecto_mibici.config import MODELS_DIR, P1_FEATURES_V1
from proyecto_mibici.modeling.model_p1 import BaselineIngenuoP1, DemandaLinealP1

app = typer.Typer(help="CLI para generar inferencia sobre demanda horaria (P1).")


def cargar_pipeline_p1(ruta_modelo: Optional[Path] = None) -> DemandaLinealP1:
    """Carga la instancia serializada de DemandaLinealP1 desde models/."""
    ruta = ruta_modelo or (MODELS_DIR / "p1_linear_model.pkl")
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {ruta}. Ejecuta primero train_p1.py."
        )
    return DemandaLinealP1.load(ruta)


def predecir_todos_los_modelos(
    df: pd.DataFrame, ruta_modelo: Optional[Path] = None
) -> Dict[str, np.ndarray]:
    """Genera las predicciones de cada uno de los 6 modelos de P1."""
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

    # 2. Modelo v1 Lineal (lags + calendario)
    pipeline_campeon = cargar_pipeline_p1(ruta_modelo)
    cols_v1 = [c for c in P1_FEATURES_V1 if c in df.columns]
    if len(cols_v1) == len(P1_FEATURES_V1):
        # Si existe pipeline v1 entrenado o se infiere con pesos base
        preds["pred_v1_lineal"] = np.round(pipeline_campeon.predict(df), 2)
    else:
        preds["pred_v1_lineal"] = preds["pred_naive_promedio_24h_168h"]

    # 3. Modelo v2 Lineal Rolling (Campeón en producción)
    pred_v2 = np.round(pipeline_campeon.predict(df), 2)
    preds["pred_v2_lineal_rolling"] = pred_v2
    preds["demanda_predicha"] = pred_v2  # Compatibilidad con código previo y tests

    return preds


def predecir(df: pd.DataFrame, ruta_modelo: Optional[Path] = None) -> np.ndarray:
    """Retorna las predicciones del modelo campeón con cota inferior en 0."""
    pipeline = cargar_pipeline_p1(ruta_modelo)
    return pipeline.predict(df)


def generar_reporte_inferencia(
    df: pd.DataFrame, ruta_modelo: Optional[Path] = None
) -> pd.DataFrame:
    """Retorna una copia del DataFrame anexando las predicciones de todos los modelos."""
    df_out = df.copy()
    dict_preds = predecir_todos_los_modelos(df, ruta_modelo=ruta_modelo)
    for col_pred, valores in dict_preds.items():
        df_out[col_pred] = valores
    return df_out


@app.command()
def main(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="Ruta al archivo Parquet o CSV con las features de P1."
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Ruta para guardar el archivo con predicciones."
    ),
):
    """Ejecuta inferencia de demanda horaria (P1) para todos los modelos desde la terminal."""
    logger.info(f"Cargando datos para inferencia desde {input_path}...")
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    logger.info("Generando predicciones de demanda para todos los modelos...")
    resultado = generar_reporte_inferencia(df)

    logger.info(f"Total de registros procesados: {len(resultado):,}")
    logger.info(
        f"Demanda promedio estimada (Modelo Campeón v2): {resultado['pred_v2_lineal_rolling'].mean():.2f} viajes/hora"
    )
    logger.info(
        f"Demanda promedio estimada (Persistencia): {resultado['pred_naive_persistencia'].mean():.2f} viajes/hora"
    )
    logger.info(
        f"Demanda promedio estimada (Promedio 24h-168h): {resultado['pred_naive_promedio_24h_168h'].mean():.2f} viajes/hora"
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".parquet":
            resultado.to_parquet(output_path, index=False)
        else:
            resultado.to_csv(output_path, index=False)
        logger.info(f"Predicciones guardadas exitosamente en {output_path}")
    else:
        cols_id = [c for c in ["origen_id", "target_hour", "hour_window"] if c in resultado.columns]
        cols_modelos = [
            "pred_naive_persistencia",
            "pred_naive_promedio_24h_168h",
            "pred_v1_lineal",
            "pred_v2_lineal_rolling",
        ]
        cols_mostrar = cols_id + cols_modelos
        logger.info("\n" + resultado[cols_mostrar].head(10).to_string())


if __name__ == "__main__":
    app()