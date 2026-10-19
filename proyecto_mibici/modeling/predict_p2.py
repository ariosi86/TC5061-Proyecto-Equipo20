#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de inferencia para P2: Predicción de Riesgo de Desbalance.
Carga el Pipeline serializado (RiesgoLogisticoP2) y calcula:
  - Inferencia de Persistencia
  - Probabilidad continua de riesgo (prob_riesgo)
  - Clasificación binaria contra --umbral (pred_riesgo)
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import typer
from loguru import logger

from proyecto_mibici.config import MODELS_DIR, P2_UMBRAL, P2_VARIABLES
from proyecto_mibici.features import construir_dataset_p2
from proyecto_mibici.modeling.model_p2 import BaselinePersistenciaP2, RiesgoLogisticoP2

app = typer.Typer(help="CLI para generar inferencia sobre el riesgo de desbalance (P2).")


def cargar_pipeline_p2(ruta_modelo: Optional[Path] = None) -> RiesgoLogisticoP2:
    """Carga la instancia serializada de RiesgoLogisticoP2 desde models/."""
    ruta = ruta_modelo or (MODELS_DIR / "p2_logistic_model.pkl")
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {ruta}. Ejecuta primero train_p2.py."
        )
    return RiesgoLogisticoP2.load(ruta)


def predecir_todos_los_modelos(
    df: pd.DataFrame, ruta_modelo: Optional[Path] = None, umbral: float = P2_UMBRAL
) -> Dict[str, np.ndarray]:
    """Genera inferencias para la línea base y la regresión logística."""
    preds = {}

    # 1. Baseline ingenuo de persistencia (si existe la columna en_riesgo)
    if "en_riesgo" in df.columns:
        modelo_naive = BaselinePersistenciaP2(col_persistencia="en_riesgo")
        preds["pred_naive_persistencia"] = modelo_naive.predict(df).astype(int)

    # 2. Modelo Logístico Campeón
    pipeline = cargar_pipeline_p2(ruta_modelo)
    
    cols_faltantes = set(P2_VARIABLES) - set(df.columns)
    if cols_faltantes:
        raise ValueError(
            f"El archivo de entrada no contiene las características requeridas por P2: {cols_faltantes}."
        )

    prob = pipeline.predict_proba(df[P2_VARIABLES])[:, 1]
    preds["prob_riesgo"] = np.round(prob, 4)
    preds["pred_riesgo"] = (prob >= umbral).astype(int)

    return preds


def generar_reporte_inferencia(
    df: pd.DataFrame, ruta_modelo: Optional[Path] = None, umbral: float = P2_UMBRAL
) -> pd.DataFrame:
    """Retorna una copia del DataFrame anexando las columnas predictivas."""
    df_out = df.copy()
    dict_preds = predecir_todos_los_modelos(df, ruta_modelo=ruta_modelo, umbral=umbral)
    for col_name, valores in dict_preds.items():
        df_out[col_name] = valores
    return df_out


@app.command()
def main(
    input_path: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="Ruta al archivo Parquet o CSV con features de P2. Si se omite, usa una muestra de validación.",
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Ruta para guardar el archivo con predicciones."
    ),
    umbral: float = typer.Option(
        P2_UMBRAL, "--umbral", "-u", help="Umbral de decisión para clasificar riesgo (0.0 a 1.0)."
    ),
):
    """Ejecuta inferencia de riesgo de desbalance (P2) desde la terminal."""
    try:
        if input_path is None:
            logger.info("No se proporcionó --input. Construyendo muestra de prueba representativa...")
            df_p2 = construir_dataset_p2()
            df = df_p2.sample(n=min(200, len(df_p2)), random_state=42).copy()
        else:
            input_path = Path(input_path)
            logger.info(f"Cargando datos para inferencia desde {input_path}...")
            if input_path.suffix == ".parquet":
                df = pd.read_parquet(input_path)
            else:
                df = pd.read_csv(input_path)

        logger.info(f"Generando predicciones de riesgo con umbral {umbral:.2f}...")
        resultado = generar_reporte_inferencia(df, umbral=umbral)

        n_riesgo = int(resultado["pred_riesgo"].sum())
        total = len(resultado)
        pct = (n_riesgo / total * 100) if total > 0 else 0.0

        logger.success(
            f"Inferencia P2 completada: {n_riesgo}/{total} estaciones en riesgo ({pct:.1f}%)"
        )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.suffix == ".parquet":
                resultado.to_parquet(output_path, index=False)
            else:
                resultado.to_csv(output_path, index=False)
            logger.info(f"Predicciones guardadas exitosamente en {output_path}")
        else:
            cols_mostrar = [
                c for c in [
                    "estacion_id", "hora", "pred_naive_persistencia", "prob_riesgo", "pred_riesgo"
                ] if c in resultado.columns
            ]
            logger.info("\n" + resultado[cols_mostrar].head(10).to_string(index=False))

        return resultado

    except Exception as e:
        logger.exception(f"Error en inferencia P2: {e}")
        raise


if __name__ == "__main__":
    app()