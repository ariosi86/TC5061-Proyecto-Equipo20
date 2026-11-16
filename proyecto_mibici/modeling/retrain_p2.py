#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Versión alternativa de P2 (Entregable 8): GBM reentrenado con datos recientes.

Mismos hiperparámetros y variables que el champion (P2_GBM_PARAMS, P2_VARIABLES).
Lo único que cambia es el periodo de entrenamiento:
  champion (v3): horas objetivo de 2025
  challenger:    horas objetivo de enero 2025 a junio 2026
Julio a septiembre 2026 queda fuera: es el conjunto del A/B testing.

Se registra como nueva versión de mibici-equipo20-p2-riesgo con alias
'challenger', en el experimento de monitoreo. No modifica el alias 'champion'.
"""

import time

import mlflow
import pandas as pd
import typer
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature

from proyecto_mibici.config import (
    P2_GBM_PARAMS,
    P2_OBJETIVO,
    P2_VARIABLES,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.modeling.model_p2 import RiesgoGBMP2
from proyecto_mibici.modeling.train_p2 import extraer_metricas_clasificacion
from proyecto_mibici.monitoring.datos_p2 import (
    FIN_ACTUAL,
    INICIO_ACTUAL,
    construir_serie,
)

NOMBRE_MODELO = "mibici-equipo20-p2-riesgo"
EXPERIMENTO = "mibici-equipo20-p2-monitoreo"
ALIAS = "challenger"
VARIABLES = list(P2_VARIABLES)

# Árboles del GBM: skops pide declararlos como confiables. Es seguro porque el
# modelo se entrena en este mismo script; no se carga de una fuente externa.
TIPOS_CONFIABLES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]

app = typer.Typer(help="Reentrenamiento de P2 para el A/B testing.")


@app.command()
def main(
    registrar: bool = typer.Option(
        True, "--registrar/--sin-registrar", help="Registrar en el Model Registry."
    ),
):
    # 1. Datos: todo lo anterior al periodo actual
    serie = construir_serie(fin_periodo=FIN_ACTUAL)
    corte = pd.Timestamp(INICIO_ACTUAL)
    df_train = serie[serie["hora_objetivo"] < corte]
    X = df_train[VARIABLES]
    y = df_train[P2_OBJETIVO].astype(bool)

    desde = df_train["hora_objetivo"].min()
    hasta = df_train["hora_objetivo"].max()
    estaciones = df_train["estacion_id"].nunique()
    logger.info(f"Entrenamiento: {len(df_train):,} filas | {estaciones} estaciones")
    logger.info(f"Periodo: {desde} a {hasta} | tasa de riesgo {y.mean():.4f}")

    # 2. Mismo modelo y mismos hiperparámetros que el champion
    inicio = time.time()
    modelo = RiesgoGBMP2(features=P2_VARIABLES, **P2_GBM_PARAMS)
    modelo.fit(X, y)
    segundos = time.time() - inicio
    estimador = getattr(modelo, "estimator_", modelo)

    pred = modelo.predict(X)
    prob = modelo.predict_proba(X)[:, 1]
    m = extraer_metricas_clasificacion(y, pred, prob, "Train")
    metricas = {
        f"train_{k}": float(v)
        for k, v in m.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    logger.info(f"Métricas de entrenamiento: {metricas}")

    # 3. MLflow: corrida en el experimento de monitoreo
    mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name="p2-gbm-reentrenado") as run:
        mlflow.set_tags({
            "question": "P2",
            "version": "v2_gbm_reentrenado",
            "model_family": "hist_gradient_boosting",
            "stage_of_work": "challenger",
        })
        mlflow.log_params({
            **P2_GBM_PARAMS,
            "modelo": "HistGradientBoostingClassifier",
            "variables_predictoras": ", ".join(VARIABLES),
            "entrenamiento_inicio": str(desde),
            "entrenamiento_fin": str(hasta),
            "filas_entrenamiento": len(df_train),
            "estaciones_entrenamiento": estaciones,
        })
        mlflow.log_metrics({**metricas, "train_seconds": segundos})

        ejemplo = X.head(5)
        firma = infer_signature(ejemplo, estimador.predict(ejemplo))
        info = mlflow.sklearn.log_model(
            estimador,
            name="model",
            signature=firma,
            input_example=ejemplo,
            skops_trusted_types=TIPOS_CONFIABLES,
        )
        run_id = run.info.run_id

    # 4. Registry: nueva versión con alias challenger
    if not registrar:
        logger.info("Sin registrar (--sin-registrar).")
        return

    version = mlflow.register_model(info.model_uri, NOMBRE_MODELO).version
    cliente = MlflowClient()
    cliente.set_registered_model_alias(NOMBRE_MODELO, ALIAS, version)
    cliente.update_model_version(
        NOMBRE_MODELO,
        version,
        description=(
            "Challenger P2: mismo GBM que el champion, reentrenado con horas "
            f"objetivo de {desde:%Y-%m-%d} a {hasta:%Y-%m-%d}. Para A/B testing."
        ),
    )
    cliente.set_model_version_tag(NOMBRE_MODELO, version, "run_origen", run_id)

    alias = cliente.get_registered_model(NOMBRE_MODELO).aliases
    logger.success(f"{NOMBRE_MODELO} versión {version} registrada como '{ALIAS}'")
    logger.info(f"Alias actuales: {alias}")


if __name__ == "__main__":
    app()