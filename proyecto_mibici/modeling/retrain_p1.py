#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reentrenamiento de P1 (v2): mismo LightGBM del champion, con datos hasta sep 2026.
No modifica train_p1.py; registra una nueva versión del mismo modelo.
"""

import ast

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import typer
from loguru import logger
from mlflow import MlflowClient

from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_MONITOREO,
    P1_ACTUAL_FIN,
    P1_FEATURES_V2,
    P1_LGBM_GRID_TWEEDIE,
    P1_TARGET,
    P1_TEST_FIN,
    RANDOM_SEED,
    TRIPS_PROCESSED_PATH,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.features import construir_dataset_p1
from proyecto_mibici.modeling.tracking import MLFlowTracker
from proyecto_mibici.modeling.train_p1 import (
    calcular_metricas_por_estacion,
    evaluar_metricas_regresion,
)
from proyecto_mibici.plots import figuras_p1

MODEL_NAME = "mibici-equipo20-p1-demanda"
HP_KEYS = set(P1_LGBM_GRID_TWEEDIE) | {
    "objective", "tweedie_variance_power", "learning_rate", "num_leaves", "n_estimators",
}

app = typer.Typer(help="Reentrena P1 (v2) y la registra como challenger.")


def _convertir(valor: str):
    """MLflow guarda los params como texto; los regresa a su tipo."""
    try:
        return ast.literal_eval(valor)
    except (ValueError, SyntaxError):
        return valor


def hiperparametros_champion(client: MlflowClient):
    mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    run = client.get_run(mv.run_id)
    hp = {k: _convertir(v) for k, v in run.data.params.items() if k in HP_KEYS}
    return mv, hp


@app.command()
def main(
    registrar: bool = typer.Option(
        True, "--registrar/--no-registrar",
        help="--no-registrar entrena y loguea el run, pero no toca el Registry.",
    ),
):
    mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
    client = MlflowClient(tracking_uri=URI_MLFLOW_TRACKING)

    mv_champion, hp = hiperparametros_champion(client)
    logger.info(f"Champion v{mv_champion.version} -> hiperparámetros: {hp}")

    # 1. Datos: tabla de modelado completa, cortada por target_hour
    df_p1, _ = construir_dataset_p1()
    df_p1 = df_p1.dropna(subset=[P1_TARGET])
    t = df_p1["target_hour"]
    df_retrain = df_p1[t < pd.Timestamp(P1_TEST_FIN)].copy()
    df_oct = df_p1[(t >= pd.Timestamp(P1_TEST_FIN)) & (t < pd.Timestamp(P1_ACTUAL_FIN))]

    # 2. Verificaciones: octubre existe, está completo y no se traslapa
    if df_oct.empty:
        raise ValueError("No hay datos de octubre en la tabla de modelado.")
    assert df_retrain["target_hour"].max() < df_oct["target_hour"].min()  # nosec B101
    logger.info(
        f"Reentrenamiento: {len(df_retrain):,} filas "
        f"({df_retrain['target_hour'].min()} -> {df_retrain['target_hour'].max()})"
    )
    logger.info(
        f"Octubre (solo verificación): {len(df_oct):,} filas, "
        f"{df_oct['target_hour'].dt.normalize().nunique()} días distintos (esperado: 31)"
    )

    # 3. Entrenar: mismos hiperparámetros, solo cambian los datos
    X, y = df_retrain[P1_FEATURES_V2], df_retrain[P1_TARGET]
    modelo = lgb.LGBMRegressor(random_state=RANDOM_SEED, **hp)
    modelo.fit(X, y)
    pred = np.clip(modelo.predict(X), 0, None)
    m_train = evaluar_metricas_regresion(y, pred, "Train")

    # 4. Loguear el run (mismo patrón de llamada que train_p1)
    tracker = MLFlowTracker(
        tracking_uri=URI_MLFLOW_TRACKING, experiment_name=MLFLOW_EXPERIMENT_MONITOREO
    )
    params_base, tags_base = tracker.armar_params_tags(
        "P1",
        {"Train": df_retrain},
        extra_params={
            "variable_objetivo": P1_TARGET,
            "frecuencia": "horaria",
            "retrain_desde_champion_version": mv_champion.version,
            "train_fin_exclusivo": P1_TEST_FIN,
        },
        col_tiempo="target_hour",
    )
    tracker.registrar_run(
        run_name="P1_lgbm_tweedie_v2_retrain",
        params={**params_base, "modelo": f"LightGBM_{hp['objective']}",
                "variables": ", ".join(P1_FEATURES_V2), **hp},
        tags={**tags_base, "version": "v2_retrain", "stage_of_work": "retrain",
              "model_family": "lightgbm"},
        tabla=pd.DataFrame([m_train]),
        tabla_est=calcular_metricas_por_estacion(df_retrain, pred, P1_TARGET),
        modelo=modelo,
        X_ejemplo=X.head(5),
        figuras=figuras_p1(df_retrain, pred, P1_TARGET, "v2_retrain"),
        serialization_format="cloudpickle",
        df_train=df_retrain,
        ruta_entrenamiento=str(TRIPS_PROCESSED_PATH),
    )

    if not registrar:
        logger.info("--no-registrar: no se toca el Registry.")
        return

    # 5. Registrar como nueva versión; v1 conserva el alias champion
    ultimo = mlflow.last_active_run()
    if ultimo is None:
        raise RuntimeError("No se encontró el run que acaba de crearse.")
    run_id = ultimo.info.run_id

    modelos = [
        m for m in mlflow.search_logged_models(
            experiment_ids=[ultimo.info.experiment_id], output_format="list"
        )
        if m.source_run_id == run_id
    ]
    if not modelos:
        raise RuntimeError(
            f"El run {run_id} no tiene un modelo guardado; revisa el error de "
            "'No se pudo guardar el modelo en MLflow' en el log de arriba."
        )
    logged = max(modelos, key=lambda m: m.creation_timestamp)
    logger.info(f"Registrando el modelo {logged.model_id} del run {run_id}")

    nueva = mlflow.register_model(f"models:/{logged.model_id}", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "challenger", nueva.version)
    logger.success(
        f"Registrada v{nueva.version} con alias 'challenger'. "
        f"Champion sigue en v{mv_champion.version}."
    )

if __name__ == "__main__":
    app()