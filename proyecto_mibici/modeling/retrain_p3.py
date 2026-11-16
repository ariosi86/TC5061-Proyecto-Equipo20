#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Versión alternativa de P3 (Entregable 8): KMeans reentrenado con datos recientes.

Mismo algoritmo e hiperparámetros que el champion (StandardScaler + KMeans, k=4,
n_init=10, semilla 42) y las mismas ocho variables. Lo único que cambia es el
periodo con el que se construyen los perfiles:
  champion:    viajes de 2025 (doce meses)
  challenger:  viajes de octubre 2025 a septiembre 2026 (los doce meses previos)
Se conserva la misma duración para que ambos perfiles promedien un año completo
de temporadas. Octubre 2026 queda fuera porque es el conjunto del A/B testing.

Se registra como nueva versión de mibici-equipo20-p3-grupos con alias
'challenger', en el experimento de monitoreo de P3. El alias 'champion' se
queda en la versión vigente. El modelo se guarda con el mismo envoltorio pyfunc
que el champion, así la API puede cargarlo sin cambios si se promueve.
"""

from loguru import logger
import mlflow
from mlflow import MlflowClient
import pandas as pd
from sklearn.metrics import calinski_harabasz_score
import typer

from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_MONITOREO_P3,
    P3_K_ELEGIDO,
    P3_N_INIT_DEFAULT,
    P3_VARIABLES,
    RANDOM_SEED,
    STATIONS_PROCESSED_PATH,
    TRIPS_PROCESSED_PATH,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.modeling.model_p3 import ClusterKMeansP3
from proyecto_mibici.modeling.tracking import MLFlowTracker
from proyecto_mibici.modeling.train_p3 import evaluar_metricas_clustering
from proyecto_mibici.monitoring.datos_p3 import PERIODOS_P3, cargar_conjunto
from proyecto_mibici.plots import graficar_mapa_grupos

NOMBRE_MODELO = "mibici-equipo20-p3-grupos"
ALIAS = "challenger"

app = typer.Typer(help="Reentrenamiento de P3 para el A/B testing.")


@app.command()
def main(
    registrar: bool = typer.Option(True, "--registrar/--sin-registrar", help="Registrar en el Model Registry."),
):
    # 1. Datos: perfiles de los doce meses previos al periodo actual
    perfil = cargar_conjunto("reciente")
    inicio, fin = PERIODOS_P3["reciente"]
    logger.info(f"Entrenamiento: {len(perfil)} estaciones con perfil de [{inicio}, {fin})")

    # 2. Mismo modelo y mismos hiperparámetros que el champion
    modelo = ClusterKMeansP3(
        n_clusters=P3_K_ELEGIDO, n_init=P3_N_INIT_DEFAULT, random_state=RANDOM_SEED, features=P3_VARIABLES
    )
    modelo.fit(perfil[P3_VARIABLES])
    etiquetas = modelo.labels_

    X_escalado = modelo.pipeline_.named_steps["scaler"].transform(perfil[P3_VARIABLES])
    m_train = evaluar_metricas_clustering(X_escalado, etiquetas, "Train")
    m_train["calinski_harabasz"] = float(calinski_harabasz_score(X_escalado, etiquetas))

    grupos = perfil[P3_VARIABLES].assign(grupo=etiquetas)
    resumen_grupos = grupos.groupby("grupo").mean().round(3)
    resumen_grupos["n_estaciones"] = grupos["grupo"].value_counts().sort_index()
    logger.info(f"Métricas de entrenamiento: {m_train}")
    logger.info("\n" + resumen_grupos.to_string())

    # 3. MLflow: corrida en el experimento de monitoreo, con el envoltorio del champion
    tracker = MLFlowTracker(tracking_uri=URI_MLFLOW_TRACKING, experiment_name=MLFLOW_EXPERIMENT_MONITOREO_P3)
    figura = graficar_mapa_grupos(
        perfil, etiquetas, pd.read_parquet(STATIONS_PROCESSED_PATH),
        f"Grupos de estaciones MiBici (KMeans K={P3_K_ELEGIDO}, oct 2025 a sep 2026)",
    )
    run_id = tracker.registrar_run(
        run_name="p3-kmeans-reentrenado",
        params={
            "algoritmo": "KMeans",
            "n_clusters": P3_K_ELEGIDO,
            "n_init": P3_N_INIT_DEFAULT,
            "random_state": RANDOM_SEED,
            "variables": ", ".join(P3_VARIABLES),
            "entrenamiento_inicio": inicio,
            "entrenamiento_fin": fin,
            "estaciones_entrenamiento": len(perfil),
        },
        tags={
            "question": "P3",
            "version": "v2_kmeans_k4_reentrenado",
            "model_family": "kmeans",
            "stage_of_work": "challenger",
        },
        tabla=pd.DataFrame([m_train]),
        tabla_est=resumen_grupos.reset_index(),
        modelo=modelo.pipeline_,
        X_ejemplo=perfil[P3_VARIABLES].head(5),
        figuras={"mapa_p3_challenger": figura},
        df_train=perfil[P3_VARIABLES],
        ruta_entrenamiento=str(TRIPS_PROCESSED_PATH),
    )

    # 4. Registry: nueva versión con alias challenger
    if not registrar:
        logger.info("Sin registrar (--sin-registrar).")
        return

    version = mlflow.register_model(f"runs:/{run_id}/model", NOMBRE_MODELO).version
    cliente = MlflowClient()
    cliente.set_registered_model_alias(NOMBRE_MODELO, ALIAS, version)
    cliente.update_model_version(
        NOMBRE_MODELO,
        version,
        description=(
            f"Challenger P3: mismo KMeans k={P3_K_ELEGIDO} que el champion, con perfiles de "
            f"estaciones de {inicio} a {fin} (sin incluir). Para A/B testing."
        ),
    )
    cliente.set_model_version_tag(NOMBRE_MODELO, version, "run_origen", run_id)
    cliente.set_model_version_tag(NOMBRE_MODELO, version, "pregunta", "P3")

    alias = cliente.get_registered_model(NOMBRE_MODELO).aliases
    logger.success(f"{NOMBRE_MODELO} versión {version} registrada como '{ALIAS}'")
    logger.info(f"Alias actuales: {alias}")


if __name__ == "__main__":
    app()
