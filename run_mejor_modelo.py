#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Registro de los mejores modelos seleccionados por pregunta en el MLflow Model Registry.

Lee el experimento de MLflow Tracking, identifica la corrida de origen de cada
mejor modelo (según el análisis en mejor_modelo.md), lo registra con un nombre
claro y consistente, y conserva su linaje: experimento, corrida de origen,
commit de Git y versión de datos. Cada modelo registrado recibe una descripción.

Nombres de modelo en el Registry:
  - mibici-equipo20-p1-demanda    (P1: LightGBM Regresión Tweedie)
  - mibici-equipo20-p2-riesgo     (P2: Regresión Logística sin peso, umbral 0.5)
  - mibici-equipo20-p3-grupos     (P3: KMeans funcional K=4)

Uso:
    python run_mejor_modelo.py
"""

from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger

from proyecto_mibici.config import MLFLOW_EXPERIMENT_NAME, URI_MLFLOW_TRACKING

# ---------------------------------------------------------------------------
# Definición de los modelos a registrar: (nombre_registry, pregunta, predicado)
# El predicado recibe (tags, params, metrics) y retorna True si el run es el
# origen del modelo elegido para esa pregunta.
# ---------------------------------------------------------------------------
MODELOS_A_REGISTRAR = [
    {
        "name": "mibici-equipo20-p1-demanda",
        "pregunta": "P1",
        "descripcion": (
            "Modelo de producción para P1 (demanda horaria de viajes por estación)." 
            "LightGBM Tweedie (objective=tweedie, tweedie_variance_power=1.8, lr=0.1,"
            "num_leaves=127, n_estimators=600) sobre features V2. Seleccionado por menor"
            "MAE de validación (0.816) en una búsqueda anidada; en prueba MAE 0.789 y WAPE" 
            "0.630 frente al baseline lineal con rolling (0.853 y 0.681)."
            "Subestima levemente la demanda (sesgo −2.1%)."
        ),
        "metrica_test": "test_mae",
        "predicado": lambda tags, params, metrics: (
            tags.get("question") == "P1"
            and (
                tags.get("version") == "lgbm_tweedie_tuned"
                or tags.get("version") == "lgbm_tweedie"
                or "tweedie" in str(params.get("modelo", "")).lower()
            )
        ),
    },
    {
        "name": "mibici-equipo20-p2-riesgo",
        "pregunta": "P2",
        "descripcion": (
            "Modelo de produccion para P2 (riesgo de desbalance de estacion). "
            "Regresion Logistica con class_weight=balanced y umbral de decision 0.5 "
            "sobre las variables de riesgo P2. Seleccionado por mejor Test F1 "
            "(0.833) y PR-AUC (0.842) frente a Regresion Logistica sin peso "
            "(class_weight=None) y al umbral sensible (0.3)."
        ),
        "metrica_test": "test_f1_score",
        "predicado": lambda tags, params, metrics: (
            tags.get("question") == "P2"
            and "logistic" in str(params.get("modelo", "")).lower()
            and str(params.get("class_weight", "")) in ("balanced", "None")
            and str(params.get("umbral", "")) == "0.5"
        ),
    },
    {
        "name": "mibici-equipo20-p2-riesgo-gbm",
        "pregunta": "P2",
        "descripcion": (
            "Modelo de produccion para P2 (riesgo de desbalance de estacion). "
            "HistGradientBoostingClassifier (Gradient Boosting) con early stopping, "
            "optimizado por PR-AUC en validacion via nested runs. "
            "Variables de riesgo P2. Seleccionado por mejor Test PR-AUC y F1 "
            "frente a Regresion Logistica."
        ),
        "metrica_test": "test_pr_auc",
        "predicado": lambda tags, params, metrics: (
            tags.get("question") == "P2"
            and "histgradientboosting" in str(params.get("modelo", "")).lower()
        ),
    },
    {
        "name": "mibici-equipo20-p3-grupos",
        "pregunta": "P3",
        "descripcion": (
            "Modelo de produccion para P3 (grupos funcionales de estaciones). "
            "KMeans con k=4 y n_init=10 sobre ocho variables de perfil escaladas con "
            "StandardScaler. k=4 obtuvo el mejor silhouette (0.183) y el menor "
            "Davies-Bouldin (1.570) en validacion (enero-abril 2026) dentro de la "
            "busqueda de k=2 a 8. En prueba (mayo-septiembre 2026) confirma la ventaja "
            "sobre el baseline municipal: silhouette 0.182 contra 0.019 y "
            "Davies-Bouldin 1.535 contra 3.712."
        ),
        "metrica_test": "test_silhouette",
        "predicado": lambda tags, params, metrics: (
            tags.get("question") == "P3"
            and tags.get("stage_of_work") == "candidate"
            and str(params.get("n_clusters", "")) == "4"
        ),
    },
]


def buscar_run_origen(client: MlflowClient, experimento_id: str, pregunta: str, predicado) -> Optional[dict]:
    """Devuelve el run mas reciente que cumple el predicado para la pregunta."""
    runs = client.search_runs([experimento_id])
    candidatos = []
    for r in runs:
        tags = r.data.tags
        if tags.get("question") != pregunta and tags.get("pregunta") != pregunta:
            continue
        if predicado(tags, r.data.params, r.data.metrics):
            candidatos.append(r)
    if not candidatos:
        return None
    candidatos.sort(key=lambda r: r.info.start_time, reverse=True)
    r = candidatos[0]
    return {
        "run_id": r.info.run_id,
        "run_name": r.info.run_name,
        "experimento": experimento_id,
        "commit": r.data.tags.get("git_commit", "no_disponible"),
        "datos_version": r.data.tags.get("datos_version", "no_disponible"),
        "metrica_test": r.data.metrics,
        "params": r.data.params,
        "tags": r.data.tags,
    }


def registrar_modelo(client: MlflowClient, entrada: dict, run_info: dict) -> None:
    """Registra el modelo en el Model Registry, asigna descripcion, tags de linaje y alias Production."""
    nombre = entrada["name"]
    run_id = run_info["run_id"]
    metrica_clave = entrada["metrica_test"]
    valor_metrica = run_info["metrica_test"].get(metrica_clave)

    model_uri = f"runs:/{run_id}/model"

    # Idempotencia: si ya existe una version registrada con este run_origen, reusarla
    version_existente = None
    try:
        versiones = client.search_model_versions(f"name='{nombre}'")
        for v in versiones:
            if v.tags.get("run_origen") == run_id:
                version_existente = int(v.version)
                break
    except Exception as e:
        logger.warning(f"No se pudieron consultar versiones existentes de {nombre}: {e}")

    if version_existente is not None:
        version = version_existente
        logger.info(f"{nombre} ya tiene version {version} para run_origen {run_id}; se reusa y actualiza.")
    else:
        logger.info(f"Registrando {nombre} desde run {run_id} ...")
        mv = mlflow.register_model(model_uri, nombre)
        version = mv.version

    # Linaje en tags de la version del modelo
    lineage_tags = {
        "experimento": MLFLOW_EXPERIMENT_NAME,
        "run_origen": run_id,
        "run_name": run_info["run_name"],
        "git_commit": run_info["commit"],
        "datos_version": run_info["datos_version"],
        "pregunta": entrada["pregunta"],
        "metrica_test": metrica_clave,
        "valor_metrica_test": str(round(valor_metrica, 4)) if valor_metrica is not None else "n/a",
        "modelo_produccion": "true",
        "etapa": "produccion",
    }
    for k, v in lineage_tags.items():
        client.set_model_version_tag(nombre, version, k, str(v))

    # Descripcion de la version
    desc_version = (
        f"{entrada['descripcion']}\n\n"
        f"Linaje: experimento={MLFLOW_EXPERIMENT_NAME} | run_origen={run_id} "
        f"| commit={run_info['commit']} | datos_version={run_info['datos_version']} "
        f"| {metrica_clave}={valor_metrica}"
    )
    client.update_model_version(nombre, version, description=desc_version)

    # Promocionar a Produccion (alias champion y stage Production)
    try:
        client.set_registered_model_alias(nombre, "champion", str(version))
    except Exception as e:
        logger.warning(f"No se pudo asignar alias champion: {e}")
    try:
        client.transition_model_version_stage(
            name=nombre, version=version, stage="Production"
        )
    except Exception as e:
        logger.warning(f"No se pudo transicionar a stage Production: {e}")

    logger.success(
        f"{nombre} version {version} registrado y promovido a Production/champion "
        f"(run_origen={run_id}, commit={run_info['commit']}, datos_version={run_info['datos_version']})"
    )


def main() -> None:
    mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
    client = MlflowClient()
    experimento = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experimento is None:
        raise SystemExit(f"Experimento '{MLFLOW_EXPERIMENT_NAME}' no encontrado en MLflow.")
    experimento_id = experimento.experiment_id
    logger.info(f"Experimento '{MLFLOW_EXPERIMENT_NAME}' (id={experimento_id})")

    for entrada in MODELOS_A_REGISTRAR:
        run_info = buscar_run_origen(client, experimento_id, entrada["pregunta"], entrada["predicado"])
        if run_info is None:
            logger.warning(f"No se encontro run origen para {entrada['name']} ({entrada['pregunta']}). Se omite.")
            continue
        logger.info(
            f"Run origen {entrada['name']}: {run_info['run_id']} | "
            f"commit={run_info['commit']} | datos_version={run_info['datos_version']}"
        )
        registrar_modelo(client, entrada, run_info)

    logger.info("Registro de mejores modelos completado.")


if __name__ == "__main__":
    main()
