#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de entrenamiento, evaluación y registro en MLflow para P3 (Segmentación de Estaciones).

Modos (opción --modelo):
  - baseline:      agrupación geográfica por municipio (línea base).
  - kmeans:        candidato KMeans con StandardScaler para el n_clusters indicado.
  - kmeans_search: búsqueda de k de 2 a 8 organizada como nested runs; elige k con
                   el silhouette de validación y no usa el conjunto de prueba.
  - ambos:         baseline + kmeans en una sola ejecución.

Partición temporal: entrenamiento 2025, validación enero-abril 2026, prueba mayo-septiembre 2026.
Métricas: silhouette y Davies-Bouldin por conjunto.
Exporta métricas en reports/p3_comparacion.parquet y asignaciones en reports/p3_predicciones_modelos.parquet.
Serializa el modelo KMeans en models/p3_kmeans_model.pkl.
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    P3_DURACION_MAX_MIN,
    P3_GRID_KMEANS,
    P3_K_ELEGIDO,
    P3_MIN_VIAJES_ESTACION,
    P3_N_INIT_DEFAULT,
    P3_TEST_FIN,
    P3_TEST_INICIO,
    P3_TRAIN_FIN,
    P3_TRAIN_INICIO,
    P3_VAL_FIN,
    P3_VAL_INICIO,
    P3_VARIABLES,
    RANDOM_SEED,
    REPORTS_DIR,
    STATIONS_PROCESSED_PATH,
    TRIPS_PROCESSED_PATH,
    TRIPS_TRAIN_PATH,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.features import construir_perfil_estaciones, limpiar_viajes_p3
from proyecto_mibici.modeling.model_p3 import BaselineMunicipioP3, ClusterKMeansP3
from proyecto_mibici.modeling.tracking import MLFlowTracker
from proyecto_mibici.plots import graficar_mapa_grupos

app = typer.Typer()


def evaluar_metricas_clustering(X: np.ndarray, etiquetas: np.ndarray, conjunto_nombre: str) -> dict:
    """Calcula Silhouette Score y Davies-Bouldin Index."""
    n_etiquetas = len(np.unique(etiquetas))
    if n_etiquetas <= 1 or n_etiquetas >= len(X):
        return {
            "conjunto": conjunto_nombre,
            "silhouette": 0.0,
            "davies_bouldin": 99.0,
        }
    return {
        "conjunto": conjunto_nombre,
        "silhouette": float(silhouette_score(X, etiquetas)),
        "davies_bouldin": float(davies_bouldin_score(X, etiquetas)),
    }


def imprimir_metricas_modelo(nombre: str, m_train: dict, m_test: dict):
    """Muestra en consola las métricas del modelo de clustering."""
    logger.info(
        f"[{nombre}] "
        f"Train -> Silhouette: {m_train['silhouette']:.3f} | Davies-Bouldin: {m_train['davies_bouldin']:.3f} || "
        f"Test -> Silhouette: {m_test['silhouette']:.3f} | Davies-Bouldin: {m_test['davies_bouldin']:.3f}"
    )


@app.command()
def main(
    modelo: str = typer.Option("ambos", "--modelo", help="Modelo a entrenar: 'baseline', 'kmeans', 'kmeans_search' o 'ambos'"),
    n_clusters: int = typer.Option(P3_K_ELEGIDO, "--n-clusters", help="Numero de clusters (P3)"),
    n_init: int = typer.Option(P3_N_INIT_DEFAULT, "--n-init", help="n_init de KMeans (P3)"),
):
    tracker = MLFlowTracker(
        tracking_uri=URI_MLFLOW_TRACKING,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )

    logger.info("Cargando y preparando perfiles funcionales para P3...")
    cols_viajes = ["viaje_id", "inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"]
    df_trips = pd.read_parquet(TRIPS_PROCESSED_PATH, columns=cols_viajes)
    df_stations = pd.read_parquet(STATIONS_PROCESSED_PATH)

    df_clean = limpiar_viajes_p3(df_trips, df_stations, duracion_max_min=P3_DURACION_MAX_MIN)

    # Filtrar particiones temporales
    mask_tr = (df_clean["inicio_del_viaje"] >= P3_TRAIN_INICIO) & (df_clean["inicio_del_viaje"] < P3_TRAIN_FIN)
    mask_ts = (df_clean["inicio_del_viaje"] >= P3_TEST_INICIO) & (df_clean["inicio_del_viaje"] < P3_TEST_FIN)
    mask_val = (df_clean["inicio_del_viaje"] >= P3_VAL_INICIO) & (df_clean["inicio_del_viaje"] < P3_VAL_FIN)
    perfil_train = construir_perfil_estaciones(df_clean[mask_tr], min_viajes=P3_MIN_VIAJES_ESTACION)
    perfil_test = construir_perfil_estaciones(df_clean[mask_ts], min_viajes=P3_MIN_VIAJES_ESTACION)
    perfil_val = construir_perfil_estaciones(df_clean[mask_val], min_viajes=P3_MIN_VIAJES_ESTACION)

    logger.info(f"Estaciones con perfil válido -> Train: {len(perfil_train)} | Test: {len(perfil_test)}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Escalado de referencia para evaluación geométrica unificada
    scaler_eval = StandardScaler()
    X_train_eval = scaler_eval.fit_transform(perfil_train[P3_VARIABLES])
    X_test_eval = scaler_eval.transform(perfil_test[P3_VARIABLES])
    X_val_eval = scaler_eval.transform(perfil_val[P3_VARIABLES])

    # Etiquetas del baseline por municipio
    municipio_serie = df_stations.set_index("id")["municipality_code"]
    etiquetas_mun_train = municipio_serie.reindex(perfil_train.index).astype("category").cat.codes.to_numpy()
    etiquetas_mun_test = municipio_serie.reindex(perfil_test.index).astype("category").cat.codes.to_numpy()
    etiquetas_mun_val = municipio_serie.reindex(perfil_val.index).astype("category").cat.codes.to_numpy()

    # Añadir columna temporal de referencia para tracking
    perfil_train_tracking = perfil_train.copy()
    perfil_test_tracking = perfil_test.copy()
    perfil_train_tracking["periodo_referencia"] = pd.Timestamp(P3_TRAIN_INICIO)
    perfil_test_tracking["periodo_referencia"] = pd.Timestamp(P3_TEST_INICIO)

    splits = {"Train": perfil_train_tracking, "Test": perfil_test_tracking}

    params_base, tags_base = tracker.armar_params_tags(
        "P3",
        splits,
        extra_params={
            "train_inicio": P3_TRAIN_INICIO,
            "train_fin": P3_TRAIN_FIN,
            "validation_inicio": P3_VAL_INICIO,
            "validation_fin": P3_VAL_FIN,
            "test_inicio": P3_TEST_INICIO,
            "test_fin": P3_TEST_FIN,
            "k_clusters": n_clusters,
            "variables": ", ".join(P3_VARIABLES),
            "min_viajes_estacion": P3_MIN_VIAJES_ESTACION,
            "duracion_max_min": P3_DURACION_MAX_MIN,
        },
        col_tiempo="periodo_referencia",
    )

    filas_comparacion = []

    # DataFrame consolidado de asignaciones sobre Test
    df_predicciones = pd.DataFrame({"origen_id": perfil_test.index})
    for col in P3_VARIABLES:
        if col in perfil_test.columns:
            df_predicciones[col] = perfil_test[col].values

    # -------------------------------------------------------------
    # 0. BÚSQUEDA DE K CON NESTED RUNS (selección en validación)
    # -------------------------------------------------------------
    if modelo.lower() == "kmeans_search":
        valores_k = P3_GRID_KMEANS["n_clusters"]
        logger.info(f"--- Búsqueda de K ({min(valores_k)} a {max(valores_k)}) con nested runs ---")
        tags_busqueda = {**tags_base, "question": "P3", "model_family": "kmeans", "stage_of_work": "search"}
        resultados = []

        with mlflow.start_run(run_name="p3-kmeans-search"):
            tracker.log_params_nuevos({
                **params_base,
                "algoritmo": "KMeans",
                "k_rango": f"{min(valores_k)}-{max(valores_k)}",
                "criterio": "validation_silhouette",
            })
            mlflow.set_tags(tags_busqueda)

            for k in valores_k:
                source_tags = tracker._get_source_tags()
                with mlflow.start_run(run_name=f"p3-search-k{k}", nested=True):
                    if source_tags:
                        mlflow.set_tags(source_tags)
                    modelo_k = ClusterKMeansP3(n_clusters=k, n_init=n_init, random_state=RANDOM_SEED)
                    modelo_k.fit(perfil_train[P3_VARIABLES])
                    m_tr = evaluar_metricas_clustering(X_train_eval, modelo_k.labels_, "Train")
                    pred_va = modelo_k.predict(perfil_val[P3_VARIABLES])
                    m_va = evaluar_metricas_clustering(X_val_eval, pred_va, "Validación")

                    mlflow.log_params({"n_clusters": k, "n_init": n_init})
                    mlflow.log_metrics(tracker.metricas_a_dict(pd.DataFrame([m_tr, m_va])))
                    mlflow.set_tags(tags_busqueda)
                    resultados.append({"k": k, "validation_silhouette": m_va["silhouette"]})

            mejor = max(resultados, key=lambda r: r["validation_silhouette"])
            mlflow.log_param("mejor_k", mejor["k"])
            mlflow.log_metric("mejor_validation_silhouette", mejor["validation_silhouette"])
            logger.success(f"Mejor K en validación: {mejor['k']} (silhouette {mejor['validation_silhouette']:.3f})")
        return

    # -------------------------------------------------------------
    # 1. EVALUACIÓN DE LÍNEA BASE GEOGRÁFICA (MUNICIPIO)
    # -------------------------------------------------------------
    if modelo.lower() in ("baseline", "municipio", "ambos"):
        logger.info("--- Evaluando Línea Base: Agrupación por Municipio ---")
        m_base_train = evaluar_metricas_clustering(X_train_eval, etiquetas_mun_train, "Train")
        m_base_test = evaluar_metricas_clustering(X_test_eval, etiquetas_mun_test, "Test")
        m_base_val = evaluar_metricas_clustering(X_val_eval, etiquetas_mun_val, "Validación")

        df_predicciones["baseline_municipio"] = etiquetas_mun_test
        imprimir_metricas_modelo("baseline_municipio", m_base_train, m_base_test)

        fig_mapa_base = graficar_mapa_grupos(
            perfil_train,
            etiquetas_mun_train,
            df_stations,
            "Agrupación por Municipio (Línea Base)",
        )

        tracker.registrar_run(
            run_name="p3-baseline-municipio",
            params={**params_base, "algoritmo": "BaselineMunicipio"},
            tags={
                **tags_base,
                "version": "baseline_municipio",
                "question": "P3",
                "model_family": "baseline_municipio",
                "stage_of_work": "baseline",
            },
            tabla=pd.DataFrame([m_base_train, m_base_val, m_base_test]),
            tabla_est=pd.DataFrame(),
            figuras={"mapa_municipio": fig_mapa_base},
            df_train=perfil_train,
            ruta_entrenamiento=str(TRIPS_TRAIN_PATH),
        )

        filas_comparacion.append({
            "modelo": "baseline_municipio",
            "train_silhouette": m_base_train["silhouette"],
            "train_davies_bouldin": m_base_train["davies_bouldin"],
            "test_silhouette": m_base_test["silhouette"],
            "test_davies_bouldin": m_base_test["davies_bouldin"],
        })

    # -------------------------------------------------------------
    # 2. MODELO CANDIDATO: K-MEANS FUNCIONAL
    # -------------------------------------------------------------
    if modelo.lower() in ("kmeans", "k-means", "ambos"):
        logger.info(f"--- Entrenando Modelo Candidato: KMeans (K={n_clusters}) ---")
        etiqueta = f"kmeans_k{n_clusters}"
        modelo_kmeans = ClusterKMeansP3(
            n_clusters=n_clusters, n_init=n_init, random_state=RANDOM_SEED
        )
        modelo_kmeans.fit(perfil_train[P3_VARIABLES])

        ruta_modelo = MODELS_DIR / "p3_kmeans_model.pkl"
        modelo_kmeans.save(ruta_modelo)
        modelo_kmeans.save(MODELS_DIR / "p3_kmeans_pipeline.pkl")
        logger.info(f"Pipeline P3 persistido exitosamente en {ruta_modelo}")

        pred_train_k = modelo_kmeans.labels_
        pred_test_k = modelo_kmeans.predict(perfil_test[P3_VARIABLES])
        pred_val_k = modelo_kmeans.predict(perfil_val[P3_VARIABLES])

        m_k_train = evaluar_metricas_clustering(X_train_eval, pred_train_k, "Train")
        m_k_test = evaluar_metricas_clustering(X_test_eval, pred_test_k, "Test")
        m_k_val = evaluar_metricas_clustering(X_val_eval, pred_val_k, "Validación")

        df_predicciones[f"grupo_{etiqueta}"] = pred_test_k
        imprimir_metricas_modelo(etiqueta, m_k_train, m_k_test)

        # Resumen descriptivo de los clusters
        perfil_grupos = perfil_train[P3_VARIABLES].assign(grupo=pred_train_k)
        resumen_grupos = perfil_grupos.groupby("grupo").mean().round(2)
        resumen_grupos["n_estaciones"] = perfil_grupos["grupo"].value_counts()

        # Mapa de clústeres
        fig_mapa_kmeans = graficar_mapa_grupos(
            perfil_train,
            pred_train_k,
            df_stations,
            f"Grupos de Estaciones MiBici (KMeans K={n_clusters})",
        )

        estimador_pipeline = getattr(modelo_kmeans, "pipeline_", getattr(modelo_kmeans, "pipeline", None))

        tracker.registrar_run(
            run_name=f"p3-kmeans-k{n_clusters}",
            params={**params_base, "algoritmo": "KMeans", "n_clusters": n_clusters, "n_init": n_init},
            tags={
                **tags_base,
                "version": "v1_kmeans",
                "question": "P3",
                "model_family": "kmeans",
                "stage_of_work": "candidate",
            },
            tabla=pd.DataFrame([m_k_train, m_k_val, m_k_test]),
            tabla_est=resumen_grupos.reset_index(),
            modelo=estimador_pipeline,
            X_ejemplo=perfil_train[P3_VARIABLES].head(5),
            figuras={f"mapa_{etiqueta}": fig_mapa_kmeans},
            df_train=perfil_train[P3_VARIABLES],
            ruta_entrenamiento=str(TRIPS_TRAIN_PATH),
        )

        filas_comparacion.append({
            "modelo": etiqueta,
            "train_silhouette": m_k_train["silhouette"],
            "train_davies_bouldin": m_k_train["davies_bouldin"],
            "test_silhouette": m_k_test["silhouette"],
            "test_davies_bouldin": m_k_test["davies_bouldin"],
        })

    # -------------------------------------------------------------
    # 3. GUARDAR Y MOSTRAR TABLA COMPARATIVA (SOLO PARQUET)
    # -------------------------------------------------------------
    df_p3_comp = pd.DataFrame(filas_comparacion).set_index("modelo")
    p3_comp_path = REPORTS_DIR / "p3_comparacion.parquet"
    df_p3_comp.round(3).reset_index().to_parquet(p3_comp_path, index=False)

    logger.info(f"Reporte de entrenamiento P3 guardado en Parquet: {p3_comp_path}")
    logger.info("\n" + df_p3_comp.round(3).to_string())

    # -------------------------------------------------------------
    # 4. GUARDAR Y MOSTRAR ASIGNACIONES DE CLÚSTERES (PARQUET)
    # -------------------------------------------------------------
    preds_out_path = REPORTS_DIR / "p3_predicciones_modelos.parquet"
    df_predicciones.to_parquet(preds_out_path, index=False)
    logger.success(f"Asignaciones consolidadas guardadas en: {preds_out_path}")


if __name__ == "__main__":
    app()