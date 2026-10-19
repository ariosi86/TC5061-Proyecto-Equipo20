#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de entrenamiento, evaluación y registro en MLflow para P3 (Segmentación de Estaciones).
Compara e imprime en consola:
  - Baseline Geográfico por Municipio
  - Modelo Candidato: KMeans (K=4) con StandardScaler
Exporta métricas en reports/p3_comparacion.csv y asignaciones en reports/p3_predicciones_modelos.parquet.
Serializa el modelo oficial en models/p3_kmeans_model.pkl.
"""

from pathlib import Path
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
    P3_K_ELEGIDO,
    P3_MIN_VIAJES_ESTACION,
    P3_TEST_FIN,
    P3_TEST_INICIO,
    P3_TRAIN_FIN,
    P3_TRAIN_INICIO,
    P3_VARIABLES,
    RANDOM_SEED,
    REPORTS_DIR,
    STATIONS_PROCESSED_PATH,
    TRIPS_PROCESSED_PATH,
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
def main():
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

    perfil_train = construir_perfil_estaciones(df_clean[mask_tr], min_viajes=P3_MIN_VIAJES_ESTACION)
    perfil_test = construir_perfil_estaciones(df_clean[mask_ts], min_viajes=P3_MIN_VIAJES_ESTACION)

    logger.info(f"Estaciones con perfil válido -> Train: {len(perfil_train)} | Test: {len(perfil_test)}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Escalado de referencia para evaluación geométrica unificada
    scaler_eval = StandardScaler()
    X_train_eval = scaler_eval.fit_transform(perfil_train[P3_VARIABLES])
    X_test_eval = scaler_eval.transform(perfil_test[P3_VARIABLES])

    # Etiquetas del baseline por municipio
    municipio_serie = df_stations.set_index("id")["municipality_code"]
    etiquetas_mun_train = municipio_serie.reindex(perfil_train.index).astype("category").cat.codes.to_numpy()
    etiquetas_mun_test = municipio_serie.reindex(perfil_test.index).astype("category").cat.codes.to_numpy()

    # Añadir columna temporal de referencia para compatibilidad con tracker.armar_params_tags
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
            "test_inicio": P3_TEST_INICIO,
            "test_fin": P3_TEST_FIN,
            "k_clusters": P3_K_ELEGIDO,
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
    # 1. EVALUACIÓN DE LÍNEA BASE GEOGRÁFICA (MUNICIPIO)
    # -------------------------------------------------------------
    logger.info("--- Evaluando Línea Base: Agrupación por Municipio ---")
    m_base_train = evaluar_metricas_clustering(X_train_eval, etiquetas_mun_train, "Train")
    m_base_test = evaluar_metricas_clustering(X_test_eval, etiquetas_mun_test, "Test")

    df_predicciones["baseline_municipio"] = etiquetas_mun_test
    imprimir_metricas_modelo("baseline_municipio", m_base_train, m_base_test)

    fig_mapa_base = graficar_mapa_grupos(
        perfil_train,
        etiquetas_mun_train,
        df_stations,
        "Agrupación por Municipio (Línea Base)",
    )

    tracker.registrar_run(
        run_name="P3_baseline_municipio",
        params={**params_base, "modelo": "BaselineMunicipio"},
        tags={**tags_base, "version": "baseline_municipio"},
        tabla=pd.DataFrame([m_base_train, m_base_test]),
        tabla_est=pd.DataFrame(),
        figuras={"mapa_municipio": fig_mapa_base},
    )

    filas_comparacion.append({
        "modelo": "baseline_municipio",
        "train_silhouette": m_base_train["silhouette"],
        "train_davies_bouldin": m_base_train["davies_bouldin"],
        "test_silhouette": m_base_test["silhouette"],
        "test_davies_bouldin": m_base_test["davies_bouldin"],
    })

    # -------------------------------------------------------------
    # 2. MODELO CANDIDATO: K-MEANS FUNCIONAL (K=4)
    # -------------------------------------------------------------
    logger.info(f"--- Entrenando Modelo Candidato: KMeans (K={P3_K_ELEGIDO}) ---")
    modelo_kmeans = ClusterKMeansP3(
        n_clusters=P3_K_ELEGIDO, n_init=10, random_state=RANDOM_SEED
    )
    modelo_kmeans.fit(perfil_train[P3_VARIABLES])

    # Guardar el modelo con el nombre esperado por pytest (p3_kmeans_model.pkl) y alias de compatibilidad
    ruta_modelo = MODELS_DIR / "p3_kmeans_model.pkl"
    modelo_kmeans.save(ruta_modelo)
    modelo_kmeans.save(MODELS_DIR / "p3_kmeans_pipeline.pkl")
    logger.info(f"Pipeline P3 persistido exitosamente en {ruta_modelo}")

    pred_train_k = modelo_kmeans.labels_
    pred_test_k = modelo_kmeans.predict(perfil_test[P3_VARIABLES])

    m_k_train = evaluar_metricas_clustering(X_train_eval, pred_train_k, "Train")
    m_k_test = evaluar_metricas_clustering(X_test_eval, pred_test_k, "Test")

    df_predicciones["grupo_kmeans_k4"] = pred_test_k
    imprimir_metricas_modelo("kmeans_k4", m_k_train, m_k_test)

    # Mapa de clústeres y resumen funcional
    fig_mapa_kmeans = graficar_mapa_grupos(
        perfil_train,
        pred_train_k,
        df_stations,
        f"Grupos de Estaciones MiBici (KMeans K={P3_K_ELEGIDO})",
    )

    tracker.registrar_run(
        run_name="P3_kmeans_k4",
        params={**params_base, "modelo": "KMeans", "n_clusters": P3_K_ELEGIDO},
        tags={**tags_base, "version": "v1_kmeans"},
        tabla=pd.DataFrame([m_k_train, m_k_test]),
        tabla_est=pd.DataFrame(),
        modelo=modelo_kmeans.pipeline,
        X_ejemplo=perfil_train[P3_VARIABLES].head(5),
        figuras={"mapa_kmeans_k4": fig_mapa_kmeans},
    )

    filas_comparacion.append({
        "modelo": "kmeans_k4",
        "train_silhouette": m_k_train["silhouette"],
        "train_davies_bouldin": m_k_train["davies_bouldin"],
        "test_silhouette": m_k_test["silhouette"],
        "test_davies_bouldin": m_k_test["davies_bouldin"],
    })

    # -------------------------------------------------------------
    # 3. GUARDAR Y MOSTRAR TABLA COMPARATIVA DE MÉTRICAS
    # -------------------------------------------------------------
    df_p3_comp = pd.DataFrame(filas_comparacion).set_index("modelo")
    p3_comp_path = REPORTS_DIR / "p3_comparacion.csv"
    df_p3_comp.round(3).to_csv(p3_comp_path)

    tabla_metricas_str = df_p3_comp.round(3).to_string()
    logger.info(
        f"\n{'='*75}\n"
        f"        RESUMEN COMPARATIVO DE MODELOS (P3 - CLUSTERING DE ESTACIONES)\n"
        f"{'='*75}\n"
        f"{tabla_metricas_str}\n"
        f"{'='*75}\n"
        f"Archivo guardado en: {p3_comp_path}"
    )

    # -------------------------------------------------------------
    # 4. GUARDAR Y MOSTRAR ASIGNACIONES DE CLÚSTERES
    # -------------------------------------------------------------
    preds_out_path = REPORTS_DIR / "p3_predicciones_modelos.parquet"
    df_predicciones.to_parquet(preds_out_path, index=False)

    cols_mostrar = [
        c for c in [
            "origen_id", "baseline_municipio", "grupo_kmeans_k4",
            "pct_manana", "pct_tarde", "pct_fin_de_semana", "viajes_dia_log"
        ] if c in df_predicciones.columns
    ]
    tabla_preds_str = df_predicciones[cols_mostrar].head(10).to_string(index=False)
    logger.info(
        f"\n{'-'*75}\n"
        f"        MUESTRA COMPARATIVA DE ASIGNACIONES (TEST - 10 ESTACIONES)\n"
        f"{'-'*75}\n"
        f"{tabla_preds_str}\n"
        f"{'-'*75}\n"
        f"Asignaciones consolidadas guardadas en: {preds_out_path}\n"
    )


if __name__ == "__main__":
    app()