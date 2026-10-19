#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Suite de pruebas de integración y validación de artefactos para MiBici.
Ejecución independiente: pytest tests/ -v
"""

import json
import numpy as np
import pandas as pd

from proyecto_mibici.config import (
    FIGURES_DIR,
    MANIFEST_PATH,
    MODELS_DIR,
    P1_FEATURES_V2,
    P2_VARIABLES,
    P3_K_ELEGIDO,
    P3_VARIABLES,
    REPORTS_DIR,
    STATIONS_PROCESSED_PATH,
    TRIPS_PROCESSED_PATH,
    TRIPS_TRAIN_PATH,
    TRIPS_TEST_PATH,
)
from proyecto_mibici.modeling import (
    ClusterEstacionesP3,
    DemandaLinealP1,
    RiesgoLogisticoP2,
)


def test_archivos_datos_y_manifiesto():
    """Verifica que los Parquets procesados (maestro, train y test), el manifest y las estaciones existan y sean válidos."""
    assert TRIPS_PROCESSED_PATH.exists(), "Falta trips.parquet maestro"
    assert STATIONS_PROCESSED_PATH.exists(), "Falta stations.parquet"
    assert TRIPS_TRAIN_PATH.exists(), "Falta el parquet de entrenamiento (2025)"
    assert TRIPS_TEST_PATH.exists(), "Falta el parquet de prueba (Abr-Sep 2026)"
    assert MANIFEST_PATH.exists(), "Falta manifest.json"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "trips" in manifest and "sha256" in manifest["trips"]
    assert manifest["trips"]["total_rows"] > 0

    # Validación adicional de rangos en los parquets particionados
    df_train = pd.read_parquet(TRIPS_TRAIN_PATH, columns=["inicio_del_viaje"])
    df_test = pd.read_parquet(TRIPS_TEST_PATH, columns=["inicio_del_viaje"])

    assert len(df_train) > 0, "El dataset de train 2025 está vacío"
    assert len(df_test) > 0, "El dataset de test 2026 está vacío"

    assert df_train["inicio_del_viaje"].min().year == 2025, "Train contiene datos fuera del año 2025"
    assert df_test["inicio_del_viaje"].min().month >= 4, "Test de 2026 inicia antes de abril"
    assert df_test["inicio_del_viaje"].max().month <= 9, "Test de 2026 supera septiembre"


def test_modelos_serializados_y_pipelines():
    """Verifica que los 3 modelos .pkl existan, carguen correctamente y tengan StandardScaler."""
    ruta_p1 = MODELS_DIR / "p1_linear_model.pkl"
    ruta_p2 = MODELS_DIR / "p2_logistic_model.pkl"
    ruta_p3 = MODELS_DIR / "p3_kmeans_model.pkl"

    assert ruta_p1.exists(), "No se generó p1_linear_model.pkl"
    assert ruta_p2.exists(), "No se generó p2_logistic_model.pkl"
    assert ruta_p3.exists(), "No se generó p3_kmeans_model.pkl"

    m1 = DemandaLinealP1.load(ruta_p1)
    m2 = RiesgoLogisticoP2.load(ruta_p2)
    m3 = ClusterEstacionesP3.load(ruta_p3)

    assert "scaler" in m1.pipeline.named_steps
    assert "scaler" in m2.pipeline.named_steps
    assert "scaler" in m3.pipeline.named_steps


def test_inferencia_sintetica_modelos():
    """Verifica que los modelos cargados respondan con formatos válidos ante datos sintéticos."""
    rng = np.random.default_rng(42)

    # P1: Predicciones >= 0
    m1 = DemandaLinealP1.load(MODELS_DIR / "p1_linear_model.pkl")
    X_p1 = pd.DataFrame(rng.uniform(0, 10, size=(10, len(P1_FEATURES_V2))), columns=P1_FEATURES_V2)
    preds_p1 = m1.predict(X_p1)
    assert len(preds_p1) == 10
    assert (preds_p1 >= 0).all(), "P1 produjo demandas negativas"

    # P2: Probabilidades en [0, 1] y clases booleanas
    m2 = RiesgoLogisticoP2.load(MODELS_DIR / "p2_logistic_model.pkl")
    X_p2 = pd.DataFrame(rng.uniform(-1, 1, size=(10, len(P2_VARIABLES))), columns=P2_VARIABLES)
    probs_p2 = m2.predict_proba(X_p2)
    preds_p2 = m2.predict(X_p2)
    assert ((probs_p2 >= 0.0) & (probs_p2 <= 1.0)).all(), "Probabilidades fuera de [0, 1]"
    assert len(preds_p2) == 10

    # P3: Clusters en [0, K-1]
    m3 = ClusterEstacionesP3.load(MODELS_DIR / "p3_kmeans_model.pkl")
    X_p3 = pd.DataFrame(rng.uniform(0, 1, size=(10, len(P3_VARIABLES))), columns=P3_VARIABLES)
    clusters_p3 = m3.predict(X_p3)
    assert set(clusters_p3).issubset(set(range(P3_K_ELEGIDO)))


def test_figuras_y_reportes_generados():
    """Verifica que las gráficas en reports/figures/ y reportes de inferencia se hayan creado."""
    assert FIGURES_DIR.exists(), "No existe el directorio reports/figures/"
    pngs = list(FIGURES_DIR.glob("*.png"))
    assert len(pngs) >= 6, f"Se esperaban al menos 6 gráficas en reports/figures/, se encontraron {len(pngs)}"

    assert (REPORTS_DIR / "p3_comparacion.csv").exists(), "Falta reporte comparativo de P3"
    assert (REPORTS_DIR / "predicciones_p1_test.parquet").exists(), "Falta salida de predict_p1"
    assert (REPORTS_DIR / "predicciones_p2_test.parquet").exists(), "Falta salida de predict_p2"
    assert (REPORTS_DIR / "predicciones_p3_test.csv").exists(), "Falta salida de predict_p3"