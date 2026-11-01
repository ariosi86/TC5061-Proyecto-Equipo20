"""Pruebas simples para funciones de inferencia P2 y P3.

Verifica carga de modelos serializados y generación de predicciones
usando modelos temporales en tmp_path.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import pytest

from proyecto_mibici.config import P2_VARIABLES, P3_VARIABLES
from proyecto_mibici.modeling.model_p2 import RiesgoLogisticoP2
from proyecto_mibici.modeling.model_p3 import ClusterKMeansP3
from proyecto_mibici.modeling.predict_p2 import (
    cargar_pipeline_p2 as cargar_p2,
    predecir_todos_los_modelos as predecir_p2,
)
from proyecto_mibici.modeling.predict_p3 import (
    cargar_pipeline_p3 as cargar_p3,
    predecir_todos_los_modelos as predecir_p3,
)


def _crear_modelo_p2_temporal(tmp_path, features):
    """Crea y guarda un modelo P2 temporal para pruebas de carga."""
    rng = np.random.default_rng(42)
    n = 50
    X = pd.DataFrame(rng.uniform(0, 10, size=(n, len(features))), columns=features)
    y = rng.integers(0, 2, n)

    modelo = RiesgoLogisticoP2(features=features)
    modelo.fit(X, y)
    ruta = tmp_path / "modelo_p2_temp.pkl"
    modelo.save(ruta)
    return ruta


def _crear_modelo_p3_temporal(tmp_path, features):
    """Crea y guarda un modelo P3 temporal para pruebas de carga."""
    rng = np.random.default_rng(42)
    n = 30
    X = pd.DataFrame(rng.uniform(0, 1, size=(n, len(features))), columns=features)

    modelo = ClusterKMeansP3(features=features, n_clusters=2, n_init=5)
    modelo.fit(X)
    ruta = tmp_path / "modelo_p3_temp.pkl"
    modelo.save(ruta)
    return ruta


def test_cargar_pipeline_p2_existente(tmp_path):
    """Cargar un modelo P2 guardado debe devolver una instancia de RiesgoLogisticoP2."""
    ruta = _crear_modelo_p2_temporal(tmp_path, P2_VARIABLES)
    modelo = cargar_p2(ruta)

    assert isinstance(modelo, RiesgoLogisticoP2)  # nosec B101


def test_cargar_pipeline_p2_no_existente_lanza_error():
    """Cargar un modelo P2 que no existe debe lanzar FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        cargar_p2(Path("/ruta/que/no/existe/modelo.pkl"))


def test_predecir_todos_los_modelos_p2(tmp_path):
    """La inferencia P2 debe devolver columnas esperadas con valores válidos."""
    ruta = _crear_modelo_p2_temporal(tmp_path, P2_VARIABLES)
    rng = np.random.default_rng(7)
    n = 10
    df = pd.DataFrame(rng.uniform(0, 10, size=(n, len(P2_VARIABLES))), columns=P2_VARIABLES)

    preds = predecir_p2(df, ruta_modelo=ruta, umbral=0.5)

    assert "prob_riesgo" in preds  # nosec B101
    assert "pred_riesgo" in preds  # nosec B101
    assert len(preds["prob_riesgo"]) == n  # nosec B101
    assert len(preds["pred_riesgo"]) == n  # nosec B101
    assert ((preds["prob_riesgo"] >= 0.0) & (preds["prob_riesgo"] <= 1.0)).all()  # nosec B101
    assert set(np.unique(preds["pred_riesgo"])).issubset({0, 1})  # nosec B101


def test_cargar_pipeline_p3_existente(tmp_path):
    """Cargar un modelo P3 guardado debe devolver una instancia de ClusterKMeansP3."""
    ruta = _crear_modelo_p3_temporal(tmp_path, P3_VARIABLES)
    modelo = cargar_p3(ruta)

    assert isinstance(modelo, ClusterKMeansP3)  # nosec B101


def test_cargar_pipeline_p3_no_existente_lanza_error():
    """Cargar un modelo P3 que no existe debe lanzar FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        cargar_p3(Path("/ruta/que/no/existe/modelo.pkl"))


def test_predecir_todos_los_modelos_p3(tmp_path):
    """La inferencia P3 debe devolver grupos válidos en el rango esperado."""
    ruta = _crear_modelo_p3_temporal(tmp_path, P3_VARIABLES)
    rng = np.random.default_rng(7)
    n = 10
    df = pd.DataFrame(rng.uniform(0, 1, size=(n, len(P3_VARIABLES))), columns=P3_VARIABLES)

    preds = predecir_p3(df, ruta_modelo=ruta)

    assert "grupo_kmeans_k4" in preds  # nosec B101
    assert "grupo" in preds  # nosec B101
    assert len(preds["grupo_kmeans_k4"]) == n  # nosec B101
    # Los grupos deben estar en el rango [0, n_clusters-1]
    assert set(np.unique(preds["grupo_kmeans_k4"])).issubset({0, 1})  # nosec B101
