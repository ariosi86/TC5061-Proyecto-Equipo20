"""Pruebas simples para funciones de métricas de entrenamiento P2 y P3.

Verifica funciones puras que calculan métricas sin requerir MLflow ni datos reales.
"""
import numpy as np
import pytest

from proyecto_mibici.modeling.train_p2 import extraer_metricas_clasificacion
from proyecto_mibici.modeling.train_p3 import evaluar_metricas_clustering


def test_extraer_metricas_clasificacion_valores_rango():
    """Las métricas de clasificación deben estar en [0, 1]."""
    y_real = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    y_prob = np.array([0.1, 0.8, 0.9, 0.7, 0.2, 0.3, 0.4, 0.6])

    m = extraer_metricas_clasificacion(y_real, y_pred, y_prob, "test")

    assert 0.0 <= m["precision"] <= 1.0  # nosec B101
    assert 0.0 <= m["recall"] <= 1.0  # nosec B101
    assert 0.0 <= m["f1_score"] <= 1.0  # nosec B101
    assert 0.0 <= m["pr_auc"] <= 1.0  # nosec B101
    assert m["conjunto"] == "test"  # nosec B101


def test_extraer_metricas_clasificacion_prediccion_perfecta():
    """Con predicciones perfectas, todas las métricas deben ser 1.0."""
    y_real = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0])

    m = extraer_metricas_clasificacion(y_real, y_pred, y_prob, "train")

    assert m["precision"] == 1.0  # nosec B101
    assert m["recall"] == 1.0  # nosec B101
    assert m["f1_score"] == 1.0  # nosec B101


def test_evaluar_metricas_clustering_rango_valido():
    """Silhouette debe estar en [-1, 1] y Davies-Bouldin >= 0."""
    X = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
    etiquetas = np.array([0, 0, 1, 1])

    m = evaluar_metricas_clustering(X, etiquetas, "validacion")

    assert -1.0 <= m["silhouette"] <= 1.0  # nosec B101
    assert m["davies_bouldin"] >= 0.0  # nosec B101
    assert m["conjunto"] == "validacion"  # nosec B101


def test_evaluar_metricas_clustering_una_sola_etiqueta():
    """Con una sola etiqueta, silhouette = 0.0 y davies_bouldin = 99.0."""
    X = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
    etiquetas = np.array([0, 0, 0])

    m = evaluar_metricas_clustering(X, etiquetas, "train")

    assert m["silhouette"] == 0.0  # nosec B101
    assert m["davies_bouldin"] == 99.0  # nosec B101
