"""Prueba de reproducibilidad del entrenamiento de KMeans en P3 (modelo champion).

La semilla RANDOM_SEED de config.py establece el sorteo de los centros iniciales de KMeans.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import davies_bouldin_score, silhouette_score

from proyecto_mibici.config import P3_K_ELEGIDO, P3_VARIABLES, RANDOM_SEED
from proyecto_mibici.modeling.model_p3 import ClusterKMeansP3


@pytest.fixture
def particion():
    """Perfiles sintéticos sin grupos claros, divididos en entrenamiento y prueba.
    se usa la misma semilla que el modelo champion para que los resultados sean reproducibles.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 1, size=(300, len(P3_VARIABLES))), columns=P3_VARIABLES)
    return X.iloc[:240], X.iloc[240:]


def entrenar_y_evaluar(semilla, X_tr, X_te):
    """Entrena un KMeans nuevo con la semilla dada y devuelve etiquetas y métricas."""
    # n_init=1: un solo arranque, para que la semilla decida el resultado
    modelo = ClusterKMeansP3(
        n_clusters=P3_K_ELEGIDO, n_init=1, random_state=semilla, features=P3_VARIABLES
    ).fit(X_tr)
    etiquetas = modelo.predict(X_te)
    X_te_escalado = modelo.pipeline.named_steps["scaler"].transform(X_te[P3_VARIABLES])
    return {
        "etiquetas": etiquetas,
        "centros": modelo.cluster_centers_,
        "silhouette": silhouette_score(X_te_escalado, etiquetas),
        "davies_bouldin": davies_bouldin_score(X_te_escalado, etiquetas),
    }


def test_misma_semilla_produce_los_mismos_resultados(particion):
    r1 = entrenar_y_evaluar(RANDOM_SEED, *particion)
    r2 = entrenar_y_evaluar(RANDOM_SEED, *particion)

    assert np.array_equal(r1["etiquetas"], r2["etiquetas"])
    assert np.array_equal(r1["centros"], r2["centros"])  # igualdad exacta, no aproximada
    assert r1["silhouette"] == r2["silhouette"]
    assert r1["davies_bouldin"] == r2["davies_bouldin"]


def test_la_semilla_si_influye_en_el_entrenamiento(particion):
    # Control: si cambiar la semilla no cambiara nada, la prueba anterior
    # pasaría aunque la semilla no se estuviera usando.
    r1 = entrenar_y_evaluar(RANDOM_SEED, *particion)
    r2 = entrenar_y_evaluar(RANDOM_SEED + 1, *particion)

    # Se comparan los centros ordenados para no confundir un cambio real
    # con una simple renumeración de los grupos
    assert not np.allclose(np.sort(r1["centros"], axis=0), np.sort(r2["centros"], axis=0))