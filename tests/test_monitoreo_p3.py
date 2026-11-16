"""Pruebas de la lógica del A/B testing de P3 con datos sintéticos (sin MLflow)."""

import numpy as np
import pandas as pd
import pytest

from proyecto_mibici.config import P3_VARIABLES
from proyecto_mibici.modeling.model_p3 import ClusterKMeansP3
from proyecto_mibici.monitoring.evidently_monitor import (
    bootstrap_diferencias,
    decidir_p3,
    emparejar_grupos,
    metricas_internas,
    proporcion_cambios,
)


@pytest.fixture
def perfil_sintetico():
    rng = np.random.default_rng(0)
    centros = rng.normal(0, 5, size=(4, len(P3_VARIABLES)))
    filas = np.vstack([c + rng.normal(0, 0.5, size=(30, len(P3_VARIABLES))) for c in centros])
    return pd.DataFrame(filas, columns=P3_VARIABLES, index=pd.Index(range(120), name="estacion_id"))


def test_metricas_internas_separacion_clara(perfil_sintetico):
    etiquetas = np.repeat(np.arange(4), 30)
    m = metricas_internas(perfil_sintetico.to_numpy(), etiquetas)
    assert m["silhouette"] > 0.7
    assert m["davies_bouldin"] < 0.5


def test_metricas_internas_un_solo_grupo(perfil_sintetico):
    m = metricas_internas(perfil_sintetico.to_numpy(), np.zeros(120, dtype=int))
    assert np.isnan(m["silhouette"])


def test_emparejar_grupos_recupera_permutacion():
    centros_a = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    centros_b = centros_a[[2, 0, 1]] + 0.1
    assert emparejar_grupos(centros_a, centros_b) == {0: 2, 1: 0, 2: 1}


def test_bootstrap_versiones_identicas_da_cero(perfil_sintetico):
    etiquetas = np.repeat(np.arange(4), 30)
    boot = bootstrap_diferencias(perfil_sintetico.to_numpy(), etiquetas, etiquetas, n_boot=20, semilla=1)
    assert len(boot) == 20
    assert np.allclose(boot.to_numpy(), 0)


def test_proporcion_cambios_mismo_perfil(perfil_sintetico):
    modelo = ClusterKMeansP3(n_clusters=4, features=P3_VARIABLES).fit(perfil_sintetico)
    r = proporcion_cambios(modelo.pipeline_, perfil_sintetico, perfil_sintetico)
    assert r["estaciones_comunes"] == 120
    assert r["cambian_de_grupo"] == 0


@pytest.mark.parametrize(
    "ic, delta_db, min_prop, promueve",
    [
        ((0.01, 0.03), -0.1, 0.10, True),
        ((-0.01, 0.03), -0.1, 0.10, False),
        ((0.01, 0.03), 0.2, 0.10, False),
        ((0.01, 0.03), -0.1, 0.02, False),
        ((-0.05, -0.01), 0.2, 0.10, False),
    ],
)
def test_regla_de_decision(ic, delta_db, min_prop, promueve):
    _, resultado = decidir_p3(ic, delta_db, min_prop)
    assert resultado is promueve
