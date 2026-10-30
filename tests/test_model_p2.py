"""Pruebas unitarias de los modelos de P2 (riesgo de desbalance).

Usan datos sintéticos pequeños: no leen los Parquet reales ni necesitan MLflow.
"""

import numpy as np
import pandas as pd
import pytest

from proyecto_mibici.modeling.model_p2 import BaselinePersistenciaP2, RiesgoGBMP2

VARIABLES = ["hora_del_dia", "dia_semana", "acumulado_relativo", "balance_ult_3h_relativo"]


@pytest.fixture
def datos():
    """300 filas sintéticas con un objetivo que depende del acumulado relativo."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({
        "hora_del_dia": rng.integers(0, 24, n),
        "dia_semana": rng.integers(0, 7, n),
        "acumulado_relativo": rng.uniform(-1, 1, n),
        "balance_ult_3h_relativo": rng.uniform(-0.5, 0.5, n),
        "en_riesgo": rng.integers(0, 2, n).astype(bool),
        "conjunto": "entrenamiento",   # columna de texto que el modelo debe ignorar
    })
    y = X["acumulado_relativo"].abs() > 0.4
    return X, y


# ---------------------------------------------------------------------------
# BaselinePersistenciaP2
# ---------------------------------------------------------------------------
def test_baseline_predice_el_estado_actual(datos):
    X, _ = datos

    modelo = BaselinePersistenciaP2().fit(X)
    pred = modelo.predict(X)
    proba = modelo.predict_proba(X)

    assert np.array_equal(pred, X["en_riesgo"].to_numpy())
    assert proba.shape == (len(X), 2)
    assert set(np.unique(proba)) <= {0.0, 1.0}


def test_baseline_falla_si_falta_la_columna(datos):
    X, _ = datos

    with pytest.raises(KeyError):
        BaselinePersistenciaP2().predict(X.drop(columns=["en_riesgo"]))


# ---------------------------------------------------------------------------
# RiesgoGBMP2
# ---------------------------------------------------------------------------
def test_gbm_entrena_y_predice_clases_validas(datos):
    X, y = datos

    modelo = RiesgoGBMP2(features=VARIABLES).fit(X, y)
    pred = modelo.predict(X)
    proba = modelo.predict_proba(X)

    assert len(pred) == len(X)
    assert set(np.unique(pred)) <= {False, True}
    assert np.allclose(proba.sum(axis=1), 1.0)
    # El patrón es sencillo: el modelo debe aprenderlo bastante bien
    assert (pred == y.to_numpy()).mean() > 0.9


def test_gbm_falla_si_se_predice_sin_entrenar(datos):
    X, _ = datos

    with pytest.raises(ValueError):
        RiesgoGBMP2(features=VARIABLES).predict(X)


def test_gbm_usa_solo_las_variables_indicadas(datos):
    X, y = datos
    modelo = RiesgoGBMP2(features=VARIABLES).fit(X, y)

    # Mismas variables en otro orden y sin columnas extra: misma predicción
    X_reducido = X[list(reversed(VARIABLES))]

    assert modelo.features_ == VARIABLES
    assert np.array_equal(modelo.predict(X), modelo.predict(X_reducido))


def test_gbm_guarda_y_carga_sin_cambiar_predicciones(datos, tmp_path):
    X, y = datos
    modelo = RiesgoGBMP2(features=VARIABLES).fit(X, y)
    ruta = tmp_path / "gbm_p2.pkl"   # carpeta temporal: no ensucia el proyecto

    modelo.save(ruta)
    cargado = RiesgoGBMP2.load(ruta)

    assert np.array_equal(modelo.predict_proba(X), cargado.predict_proba(X))
