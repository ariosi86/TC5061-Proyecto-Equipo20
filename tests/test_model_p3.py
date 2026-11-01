"""Pruebas unitarias de los modelos de P3 (clustering de estaciones).

Usan datos sintéticos por lo que no leen los Parquet ni se necesita MLflow.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from proyecto_mibici.config import P3_VARIABLES
from proyecto_mibici.modeling.model_p3 import BaselineMunicipioP3, ClusterKMeansP3


@pytest.fixture
def perfiles():
    """120 estaciones sintéticas en 4 grupos bien separados (30 por grupo)."""
    rng = np.random.default_rng(0)
    centros = rng.uniform(-10, 10, size=(4, len(P3_VARIABLES)))
    grupo_real = np.repeat(np.arange(4), 30)
    X = pd.DataFrame(
        centros[grupo_real] + rng.normal(0, 0.3, size=(120, len(P3_VARIABLES))),
        columns=P3_VARIABLES,
    )
    X["municipality_code"] = rng.integers(1, 4, 120)  # columna extra que KMeans debe ignorar
    return X, grupo_real


# ---------------------------------------------------------------------------
# BaselineMunicipioP3
# ---------------------------------------------------------------------------
def test_baseline_asigna_un_grupo_por_municipio():
    X = pd.DataFrame({"municipality_code": [14, 39, 14, 120]})

    etiquetas = BaselineMunicipioP3().fit_predict(X)

    # Los municipios ordenados (14, 39, 120) reciben 0, 1, 2
    assert list(etiquetas) == [0, 1, 0, 2]


def test_baseline_usa_el_catalogo_si_falta_la_columna():
    X = pd.DataFrame({"pct_manana": [0.1, 0.2]}, index=[1, 2])
    catalogo = pd.DataFrame({"id": [1, 2], "municipality_code": [39, 14]})

    etiquetas = BaselineMunicipioP3().fit_predict(X, df_stations=catalogo)

    assert list(etiquetas) == [1, 0]


def test_baseline_municipio_nuevo_recibe_menos_uno():
    modelo = BaselineMunicipioP3().fit(pd.DataFrame({"municipality_code": [14, 39]}))

    etiquetas = modelo.predict(pd.DataFrame({"municipality_code": [14, 98]}))

    assert list(etiquetas) == [0, -1]


def test_baseline_falla_sin_municipio_ni_catalogo():
    with pytest.raises(KeyError):
        BaselineMunicipioP3().fit(pd.DataFrame({"pct_manana": [0.1]}))


def test_baseline_falla_si_predice_sin_entrenar():
    with pytest.raises(ValueError):
        BaselineMunicipioP3().predict(pd.DataFrame({"municipality_code": [14]}))


# ---------------------------------------------------------------------------
# ClusterKMeansP3
# ---------------------------------------------------------------------------
def test_kmeans_entrega_una_etiqueta_valida_por_estacion(perfiles):
    X, _ = perfiles

    etiquetas = ClusterKMeansP3(n_clusters=4, features=P3_VARIABLES).fit_predict(X)

    assert len(etiquetas) == len(X)
    assert set(np.unique(etiquetas)) == {0, 1, 2, 3}


def test_kmeans_recupera_grupos_bien_separados(perfiles):
    X, grupo_real = perfiles

    etiquetas = ClusterKMeansP3(n_clusters=4, features=P3_VARIABLES).fit_predict(X)

    # ARI = 1 significa la misma partición, aunque los números de grupo cambien
    assert adjusted_rand_score(grupo_real, etiquetas) == pytest.approx(1.0)


def test_kmeans_escala_dentro_del_pipeline(perfiles):
    X, _ = perfiles

    modelo = ClusterKMeansP3(n_clusters=4, features=P3_VARIABLES).fit(X)
    escalador = modelo.pipeline.named_steps["scaler"]

    assert list(modelo.pipeline.named_steps) == ["scaler", "clusterer"]
    assert np.allclose(escalador.mean_, X[P3_VARIABLES].mean())


def test_kmeans_usa_solo_las_variables_indicadas(perfiles):
    X, _ = perfiles
    modelo = ClusterKMeansP3(n_clusters=4, features=P3_VARIABLES).fit(X)

    # Sin la columna extra y con las variables en otro orden: mismas etiquetas
    X_reducido = X[list(reversed(P3_VARIABLES))]

    assert modelo.features_ == P3_VARIABLES
    assert np.array_equal(modelo.predict(X), modelo.predict(X_reducido))


def test_kmeans_falla_si_predice_sin_entrenar(perfiles):
    X, _ = perfiles

    with pytest.raises(ValueError):
        ClusterKMeansP3(features=P3_VARIABLES).predict(X)


def test_kmeans_guarda_y_carga_sin_cambiar_etiquetas(perfiles, tmp_path):
    X, _ = perfiles
    modelo = ClusterKMeansP3(n_clusters=4, features=P3_VARIABLES).fit(X)
    ruta = tmp_path / "kmeans_p3.pkl"  # carpeta temporal: no ensucia el proyecto

    modelo.save(ruta)
    cargado = ClusterKMeansP3.load(ruta)

    assert np.array_equal(modelo.predict(X), cargado.predict(X))