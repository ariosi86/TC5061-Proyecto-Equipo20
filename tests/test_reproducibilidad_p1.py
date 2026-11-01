"""Pruebas de reproducibilidad del entrenamiento de P1 (demanda horaria).

Un entrenamiento es reproducible si, con los mismos datos y la misma semilla (RANDOM_SEED de
config.py), dos corridas dan exactamente las mismas predicciones y métricas.

- Regresión lineal y regresión de Poisson (sklearn) no tienen azar: siempre dan lo mismo.
- LightGBM solo usa la semilla cuando hay submuestreo (filas o columnas). Con los hiperparámetros
  de los candidatos de train_p1 (sin submuestreo) es determinista; por eso, además, se prueba con
  submuestreo activo, que es el caso en que la semilla decide el resultado.
- Control: si cambiar la semilla no cambiara nada, las pruebas de igualdad pasarían aunque la
  semilla no se estuviera usando.

Los datos son sintéticos (fixture `datos_modelado` de conftest.py): no usan el dataset real ni MLflow.
"""
import joblib
import lightgbm as lgb
import numpy as np
import pytest
from sklearn.base import clone

from proyecto_mibici.config import MODELS_DIR, RANDOM_SEED
from proyecto_mibici.modeling.model_p1 import DemandaLinealP1, DemandaPoissonP1
from proyecto_mibici.modeling.train_p1 import evaluar_metricas_regresion

# Mismos objetivos que usa train_p1 (poisson y tweedie, este último es el champion).
# Se reducen n_estimators y num_leaves para que la prueba corra en segundos.
OBJETIVOS_LGBM = [
    pytest.param({"objective": "poisson"}, id="poisson"),
    pytest.param({"objective": "tweedie", "tweedie_variance_power": 1.8}, id="tweedie"),
]


def _lgbm(semilla, **objetivo):
    return lgb.LGBMRegressor(learning_rate=0.1, num_leaves=15, n_estimators=50,
                             random_state=semilla, verbose=-1, **objetivo)


def _lgbm_con_submuestreo(semilla, **objetivo):
    """LightGBM con submuestreo de filas: aquí la semilla sí decide qué filas usa cada árbol."""
    return lgb.LGBMRegressor(n_estimators=50, subsample=0.7, subsample_freq=1,
                             random_state=semilla, verbose=-1, **objetivo)


# ---------- misma semilla -> mismos resultados ----------

@pytest.mark.parametrize("objetivo", OBJETIVOS_LGBM)
def test_lightgbm_misma_semilla_mismas_predicciones(objetivo, datos_modelado):
    X, y = datos_modelado
    p1 = np.clip(_lgbm(RANDOM_SEED, **objetivo).fit(X, y).predict(X), 0, None)
    p2 = np.clip(_lgbm(RANDOM_SEED, **objetivo).fit(X, y).predict(X), 0, None)
    np.testing.assert_array_equal(p1, p2)


@pytest.mark.parametrize("objetivo", OBJETIVOS_LGBM)
def test_lightgbm_con_submuestreo_misma_semilla_mismo_resultado(objetivo, datos_modelado):
    X, y = datos_modelado
    p1 = _lgbm_con_submuestreo(RANDOM_SEED, **objetivo).fit(X, y).predict(X)
    p2 = _lgbm_con_submuestreo(RANDOM_SEED, **objetivo).fit(X, y).predict(X)
    np.testing.assert_array_equal(p1, p2)


@pytest.mark.parametrize("Modelo", [DemandaLinealP1, DemandaPoissonP1])
def test_modelos_sklearn_entrenamiento_reproducible(Modelo, datos_modelado):
    X, y = datos_modelado
    p1 = Modelo(features=list(X.columns)).fit(X, y).predict(X)
    p2 = Modelo(features=list(X.columns)).fit(X, y).predict(X)
    np.testing.assert_array_equal(p1, p2)


@pytest.mark.parametrize("objetivo", OBJETIVOS_LGBM)
def test_metricas_reproducibles_entre_dos_entrenamientos(objetivo, datos_modelado):
    X, y = datos_modelado
    m1 = evaluar_metricas_regresion(y, _lgbm(RANDOM_SEED, **objetivo).fit(X, y).predict(X), "Train")
    m2 = evaluar_metricas_regresion(y, _lgbm(RANDOM_SEED, **objetivo).fit(X, y).predict(X), "Train")
    assert m1 == m2


RUTA_CHAMPION = MODELS_DIR / "p1_champion.pkl"


@pytest.mark.skipif(not RUTA_CHAMPION.exists(), reason="No existe models/p1_champion.pkl (clon sin modelos entrenados)")
def test_champion_guardado_es_reproducible_con_sus_hiperparametros(datos_modelado):
    """Lee los hiperparámetros del champion guardado (los de la búsqueda anidada), no los copia:
    así la prueba no se desactualiza si el champion cambia. Entrena ese mismo modelo dos veces con
    RANDOM_SEED sobre datos sintéticos y compara las predicciones y las métricas."""
    champion = joblib.load(RUTA_CHAMPION)
    if not hasattr(champion, "get_params"):
        pytest.skip("El champion guardado no es un estimador con get_params()")
    X, y = datos_modelado
    # Solo inyectar random_state si el campeón lo acepta (p. ej. LightGBM).
    # DemandaLinealP1 solo expone 'features', así que no se le puede pasar random_state.
    params = champion.get_params(deep=False)
    if "random_state" in params:
        params["random_state"] = RANDOM_SEED
    p1 = np.clip(clone(champion).set_params(**params).fit(X, y).predict(X), 0, None)
    p2 = np.clip(clone(champion).set_params(**params).fit(X, y).predict(X), 0, None)
    np.testing.assert_array_equal(p1, p2)
    assert evaluar_metricas_regresion(y, p1, "Train") == evaluar_metricas_regresion(y, p2, "Train")


# ---------- control: la semilla sí influye ----------

@pytest.mark.parametrize("objetivo", OBJETIVOS_LGBM)
def test_la_semilla_si_influye_en_el_entrenamiento(objetivo, datos_modelado):
    """Con submuestreo activo, otra semilla debe dar otro resultado; si no, las pruebas de
    igualdad de arriba pasarían aunque el modelo ignorara la semilla."""
    X, y = datos_modelado
    p1 = _lgbm_con_submuestreo(RANDOM_SEED, **objetivo).fit(X, y).predict(X)
    p2 = _lgbm_con_submuestreo(RANDOM_SEED + 1, **objetivo).fit(X, y).predict(X)
    assert not np.array_equal(p1, p2)
