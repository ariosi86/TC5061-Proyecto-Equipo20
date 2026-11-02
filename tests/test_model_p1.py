import numpy as np
import pandas as pd
import pytest

from proyecto_mibici.modeling.model_p1 import (
    BaselineIngenuoP1,
    DemandaLinealP1,
    DemandaPoissonP1,
)


# ---------- BaselineIngenuoP1 ----------

@pytest.fixture
def X_lags():
    return pd.DataFrame({"lag_1h": [1, 2, 3], "lag_24h": [4, 6, np.nan], "lag_168h": [2, 2, 2]})


def test_baseline_persistencia_devuelve_lag_1h(X_lags):
    pred = BaselineIngenuoP1("persistencia").fit(X_lags).predict(X_lags)
    np.testing.assert_array_equal(pred, [1, 2, 3])


def test_baseline_promedio_24h_168h(X_lags):
    pred = BaselineIngenuoP1("promedio_24h_168h").predict(X_lags)
    np.testing.assert_allclose(pred, [3.0, 4.0, 1.0])  # el NaN cuenta como 0


def test_baseline_rellena_nulos_con_cero(X_lags):
    pred = BaselineIngenuoP1("lag_24h").predict(X_lags)
    assert not np.isnan(pred).any()
    assert pred[2] == 0


def test_baseline_nunca_predice_negativos():
    X = pd.DataFrame({"lag_1h": [-5, 2]})
    assert (BaselineIngenuoP1("persistencia").predict(X) >= 0).all()


def test_baseline_estrategia_invalida_lanza_error():
    with pytest.raises(ValueError, match="no válida"):
        BaselineIngenuoP1("estrategia_inventada")


# ---------- DemandaLinealP1 / DemandaPoissonP1 ----------

MODELOS = [DemandaLinealP1, DemandaPoissonP1]


@pytest.mark.parametrize("Modelo", MODELOS)
def test_modelo_fit_predict_forma_y_sin_negativos(Modelo, datos_modelado):
    X, y = datos_modelado
    pred = Modelo(features=list(X.columns)).fit(X, y).predict(X)
    assert pred.shape == (len(X),)
    assert not np.isnan(pred).any()
    assert (pred >= 0).all()


@pytest.mark.parametrize("Modelo", MODELOS)
def test_modelo_expone_scaler_en_el_pipeline(Modelo):
    assert "scaler" in Modelo().pipeline.named_steps


@pytest.mark.parametrize("Modelo", MODELOS)
def test_modelo_usa_solo_las_features_indicadas(Modelo, datos_modelado):
    """Una columna extra en X no debe afectar las predicciones."""
    X, y = datos_modelado
    feats = list(X.columns)
    X_extra = X.assign(columna_ruido=999)
    p1 = Modelo(features=feats).fit(X, y).predict(X)
    p2 = Modelo(features=feats).fit(X_extra, y).predict(X_extra)
    np.testing.assert_allclose(p1, p2)


@pytest.mark.parametrize("Modelo", MODELOS)
def test_modelo_falla_si_faltan_columnas(Modelo, datos_modelado):
    X, y = datos_modelado
    modelo = Modelo(features=list(X.columns)).fit(X, y)
    with pytest.raises(KeyError):
        modelo.predict(X.drop(columns=["lag_1h"]))


@pytest.mark.parametrize("Modelo", MODELOS)
def test_modelo_guardar_y_cargar_da_mismas_predicciones(Modelo, datos_modelado, tmp_path):
    X, y = datos_modelado
    modelo = Modelo(features=list(X.columns)).fit(X, y)
    ruta = tmp_path / "subcarpeta" / "modelo.pkl"   # la subcarpeta no existe: save() debe crearla
    modelo.save(ruta)
    cargado = Modelo.load(ruta)
    np.testing.assert_allclose(modelo.predict(X), cargado.predict(X))


def test_lineal_aprende_una_relacion_lineal_exacta():
    X = pd.DataFrame({"a": np.arange(50, dtype=float)})
    y = 2 * X["a"] + 3
    pred = DemandaLinealP1(features=["a"]).fit(X, y).predict(X)
    np.testing.assert_allclose(pred, y, atol=1e-8)


def test_lineal_recorta_predicciones_negativas_a_cero():
    X = pd.DataFrame({"a": np.arange(10, dtype=float)})
    y = X["a"] - 5                       # y tiene valores negativos
    pred = DemandaLinealP1(features=["a"]).fit(X, y).predict(X)
    assert pred.min() == 0
