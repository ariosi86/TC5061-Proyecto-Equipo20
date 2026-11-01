import numpy as np
import pandas as pd
import pytest

from proyecto_mibici.config import P1_FEATURES_V2
from proyecto_mibici.modeling.model_p1 import DemandaLinealP1
from proyecto_mibici.modeling.predict_p1 import (
    cargar_modelo,
    generar_reporte_inferencia,
    predecir_todos_los_modelos,
)

COLUMNAS_ESPERADAS = {
    "pred_naive_persistencia", "pred_naive_lag_24h", "pred_naive_lag_168h",
    "pred_naive_promedio_24h_168h", "pred_v1_lineal", "pred_v2_lineal_rolling",
    "demanda_predicha",
}


def test_predecir_devuelve_las_columnas_esperadas(datos_modelado, carpeta_modelos):
    X, _ = datos_modelado
    preds = predecir_todos_los_modelos(X, models_dir=carpeta_modelos)
    assert set(preds) == COLUMNAS_ESPERADAS


def test_predicciones_tienen_longitud_correcta_sin_nan_ni_negativos(datos_modelado, carpeta_modelos):
    X, _ = datos_modelado
    for nombre, valores in predecir_todos_los_modelos(X, models_dir=carpeta_modelos).items():
        assert len(valores) == len(X), nombre
        assert not np.isnan(valores).any(), nombre
        assert (valores >= 0).all(), nombre


def test_demanda_predicha_es_alias_de_v2(datos_modelado, carpeta_modelos):
    """pipeline.py espera 'demanda_predicha': debe ser igual a la predicción de v2."""
    X, _ = datos_modelado
    preds = predecir_todos_los_modelos(X, models_dir=carpeta_modelos)
    np.testing.assert_array_equal(preds["demanda_predicha"], preds["pred_v2_lineal_rolling"])


def test_modelo_cargado_da_las_mismas_predicciones_que_el_original(datos_modelado, carpeta_modelos):
    X, y = datos_modelado
    original = DemandaLinealP1(features=P1_FEATURES_V2).fit(X, y)
    preds = predecir_todos_los_modelos(X, models_dir=carpeta_modelos)
    np.testing.assert_allclose(preds["pred_v2_lineal_rolling"], np.round(original.predict(X), 2))


def test_cargar_modelo_inexistente_lanza_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_p1"):
        cargar_modelo("no_existe.pkl", tmp_path)


def test_usa_p1_linear_model_si_no_existe_v2(datos_modelado, carpeta_modelos):
    """Fallback: sin p1_v2_model.pkl se usa p1_linear_model.pkl."""
    (carpeta_modelos / "p1_v2_model.pkl").rename(carpeta_modelos / "p1_linear_model.pkl")
    X, _ = datos_modelado
    preds = predecir_todos_los_modelos(X, models_dir=carpeta_modelos)
    assert "pred_v2_lineal_rolling" in preds


def test_falla_si_faltan_columnas_de_features(datos_modelado, carpeta_modelos):
    X, _ = datos_modelado
    with pytest.raises(KeyError):
        predecir_todos_los_modelos(X.drop(columns=["rolling_3h"]), models_dir=carpeta_modelos)


def test_reporte_conserva_columnas_originales_y_no_modifica_la_entrada(datos_modelado, carpeta_modelos):
    X, _ = datos_modelado
    X = X.assign(origen_id=1)
    copia = X.copy()
    out = generar_reporte_inferencia(X, models_dir=carpeta_modelos)
    assert set(X.columns).issubset(out.columns)
    assert COLUMNAS_ESPERADAS.issubset(out.columns)
    assert len(out) == len(X)
    pd.testing.assert_frame_equal(X, copia)          # la entrada queda intacta


def test_inferencia_es_determinista(datos_modelado, carpeta_modelos):
    X, _ = datos_modelado
    a = generar_reporte_inferencia(X, models_dir=carpeta_modelos)
    b = generar_reporte_inferencia(X, models_dir=carpeta_modelos)
    pd.testing.assert_frame_equal(a, b)
