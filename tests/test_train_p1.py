import numpy as np
import pandas as pd
import pytest

from proyecto_mibici.modeling.train_p1 import (
    calcular_metricas_por_estacion,
    evaluar_metricas_regresion,
    registrar_fila_comparacion,
)


# ---------- evaluar_metricas_regresion ----------

def test_metricas_prediccion_perfecta():
    m = evaluar_metricas_regresion([1, 2, 3], [1, 2, 3], "Test")
    assert m["conjunto"] == "Test"
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["WAPE"] == 0 and m["Sesgo"] == 0


def test_metricas_valores_calculados_a_mano():
    # errores: 1, 1, 2 (prediciendo de más)  | reales suman 10
    m = evaluar_metricas_regresion([2, 3, 5], [3, 4, 7], "Val")
    assert m["MAE"] == pytest.approx(4 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt(6 / 3))
    assert m["WAPE"] == pytest.approx(4 / 10)
    assert m["Sesgo"] == pytest.approx(4 / 3)       # positivo = sobreestima


def test_metricas_sesgo_negativo_si_subestima():
    assert evaluar_metricas_regresion([5, 5], [3, 3], "Test")["Sesgo"] < 0


def test_metricas_wape_con_reales_en_cero_no_divide_entre_cero():
    m = evaluar_metricas_regresion([0, 0, 0], [1, 1, 1], "Test")
    assert m["WAPE"] == 0.0         # comportamiento actual: documentado por esta prueba


def test_metricas_aceptan_series_de_pandas():
    m = evaluar_metricas_regresion(pd.Series([1, 2]), pd.Series([1, 2]), "Test")
    assert m["MAE"] == 0


# ---------- calcular_metricas_por_estacion ----------

@pytest.fixture
def df_estaciones():
    return pd.DataFrame({"origen_id": [1, 1, 2, 2], "target_next_hour": [2, 4, 1, 3]})


def test_por_estacion_una_fila_por_estacion(df_estaciones):
    out = calcular_metricas_por_estacion(df_estaciones, np.array([2, 4, 2, 2]), "target_next_hour")
    assert sorted(out["origen_id"]) == [1, 2]
    fila1 = out[out["origen_id"] == 1].iloc[0]
    assert fila1["MAE"] == 0 and fila1["horas_evaluadas"] == 2
    assert fila1["viajes_reales_total"] == 6


def test_por_estacion_error_abs_total(df_estaciones):
    out = calcular_metricas_por_estacion(df_estaciones, np.array([2, 4, 2, 2]), "target_next_hour")
    fila2 = out[out["origen_id"] == 2].iloc[0]
    assert fila2["error_abs_total"] == pytest.approx(2.0)   # |1-2| + |3-2|


def test_por_estacion_agrega_nombre_si_hay_catalogo(df_estaciones):
    catalogo = pd.DataFrame({"id": [1, 2], "name": ["Centro", "Chapultepec"]})
    out = calcular_metricas_por_estacion(df_estaciones, np.zeros(4), "target_next_hour", catalogo)
    assert set(out["nombre_estacion"]) == {"Centro", "Chapultepec"}
    assert "id" not in out.columns


def test_por_estacion_no_modifica_el_dataframe_original(df_estaciones):
    copia = df_estaciones.copy()
    calcular_metricas_por_estacion(df_estaciones, np.zeros(4), "target_next_hour")
    pd.testing.assert_frame_equal(df_estaciones, copia)


def test_registrar_fila_comparacion_agrega_una_fila():
    m = {"MAE": 1.0, "RMSE": 2.0, "WAPE": 0.3, "Sesgo": 0.1}
    filas = []
    registrar_fila_comparacion(filas, "baseline", m, m, m)
    assert len(filas) == 1 and filas[0]["modelo"] == "baseline"
    assert filas[0]["test_rmse"] == 2.0
