import numpy as np
import pandas as pd
import pytest

from proyecto_mibici.features_p1 import (
    GeneradorDemandaLagsP1,
    LimpiadorFiltroEstacionesP1,
)

# ---------- PRUEBAS PARA FUNCIÓN LimpiadorFiltroEstacionesP1 ----------

def test_limpiador_elimina_duplicados(viajes_chicos, estaciones):
    """Prueba que la función elimina registros idénticos sin perder los originales"""
    con_duplicados = pd.concat([viajes_chicos, viajes_chicos.iloc[:10]]) #10 registros duplicados intencionalmente
    limpiador = LimpiadorFiltroEstacionesP1().fit(con_duplicados, df_stations=estaciones)
    out = limpiador.transform(con_duplicados)
    assert len(out) == len(viajes_chicos) #Tabla con 200 registros
    assert out["viaje_id"].is_unique 


def test_limpiador_descarta_estacion_fuera_de_catalogo(viajes_chicos):
    """Estación 1 existe en los viajes pero no en el catálogo -> se descarta."""
    catalogo = pd.DataFrame({"id": [99]})
    limpiador = LimpiadorFiltroEstacionesP1().fit(viajes_chicos, df_stations=catalogo)
    assert limpiador.transform(viajes_chicos).empty


def test_limpiador_descarta_estacion_con_muchas_horas_vacias(estaciones):
    """Viajes solo en la hora 0 y la hora 99 -> ~98% de horas vacías."""
    df = pd.DataFrame({
        "viaje_id": [1, 2],
        "origen_id": [1, 1],
        "inicio_del_viaje": pd.to_datetime(["2023-01-01 00:00", "2023-01-05 03:00"]),
    })
    limpiador = LimpiadorFiltroEstacionesP1(max_pct_vacias=20).fit(df, df_stations=estaciones)
    assert 1 not in limpiador.estaciones_validas_


def test_limpiador_falla_con_viaje_id_repetido_y_datos_distintos(estaciones):
    df = pd.DataFrame({
        "viaje_id": [1, 1],
        "origen_id": [1, 1],
        "inicio_del_viaje": pd.to_datetime(["2023-01-01 00:00", "2023-01-01 01:00"]),
    })
    with pytest.raises(AssertionError):
        LimpiadorFiltroEstacionesP1().fit(df, df_stations=estaciones).transform(df)


# ---------- GeneradorDemandaLagsP1 ----------

def test_generador_crea_columnas_esperadas(viajes_chicos):
    out = GeneradorDemandaLagsP1().transform(viajes_chicos)
    esperadas = {
        "demand_count", "lag_1h", "lag_24h", "lag_168h", "rolling_3h",
        "rolling_168h", "target_next_hour", "target_hour_sin", "target_hour_cos",
        "target_is_weekend",
    }
    assert esperadas.issubset(out.columns)


def test_generador_sin_nulos_en_columnas_requeridas(viajes_chicos):
    out = GeneradorDemandaLagsP1().transform(viajes_chicos)
    assert not out[["lag_1h", "lag_24h", "lag_168h", "target_next_hour"]].isna().any().any()


def test_generador_rellena_horas_sin_viajes_con_cero():
    """Si una hora no tiene viajes, la malla horaria la incluye con demanda 0."""
    horas = pd.date_range("2023-01-01", periods=400, freq="h")
    horas = horas.delete([250, 251])  # dos horas sin viajes, lejos del inicio
    df = pd.DataFrame({"viaje_id": np.arange(len(horas)), "origen_id": 1, "inicio_del_viaje": horas})
    out = GeneradorDemandaLagsP1().transform(df)
    huecos = out[out["hour_window"].isin(horas[[249]] + pd.to_timedelta(1, unit="h"))]
    assert len(huecos) == 1
    assert huecos["demand_count"].iloc[0] == 0


def test_generador_semantica_de_los_rezagos():
    """Con demanda creciente (hora i -> i viajes) se verifica qué hora mira cada rezago."""
    n_horas = 300
    filas = []
    vid = 0
    for i, h in enumerate(pd.date_range("2023-01-01", periods=n_horas, freq="h")):
        for _ in range(i % 5 + 1):  # demanda entre 1 y 5, patrón conocido
            filas.append({"viaje_id": vid, "origen_id": 1, "inicio_del_viaje": h})
            vid += 1
    df = pd.DataFrame(filas)
    out = GeneradorDemandaLagsP1().transform(df)
    fila = out.iloc[0]
    i = int((fila["hour_window"] - pd.Timestamp("2023-01-01")).total_seconds() // 3600)
    assert fila["demand_count"] == i % 5 + 1
    assert fila["lag_1h"] == i % 5 + 1                 # demanda de la hora actual
    assert fila["lag_24h"] == (i - 23) % 5 + 1         # 24h antes de la hora objetivo (t+1)
    assert fila["lag_168h"] == (i - 167) % 5 + 1       # 168h antes de la hora objetivo
    assert fila["target_next_hour"] == (i + 1) % 5 + 1


def test_generador_variables_ciclicas_en_rango(viajes_chicos):
    out = GeneradorDemandaLagsP1().transform(viajes_chicos)
    assert out["target_hour_sin"].between(-1, 1).all()
    assert out["target_hour_cos"].between(-1, 1).all()
    assert np.allclose(out["target_hour_sin"] ** 2 + out["target_hour_cos"] ** 2, 1)


def test_generador_target_es_la_demanda_de_la_siguiente_hora(viajes_chicos):
    out = GeneradorDemandaLagsP1().transform(viajes_chicos)
    # 1 viaje por hora -> la demanda siempre es 1, el target también
    assert (out["target_next_hour"] == 1).all()


def test_generador_no_modifica_el_dataframe_de_entrada(viajes_chicos):
    copia = viajes_chicos.copy(deep=True)
    GeneradorDemandaLagsP1().transform(viajes_chicos)
    pd.testing.assert_frame_equal(viajes_chicos, copia)


def test_generador_es_determinista(viajes_chicos):
    """Misma entrada -> misma salida (reproducibilidad del feature engineering)."""
    a = GeneradorDemandaLagsP1().transform(viajes_chicos)
    b = GeneradorDemandaLagsP1().transform(viajes_chicos)
    pd.testing.assert_frame_equal(a, b)


def test_generador_con_pocos_datos_devuelve_vacio():
    """Con 10 horas no alcanzan los 168 rezagos -> DataFrame vacío, sin error."""
    horas = pd.date_range("2023-01-01", periods=10, freq="h")
    df = pd.DataFrame({"viaje_id": np.arange(10), "origen_id": 1, "inicio_del_viaje": horas})
    out = GeneradorDemandaLagsP1().transform(df)
    assert out.empty