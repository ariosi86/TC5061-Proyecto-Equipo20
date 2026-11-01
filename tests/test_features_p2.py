"""Pruebas unitarias de la ingeniería de características de P2 (riesgo de desbalance).

Usan datos sintéticos pequeños: no leen los Parquet reales ni necesitan MLflow.
"""

import pandas as pd
import pytest

from proyecto_mibici.config import P2_INICIO_VALIDACION, P2_MES_ESTACION_NUEVA, P2_OBJETIVO
from proyecto_mibici.features_p2 import ConstructorSerieRiesgoP2, LimpiadorFlujoP2

# Corte de periodo fijo para las pruebas (independiente del de config.py)
FIN_PRUEBA = "2025-01-01 23:00"


def hacer_viajes(filas):
    """Arma un DataFrame de viajes a partir de tuplas (inicio, fin, origen, destino)."""
    df = pd.DataFrame(
        filas, columns=["inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"]
    )
    df["inicio_del_viaje"] = pd.to_datetime(df["inicio_del_viaje"])
    df["fin_del_viaje"] = pd.to_datetime(df["fin_del_viaje"])
    return df


def flujo_de(flujo, estacion, hora):
    """Devuelve (salidas, llegadas) de una estación en una hora; (0, 0) si no aparece."""
    fila = flujo[(flujo["estacion_id"] == estacion) & (flujo["hora"] == pd.Timestamp(hora))]
    if fila.empty:
        return (0, 0)
    return (int(fila["salidas"].iloc[0]), int(fila["llegadas"].iloc[0]))


# ---------------------------------------------------------------------------
# LimpiadorFlujoP2: caso normal
# ---------------------------------------------------------------------------
def test_limpiador_cuenta_salidas_y_llegadas_por_hora():
    viajes = hacer_viajes([
        ("2025-01-01 08:10", "2025-01-01 08:30", 1, 2),
        ("2025-01-01 08:40", "2025-01-01 09:05", 1, 3),
        ("2025-01-01 09:15", "2025-01-01 09:20", 2, 1),
    ])

    flujo = LimpiadorFlujoP2(fin_periodo=FIN_PRUEBA).fit_transform(viajes)

    assert list(flujo.columns) == ["estacion_id", "hora", "salidas", "llegadas"]
    assert flujo_de(flujo, 1, "2025-01-01 08:00") == (2, 0)
    assert flujo_de(flujo, 1, "2025-01-01 09:00") == (0, 1)
    assert flujo_de(flujo, 2, "2025-01-01 08:00") == (0, 1)
    assert flujo_de(flujo, 2, "2025-01-01 09:00") == (1, 0)
    assert flujo_de(flujo, 3, "2025-01-01 09:00") == (0, 1)
    # Cada viaje aporta exactamente una salida y una llegada
    assert flujo["salidas"].sum() == 3
    assert flujo["llegadas"].sum() == 3


# ---------------------------------------------------------------------------
# LimpiadorFlujoP2: casos límite
# ---------------------------------------------------------------------------
def test_limpiador_quita_viajes_duplicados():
    viaje = ("2025-01-01 08:10", "2025-01-01 08:30", 1, 2)
    viajes = hacer_viajes([viaje, viaje])  # el mismo viaje dos veces

    flujo = LimpiadorFlujoP2(fin_periodo=FIN_PRUEBA).fit_transform(viajes)

    assert flujo_de(flujo, 1, "2025-01-01 08:00") == (1, 0)
    assert flujo_de(flujo, 2, "2025-01-01 08:00") == (0, 1)


def test_limpiador_quita_viajes_imposibles_entre_estaciones_distintas():
    # 5 segundos de la estación 1 a la 2: físicamente imposible
    viajes = hacer_viajes([("2025-01-01 08:10:00", "2025-01-01 08:10:05", 1, 2)])

    flujo = LimpiadorFlujoP2(fin_periodo=FIN_PRUEBA).fit_transform(viajes)

    assert flujo.empty


def test_limpiador_conserva_viajes_cortos_en_la_misma_estacion():
    # 5 segundos saliendo y regresando a la estación 1: se conserva
    viajes = hacer_viajes([("2025-01-01 08:10:00", "2025-01-01 08:10:05", 1, 1)])

    flujo = LimpiadorFlujoP2(fin_periodo=FIN_PRUEBA).fit_transform(viajes)

    assert flujo_de(flujo, 1, "2025-01-01 08:00") == (1, 1)


def test_limpiador_conserva_viaje_de_exactamente_10_segundos():
    # El filtro es "menos de 10 s"; 10 s exactos se conservan
    viajes = hacer_viajes([("2025-01-01 08:10:00", "2025-01-01 08:10:10", 1, 2)])

    flujo = LimpiadorFlujoP2(fin_periodo=FIN_PRUEBA).fit_transform(viajes)

    assert flujo_de(flujo, 1, "2025-01-01 08:00") == (1, 0)
    assert flujo_de(flujo, 2, "2025-01-01 08:00") == (0, 1)


def test_limpiador_descarta_horas_posteriores_al_fin_del_periodo():
    viajes = hacer_viajes([
        ("2025-01-01 22:10", "2025-01-01 22:30", 1, 2),  # dentro del periodo
        ("2025-01-01 23:50", "2025-01-02 00:10", 1, 2),  # la llegada cae después del corte
    ])

    flujo = LimpiadorFlujoP2(fin_periodo=FIN_PRUEBA).fit_transform(viajes)

    assert flujo["hora"].max() <= pd.Timestamp(FIN_PRUEBA)
    assert flujo_de(flujo, 1, "2025-01-01 23:00") == (1, 0)   # la salida de las 23:50 sí cuenta
    assert flujo_de(flujo, 2, "2025-01-02 00:00") == (0, 0)   # su llegada queda fuera


# ===========================================================================
# ConstructorSerieRiesgoP2
# ===========================================================================
@pytest.fixture
def estaciones():
    """Catálogo sintético: 1 y 2 con capacidad 10; la 9 sin capacidad."""
    return pd.DataFrame({"id": [1, 2, 9], "dpcapacity": [10, 10, 0]})


@pytest.fixture
def viajes_fit():
    """Viajes mínimos para fit: la 1 es antigua; la 2 aparece en el mes de estación nueva."""
    return pd.DataFrame({
        "origen_id": [1, 2, 9],
        "destino_id": [1, 2, 9],
        "periodo": ["2000_01", P2_MES_ESTACION_NUEVA, "2000_01"],
    })


def hacer_flujo(filas):
    """Arma un flujo a partir de tuplas (estacion_id, hora, salidas, llegadas)."""
    df = pd.DataFrame(filas, columns=["estacion_id", "hora", "salidas", "llegadas"])
    df["hora"] = pd.to_datetime(df["hora"])
    return df


def construir_serie(flujo, viajes_fit, estaciones, fin_periodo):
    constructor = ConstructorSerieRiesgoP2(umbral=0.5, hora_inicio_dia=3, fin_periodo=fin_periodo)
    constructor.fit(flujo, df_trips=viajes_fit, df_stations=estaciones)
    return constructor.transform(flujo, df_stations=estaciones)


def fila(serie, estacion, hora):
    return serie[(serie["estacion_id"] == estacion) & (serie["hora"] == pd.Timestamp(hora))].iloc[0]


# ---------------------------------------------------------------------------
# ConstructorSerieRiesgoP2: caso normal
# ---------------------------------------------------------------------------
def test_constructor_rellena_horas_sin_movimiento_y_acumula_balance(viajes_fit, estaciones):
    flujo = hacer_flujo([
        (1, "2025-01-01 04:00", 0, 2),   # llegan 2
        (1, "2025-01-01 06:00", 1, 0),   # sale 1 (a las 05:00 no hubo movimiento)
    ])

    serie = construir_serie(flujo, viajes_fit, estaciones, "2025-01-01 08:00")
    hora_vacia = fila(serie, 1, "2025-01-01 05:00")

    assert hora_vacia["salidas"] == 0 and hora_vacia["llegadas"] == 0
    assert hora_vacia["balance_acumulado_dia"] == 2
    assert hora_vacia["capacidad"] == 10
    assert hora_vacia["acumulado_relativo"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# ConstructorSerieRiesgoP2: casos límite
# ---------------------------------------------------------------------------
def test_constructor_riesgo_exactamente_en_el_umbral(viajes_fit, estaciones):
    # Capacidad 10 y umbral 50 %: el riesgo empieza en un acumulado de 5 (o -5)
    flujo = hacer_flujo([
        (1, "2025-01-01 04:00", 0, 4),   # acumulado +4: sin riesgo
        (1, "2025-01-01 05:00", 0, 1),   # acumulado +5: riesgo de llenarse
        (1, "2025-01-01 06:00", 0, 0),
        (2, "2025-01-01 04:00", 5, 0),   # acumulado -5: riesgo de vaciarse
        (2, "2025-01-01 05:00", 0, 0),
    ])

    serie = construir_serie(flujo, viajes_fit, estaciones, "2025-01-01 08:00")

    assert not fila(serie, 1, "2025-01-01 04:00")["en_riesgo"]
    assert fila(serie, 1, "2025-01-01 05:00")["riesgo_llena"]
    assert fila(serie, 2, "2025-01-01 04:00")["riesgo_vacia"]


def test_constructor_objetivo_es_el_riesgo_de_la_hora_siguiente(viajes_fit, estaciones):
    flujo = hacer_flujo([
        (1, "2025-01-01 04:00", 0, 4),   # sin riesgo ahora...
        (1, "2025-01-01 05:00", 0, 1),   # ...pero sí en la hora siguiente
        (1, "2025-01-01 06:00", 0, 0),
    ])

    serie = construir_serie(flujo, viajes_fit, estaciones, "2025-01-01 08:00")
    a_las_4 = fila(serie, 1, "2025-01-01 04:00")

    assert not a_las_4["en_riesgo"]
    assert a_las_4[P2_OBJETIVO]
    # La última hora de la estación no tiene "hora siguiente" y se descarta
    assert pd.Timestamp("2025-01-01 06:00") not in set(serie.loc[serie["estacion_id"] == 1, "hora"])


def test_constructor_reinicia_el_acumulado_a_las_3_am(viajes_fit, estaciones):
    flujo = hacer_flujo([
        (1, "2025-01-01 02:00", 0, 3),   # todavía es el día operativo anterior
        (1, "2025-01-01 03:00", 0, 3),   # empieza un día operativo nuevo
        (1, "2025-01-01 04:00", 0, 0),
    ])

    serie = construir_serie(flujo, viajes_fit, estaciones, "2025-01-01 08:00")

    assert fila(serie, 1, "2025-01-01 02:00")["balance_acumulado_dia"] == 3
    assert fila(serie, 1, "2025-01-01 03:00")["balance_acumulado_dia"] == 3   # no 6


def test_constructor_excluye_estaciones_sin_capacidad(viajes_fit, estaciones):
    flujo = hacer_flujo([
        (1, "2025-01-01 04:00", 0, 1),
        (1, "2025-01-01 05:00", 0, 0),
        (9, "2025-01-01 04:00", 0, 1),   # la 9 tiene capacidad 0
        (9, "2025-01-01 05:00", 0, 0),
    ])

    serie = construir_serie(flujo, viajes_fit, estaciones, "2025-01-01 08:00")

    assert 9 not in set(serie["estacion_id"])
    assert 1 in set(serie["estacion_id"])


def test_constructor_marca_estaciones_nuevas(viajes_fit, estaciones):
    flujo = hacer_flujo([
        (1, "2025-01-01 04:00", 0, 1),
        (1, "2025-01-01 05:00", 0, 0),
        (2, "2025-01-01 04:00", 0, 1),
        (2, "2025-01-01 05:00", 0, 0),
    ])

    serie = construir_serie(flujo, viajes_fit, estaciones, "2025-01-01 08:00")

    assert not fila(serie, 1, "2025-01-01 04:00")["estacion_nueva"]
    assert fila(serie, 2, "2025-01-01 04:00")["estacion_nueva"]


def test_constructor_particion_temporal_sin_fuga(viajes_fit, estaciones):
    # La partición se decide por la hora del objetivo (hora + 1), no por la hora actual
    inicio_val = pd.Timestamp(P2_INICIO_VALIDACION)
    horas = [inicio_val - pd.Timedelta(hours=h) for h in (2, 1, 0)]
    flujo = hacer_flujo([(1, h, 0, 0) for h in horas] + [(1, inicio_val + pd.Timedelta(hours=1), 0, 0)])

    serie = construir_serie(flujo, viajes_fit, estaciones, inicio_val + pd.Timedelta(hours=3))

    assert fila(serie, 1, horas[0])["conjunto"] == "entrenamiento"   # objetivo una hora antes del corte
    assert fila(serie, 1, horas[1])["conjunto"] == "validacion"      # objetivo justo en el corte
