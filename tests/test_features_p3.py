"""Pruebas unitarias de la ingeniería de características de P3 (clustering de estaciones).

Usan datos sintéticos, por lq que no leen los Parquet ni se necesita MLflow.
"""

import numpy as np
import pandas as pd
import pytest

from proyecto_mibici.config import P3_VARIABLES
from proyecto_mibici.features_p3 import construir_perfil_estaciones, limpiar_viajes_p3


def hacer_viajes(filas):
    """Crea un DataFrame de viajes a partir de tuplas (viaje_id, inicio, fin, origen, destino)."""
    df = pd.DataFrame(
        filas, columns=["viaje_id", "inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"]
    )
    df["inicio_del_viaje"] = pd.to_datetime(df["inicio_del_viaje"])
    df["fin_del_viaje"] = pd.to_datetime(df["fin_del_viaje"])
    return df


@pytest.fixture
def estaciones():
    """Catálogo sintético con tres estaciones válidas."""
    return pd.DataFrame({"id": [1, 2, 3]})


# ---------------------------------------------------------------------------
# limpiar_viajes_p3
# ---------------------------------------------------------------------------
def test_limpiar_calcula_duracion_en_minutos(estaciones):
    viajes = hacer_viajes([(1, "2025-01-06 08:00", "2025-01-06 08:25", 1, 2)])

    limpios = limpiar_viajes_p3(viajes, estaciones)

    assert limpios["duracion_min"].iloc[0] == 25


def test_limpiar_quita_viajes_duplicados(estaciones):
    viaje = (1, "2025-01-06 08:00", "2025-01-06 08:20", 1, 2)
    viajes = hacer_viajes([viaje, viaje])  # mismo viaje_id dos veces

    limpios = limpiar_viajes_p3(viajes, estaciones)

    assert len(limpios) == 1


def test_limpiar_respeta_los_limites_de_duracion(estaciones):
    viajes = hacer_viajes([
        (1, "2025-01-06 08:00", "2025-01-06 08:00", 1, 2),  # 0 min: se quita
        (2, "2025-01-06 08:00", "2025-01-06 11:00", 1, 2),  # 180 min exactos: se conserva
        (3, "2025-01-06 08:00", "2025-01-06 11:01", 1, 2),  # 181 min: se quita
    ])

    limpios = limpiar_viajes_p3(viajes, estaciones, duracion_max_min=180)

    assert list(limpios["viaje_id"]) == [2]


def test_limpiar_quita_estaciones_fuera_del_catalogo(estaciones):
    viajes = hacer_viajes([
        (1, "2025-01-06 08:00", "2025-01-06 08:20", 1, 2),   # ambas existen
        (2, "2025-01-06 08:00", "2025-01-06 08:20", 1, 99),  # destino desconocido
        (3, "2025-01-06 08:00", "2025-01-06 08:20", 99, 1),  # origen desconocido
    ])

    limpios = limpiar_viajes_p3(viajes, estaciones)

    assert list(limpios["viaje_id"]) == [1]


# ---------------------------------------------------------------------------
# construir_perfil_estaciones: caso normal
# ---------------------------------------------------------------------------
@pytest.fixture
def viajes_limpios(estaciones):
    """Estación 1: tres salidas entre semana. Estación 2: una salida en sábado por la noche."""
    viajes = hacer_viajes([
        (1, "2025-01-06 08:00", "2025-01-06 08:20", 1, 2),  # lunes, mañana
        (2, "2025-01-06 12:00", "2025-01-06 12:10", 1, 2),  # lunes, mediodía
        (3, "2025-01-06 18:00", "2025-01-06 18:30", 1, 3),  # lunes, tarde
        (4, "2025-01-04 21:00", "2025-01-04 21:40", 2, 1),  # sábado, noche
    ])
    return limpiar_viajes_p3(viajes, estaciones)


def test_perfil_tiene_todas_las_variables_del_modelo(viajes_limpios):
    perfil = construir_perfil_estaciones(viajes_limpios, min_viajes=1)

    assert set(P3_VARIABLES) <= set(perfil.columns)
    assert perfil.index.is_unique  # una fila por estación


def test_perfil_proporciones_horarias_suman_uno(viajes_limpios):
    perfil = construir_perfil_estaciones(viajes_limpios, min_viajes=1)
    columnas_pct = [c for c in perfil.columns if c.startswith("pct_") and c != "pct_fin_de_semana"]

    assert np.allclose(perfil[columnas_pct].sum(axis=1), 1.0)


def test_perfil_valores_de_la_estacion(viajes_limpios):
    perfil = construir_perfil_estaciones(viajes_limpios, min_viajes=1)
    est1 = perfil.loc[1]
    est2 = perfil.loc[2]

    assert est1["pct_manana"] == pytest.approx(1 / 3)
    assert est1["pct_fin_de_semana"] == 0
    assert est2["pct_fin_de_semana"] == 1
    assert est1["duracion_media"] == pytest.approx(20)  # (20 + 10 + 30) / 3
    # Estación 1: 3 salidas y 1 llegada -> (1 - 3) / (1 + 3)
    assert est1["balance_neto"] == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# construir_perfil_estaciones: casos límite
# ---------------------------------------------------------------------------
def test_perfil_rangos_de_valores(viajes_limpios):
    perfil = construir_perfil_estaciones(viajes_limpios, min_viajes=1)

    assert perfil["pct_fin_de_semana"].between(0, 1).all()
    assert perfil["balance_neto"].between(-1, 1).all()
    assert (perfil["viajes_dia_log"] > 0).all()


def test_perfil_descarta_estaciones_con_pocos_viajes(viajes_limpios):
    # La estación 1 tiene 3 salidas y la 2 solo 1
    perfil = construir_perfil_estaciones(viajes_limpios, min_viajes=2)

    assert list(perfil.index) == [1]


def test_perfil_con_un_rango_horario_sin_viajes(estaciones):
    # Ningún viaje de noche: la columna pct_noche debe existir en 0
    viajes = hacer_viajes([
        (1, "2025-01-06 08:00", "2025-01-06 08:20", 1, 2),
        (2, "2025-01-06 12:00", "2025-01-06 12:10", 2, 1),
    ])

    perfil = construir_perfil_estaciones(limpiar_viajes_p3(viajes, estaciones), min_viajes=1)

    assert (perfil["pct_noche"] == 0).all()
    assert not perfil[P3_VARIABLES].isna().any().any()