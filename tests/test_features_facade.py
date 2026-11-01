"""Pruebas simples para la fachada de características (features.py).

Verifica que el módulo features expone correctamente la API pública de ingeniería
de características para P1, P2 y P3.
"""
import pytest

from proyecto_mibici import features


def test_features_expone_api_publica_p1():
    """La fachada expone las clases y funciones de P1."""
    assert hasattr(features, "LimpiadorFiltroEstacionesP1")  # nosec B101
    assert hasattr(features, "GeneradorDemandaLagsP1")  # nosec B101
    assert hasattr(features, "construir_dataset_p1")  # nosec B101


def test_features_expone_api_publica_p2():
    """La fachada expone las clases y funciones de P2."""
    assert hasattr(features, "LimpiadorFlujoP2")  # nosec B101
    assert hasattr(features, "ConstructorSerieRiesgoP2")  # nosec B101
    assert hasattr(features, "construir_dataset_p2")  # nosec B101


def test_features_expone_api_publica_p3():
    """La fachada expone las clases y funciones de P3."""
    assert hasattr(features, "LimpiadorPerfilP3")  # nosec B101
    assert hasattr(features, "limpiar_viajes_p3")  # nosec B101
    assert hasattr(features, "construir_perfil_estaciones")  # nosec B101


def test_features_todos_los_elementos_son_importables():
    """Cada elemento listado en __all__ se puede importar directamente de features."""
    for nombre in features.__all__:
        assert hasattr(features, nombre), f"Falta exportar {nombre} en features.py"  # nosec B101
