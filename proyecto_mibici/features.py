#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exporta la API pública de ingeniería de características desde los módulos P1, P2 y P3.
Archivo: proyecto_mibici/features.py
"""

from proyecto_mibici.features_p1 import (
    LimpiadorFiltroEstacionesP1,
    GeneradorDemandaLagsP1,
    construir_dataset_p1,
)
from proyecto_mibici.features_p2 import (
    LimpiadorFlujoP2,
    ConstructorSerieRiesgoP2,
    construir_dataset_p2,
)
from proyecto_mibici.features_p3 import (
    LimpiadorPerfilP3,
    limpiar_viajes_p3,
    construir_perfil_estaciones,
)

__all__ = [
    "LimpiadorFiltroEstacionesP1",
    "GeneradorDemandaLagsP1",
    "construir_dataset_p1",
    "LimpiadorFlujoP2",
    "ConstructorSerieRiesgoP2",
    "construir_dataset_p2",
    "LimpiadorPerfilP3",
    "limpiar_viajes_p3",
    "construir_perfil_estaciones",
]