#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ingeniería de características para P3: Clustering de Estaciones (Uso Funcional).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from proyecto_mibici.config import (
    P3_DURACION_MAX_MIN,
    P3_MIN_VIAJES_ESTACION,
)


class LimpiadorPerfilP3(BaseEstimator, TransformerMixin):
    """Filtra viajes con duraciones válidas y construye el vector de características funcionales."""

    def __init__(self, duracion_max_min: int = P3_DURACION_MAX_MIN, min_viajes: int = P3_MIN_VIAJES_ESTACION):
        self.duracion_max_min = duracion_max_min
        self.min_viajes = min_viajes

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def limpiar_viajes(self, df: pd.DataFrame, df_stations: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.drop_duplicates(subset="viaje_id").copy()
        duracion = (df_clean["fin_del_viaje"] - df_clean["inicio_del_viaje"]).dt.total_seconds() / 60
        df_clean = df_clean.assign(duracion_min=duracion)
        df_clean = df_clean[(df_clean["duracion_min"] > 0) & (df_clean["duracion_min"] <= self.duracion_max_min)]

        ids_validos = set(df_stations["id"])
        return df_clean[df_clean["origen_id"].isin(ids_validos) & df_clean["destino_id"].isin(ids_validos)].reset_index(drop=True)

    def fit_transform_perfil(self, df_viajes_limpios: pd.DataFrame) -> pd.DataFrame:
        hora = df_viajes_limpios["inicio_del_viaje"].dt.hour
        df = df_viajes_limpios.assign(
            rango_horario=pd.cut(
                hora,
                bins=[0, 6, 10, 16, 20, 24],
                right=False,
                labels=["madrugada", "manana", "mediodia", "tarde", "noche"],
            ).astype(str),
            fin_de_semana=df_viajes_limpios["inicio_del_viaje"].dt.dayofweek >= 5,
        )

        salidas = df.groupby("origen_id")
        n_salidas = salidas.size()
        n_llegadas = df.groupby("destino_id").size().reindex(n_salidas.index, fill_value=0)
        n_dias = df["inicio_del_viaje"].dt.normalize().nunique()

        perfil = (
            pd.crosstab(df["origen_id"], df["rango_horario"], normalize="index")
            .add_prefix("pct_")
            .rename_axis(columns=None)
        )
        perfil["pct_fin_de_semana"] = salidas["fin_de_semana"].mean()
        perfil["duracion_media"] = salidas["duracion_min"].mean()
        perfil["viajes_dia_log"] = np.log1p(n_salidas / n_dias)
        perfil["balance_neto"] = (n_llegadas - n_salidas) / (n_llegadas + n_salidas)

        return perfil[n_salidas >= self.min_viajes]


def limpiar_viajes_p3(df: pd.DataFrame, df_stations: pd.DataFrame, duracion_max_min: int = P3_DURACION_MAX_MIN) -> pd.DataFrame:
    """Función de compatibilidad para P3."""
    return LimpiadorPerfilP3(duracion_max_min=duracion_max_min).limpiar_viajes(df, df_stations)


def construir_perfil_estaciones(df: pd.DataFrame, min_viajes: int = P3_MIN_VIAJES_ESTACION) -> pd.DataFrame:
    """Función de compatibilidad para construir el perfil funcional en P3."""
    return LimpiadorPerfilP3(min_viajes=min_viajes).fit_transform_perfil(df)