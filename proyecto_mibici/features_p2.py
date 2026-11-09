#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ingeniería de características para P2: Riesgo de Desbalance (Clasificación).
"""

from typing import Optional, Set
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from proyecto_mibici.config import (
    P2_FIN_PERIODO,
    P2_INICIO_VALIDACION,
    P2_INICIO_PRUEBA,
    P2_MES_ESTACION_NUEVA,
    P2_HORA_INICIO_DIA,
    P2_UMBRAL,
    P2_OBJETIVO,
    STATIONS_PROCESSED_PATH,
    TRIPS_PROCESSED_PATH,
)


class LimpiadorFlujoP2(BaseEstimator, TransformerMixin):
    """Filtra anomalías (<10s distintas estaciones) y genera flujos por estación y hora."""

    def __init__(self, fin_periodo: str = P2_FIN_PERIODO):
        self.fin_periodo = pd.Timestamp(fin_periodo)

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        duracion_seg = (df["fin_del_viaje"] - df["inicio_del_viaje"]).dt.total_seconds()
        misma_estacion = df["origen_id"] == df["destino_id"]

        cols_movimiento = ["inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"]
        es_copia = df.duplicated(subset=cols_movimiento)
        es_imposible = (duracion_seg < 10) & (~misma_estacion)
        viajes_limpios = df[~(es_copia | es_imposible)].reset_index(drop=True)

        salidas = viajes_limpios.groupby([viajes_limpios["origen_id"], viajes_limpios["inicio_del_viaje"].dt.floor("h")]).size()
        llegadas = viajes_limpios.groupby([viajes_limpios["destino_id"], viajes_limpios["fin_del_viaje"].dt.floor("h")]).size()
        salidas.index.names = ["estacion_id", "hora"]
        llegadas.index.names = ["estacion_id", "hora"]

        flujo = pd.concat([salidas.rename("salidas"), llegadas.rename("llegadas")], axis=1)
        flujo = flujo.fillna(0).astype(int).reset_index()
        return flujo[flujo["hora"] <= self.fin_periodo].reset_index(drop=True)


class ConstructorSerieRiesgoP2(BaseEstimator, TransformerMixin):
    """Completa la serie continua por estación, día operativo (03:00) y calcula riesgo."""

    def __init__(
        self,
        umbral: float = P2_UMBRAL,
        hora_inicio_dia: int = P2_HORA_INICIO_DIA,
        fin_periodo: str = P2_FIN_PERIODO,
    ):
        self.umbral = umbral
        self.hora_inicio_dia = hora_inicio_dia
        self.fin_periodo = pd.Timestamp(fin_periodo)
        self.modelables_: Optional[Set[int]] = None
        self.nuevas_: Optional[Set[int]] = None

    def fit(self, flujo: pd.DataFrame, y=None, df_trips: Optional[pd.DataFrame] = None, df_stations: Optional[pd.DataFrame] = None):
        if df_trips is None:
            df_trips = pd.read_parquet(TRIPS_PROCESSED_PATH)
        if df_stations is None:
            df_stations = pd.read_parquet(STATIONS_PROCESSED_PATH)

        movimientos = pd.concat([
            df_trips[["origen_id", "periodo"]].rename(columns={"origen_id": "estacion_id"}),
            df_trips[["destino_id", "periodo"]].rename(columns={"destino_id": "estacion_id"}),
        ])
        primer_mes = movimientos.groupby("estacion_id")["periodo"].min()

        en_viajes = set(primer_mes.index)
        con_capacidad = set(df_stations.loc[df_stations["dpcapacity"] > 0, "id"])
        self.modelables_ = en_viajes & con_capacidad
        self.nuevas_ = set(primer_mes.index[primer_mes >= P2_MES_ESTACION_NUEVA])
        return self

    def transform(self, flujo: pd.DataFrame, df_stations: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df_stations is None:
            df_stations = pd.read_parquet(STATIONS_PROCESSED_PATH)

        flujo_mod = flujo[flujo["estacion_id"].isin(self.modelables_)].copy()
        rango = flujo_mod.groupby("estacion_id")["hora"].agg(primera_hora="min", ultima_hora="max")

        horas = pd.date_range(flujo_mod["hora"].min(), self.fin_periodo, freq="h")
        indice = pd.MultiIndex.from_product([sorted(self.modelables_), horas], names=["estacion_id", "hora"])
        serie = (
            flujo_mod.set_index(["estacion_id", "hora"])[["salidas", "llegadas"]]
            .reindex(indice, fill_value=0)
            .reset_index()
        )

        serie = serie.merge(rango, on="estacion_id")
        dentro = (serie["hora"] >= serie["primera_hora"]) & (serie["hora"] <= serie["ultima_hora"])
        serie = serie[dentro].drop(columns=["primera_hora", "ultima_hora"]).reset_index(drop=True)

        estaciones_map = df_stations[["id", "dpcapacity"]].rename(
            columns={"id": "estacion_id", "dpcapacity": "capacidad"}
        )
        serie = serie.merge(estaciones_map, on="estacion_id", how="left")

        serie["balance"] = serie["llegadas"] - serie["salidas"]
        serie["dia_operativo"] = (serie["hora"] - pd.Timedelta(hours=self.hora_inicio_dia)).dt.floor("D")
        serie["balance_acumulado_dia"] = serie.groupby(["estacion_id", "dia_operativo"])["balance"].cumsum()

        serie["riesgo_vacia"] = serie["balance_acumulado_dia"] <= -self.umbral * serie["capacidad"]
        serie["riesgo_llena"] = serie["balance_acumulado_dia"] >= self.umbral * serie["capacidad"]
        serie["en_riesgo"] = serie["riesgo_vacia"] | serie["riesgo_llena"]

        # Variables predictoras y desfase temporal t+1
        por_estacion = serie.sort_values(["estacion_id", "hora"]).groupby("estacion_id")
        serie["hora_del_dia"] = serie["hora"].dt.hour
        serie["dia_semana"] = serie["hora"].dt.dayofweek
        serie["fin_de_semana"] = (serie["dia_semana"] >= 5).astype(int)

        serie["acumulado_relativo"] = serie["balance_acumulado_dia"] / serie["capacidad"]
        serie["balance_ult_3h_relativo"] = (
            por_estacion["balance"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
            / serie["capacidad"]
        )

        serie[P2_OBJETIVO] = por_estacion["en_riesgo"].shift(-1)
        serie["hora_objetivo"] = serie["hora"] + pd.Timedelta(hours=1)
        serie = serie.dropna(subset=[P2_OBJETIVO]).reset_index(drop=True)
        serie[P2_OBJETIVO] = serie[P2_OBJETIVO].astype(bool)

        inicio_val = pd.Timestamp(P2_INICIO_VALIDACION)
        inicio_tst = pd.Timestamp(P2_INICIO_PRUEBA)

        serie["conjunto"] = np.select(
            [serie["hora_objetivo"] < inicio_val, serie["hora_objetivo"] < inicio_tst],
            ["entrenamiento", "validacion"],
            default="prueba",
        )
        serie["estacion_nueva"] = serie["estacion_id"].isin(self.nuevas_)
        return serie


def construir_dataset_p2() -> pd.DataFrame:
    """Pipeline completo de P2: parquets procesados -> tabla de modelado con particiones."""
    df_trips = pd.read_parquet(TRIPS_PROCESSED_PATH)
    df_stations = pd.read_parquet(STATIONS_PROCESSED_PATH)

    flujo = LimpiadorFlujoP2().fit_transform(df_trips)
    constructor = ConstructorSerieRiesgoP2().fit(flujo, df_trips=df_trips, df_stations=df_stations)
    return constructor.transform(flujo, df_stations=df_stations)