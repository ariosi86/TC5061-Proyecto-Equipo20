#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ingeniería de características para P1: Demanda Horaria (Regresión).
"""

from typing import Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from proyecto_mibici.config import (
    P1_MAX_PCT_VACIAS,
    STATIONS_PROCESSED_PATH,
    TRIPS_PROCESSED_PATH,
)


class LimpiadorFiltroEstacionesP1(BaseEstimator, TransformerMixin):
    """Elimina duplicados y descarta estaciones inactivas o fuera de catálogo."""

    def __init__(self, max_pct_vacias: float = P1_MAX_PCT_VACIAS):
        self.max_pct_vacias = max_pct_vacias
        self.estaciones_validas_: Optional[Set[int]] = None

    def fit(self, X: pd.DataFrame, y=None, df_stations: Optional[pd.DataFrame] = None):
        if df_stations is None:
            df_stations = pd.read_parquet(STATIONS_PROCESSED_PATH)

        df_clean = X.drop_duplicates()
        df = df_clean.assign(hour_window=df_clean["inicio_del_viaje"].dt.floor("h"))
        cov = df.groupby("origen_id").agg(
            first_hour=("hour_window", "min"),
            last_hour=("hour_window", "max"),
            observed_hours=("hour_window", "nunique"),
        )
        expected = ((cov["last_hour"] - cov["first_hour"]) / pd.Timedelta(hours=1)).astype(int) + 1
        pct_vacias = (expected - cov["observed_hours"]) / expected * 100

        validas = pct_vacias[pct_vacias <= self.max_pct_vacias].index
        self.estaciones_validas_ = set(validas.intersection(df_stations["id"]))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_clean = X.drop_duplicates().copy()
        assert df_clean["viaje_id"].is_unique, "Hay viaje_id repetidos con datos distintos."
        if self.estaciones_validas_ is not None:
            return df_clean[df_clean["origen_id"].isin(self.estaciones_validas_)].reset_index(drop=True)
        return df_clean.reset_index(drop=True)


class GeneradorDemandaLagsP1(BaseEstimator, TransformerMixin):
    """Construye la malla horaria completa, rezagos, rolling means y variables cíclicas."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.assign(hour_window=pd.to_datetime(X["inicio_del_viaje"]).dt.floor("h"))

        df_demand = (
            df.groupby(["origen_id", "hour_window"])
            .size()
            .rename("demand_count")
            .reset_index()
        )

        df_bounds = df_demand.groupby("origen_id")["hour_window"].agg(["min", "max"])
        grid = pd.concat([
            pd.DataFrame({
                "origen_id": sid,
                "hour_window": pd.date_range(r["min"], r["max"], freq="h"),
            })
            for sid, r in df_bounds.iterrows()
        ])

        df_out = grid.merge(df_demand, on=["origen_id", "hour_window"], how="left")
        df_out["demand_count"] = df_out["demand_count"].fillna(0).astype(int)
        df_out = df_out.sort_values(["origen_id", "hour_window"]).reset_index(drop=True)

        por_estacion = df_out.groupby("origen_id")["demand_count"]
        df_out["lag_1h"] = df_out["demand_count"]
        df_out["lag_24h"] = por_estacion.shift(23)
        df_out["lag_168h"] = por_estacion.shift(167)

        df_out["rolling_3h"] = (
            por_estacion.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df_out["rolling_168h"] = (
            por_estacion.rolling(168, min_periods=24).mean().reset_index(level=0, drop=True)
        )

        df_out["target_next_hour"] = por_estacion.shift(-1)
        next_hour = df_out.groupby("origen_id")["hour_window"].shift(-1)
        valid_next_hour = next_hour == df_out["hour_window"] + pd.Timedelta(hours=1)
        df_out.loc[~valid_next_hour, "target_next_hour"] = np.nan

        # Calendario y variables cíclicas
        df_out["target_hour"] = df_out["hour_window"] + pd.Timedelta(hours=1)
        df_out["target_hour_of_day"] = df_out["target_hour"].dt.hour
        df_out["target_dayofweek"] = df_out["target_hour"].dt.dayofweek
        df_out["target_is_weekend"] = df_out["target_dayofweek"].isin([5, 6]).astype(int)
        df_out["target_hour_sin"] = np.sin(2 * np.pi * df_out["target_hour_of_day"] / 24)
        df_out["target_hour_cos"] = np.cos(2 * np.pi * df_out["target_hour_of_day"] / 24)

        required_columns = ["lag_1h", "lag_24h", "lag_168h", "target_next_hour"]
        return df_out.dropna(subset=required_columns).reset_index(drop=True)


def construir_dataset_p1() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline completo de P1: parquets procesados -> tabla de modelado."""
    df_trips = pd.read_parquet(TRIPS_PROCESSED_PATH)
    df_stations = pd.read_parquet(STATIONS_PROCESSED_PATH)

    limpiador = LimpiadorFiltroEstacionesP1().fit(df_trips, df_stations=df_stations)
    trips_filtrados = limpiador.transform(df_trips)
    generador = GeneradorDemandaLagsP1().fit(trips_filtrados)
    df_p1 = generador.transform(trips_filtrados)
    return df_p1, df_stations