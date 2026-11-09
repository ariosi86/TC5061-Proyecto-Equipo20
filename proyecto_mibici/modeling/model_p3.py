#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clases orientadas a objetos para los modelos de P3 (Segmentación de Estaciones).
Incluye la línea base geográfica por municipio y el pipeline KMeans con StandardScaler.
"""

from pathlib import Path
from typing import List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BaselineMunicipioP3(BaseEstimator, ClusterMixin):
    """
    Línea base no supervisada para P3 basada en demarcación municipal:
    Asigna identificadores numéricos de grupo según el código de municipio de cada estación.
    """

    def __init__(self, col_municipio: str = "municipality_code"):
        self.col_municipio = col_municipio

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        df_stations: Optional[pd.DataFrame] = None,
    ):
        # 1. Si la columna viene dentro del DataFrame
        if self.col_municipio in X.columns:
            series_mun = X[self.col_municipio]
        # 2. Si se proporciona el catálogo de estaciones para cruzar por el índice
        elif df_stations is not None and "id" in df_stations.columns and self.col_municipio in df_stations.columns:
            mapa = df_stations.set_index("id")[self.col_municipio]
            series_mun = X.index.map(mapa)
        else:
            raise KeyError(
                f"No se encontró '{self.col_municipio}' en las columnas de entrada ni catálogo disponible."
            )

        unicos = sorted(pd.Series(series_mun).dropna().unique())
        self.categorias_ = {val: idx for idx, val in enumerate(unicos)}
        return self

    def predict(
        self, X: pd.DataFrame, df_stations: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        if not hasattr(self, "categorias_"):
            raise ValueError("El modelo debe ajustarse antes de predecir.")

        if self.col_municipio in X.columns:
            series_mun = X[self.col_municipio]
        elif df_stations is not None and "id" in df_stations.columns:
            mapa = df_stations.set_index("id")[self.col_municipio]
            series_mun = X.index.map(mapa)
        else:
            raise KeyError("Se requiere la columna de municipio o catálogo df_stations para inferir.")

        return series_mun.map(self.categorias_).fillna(-1).astype(int).to_numpy()

    def fit_predict(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        df_stations: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        return self.fit(X, y, df_stations=df_stations).predict(X, df_stations=df_stations)


class ClusterKMeansP3(BaseEstimator, ClusterMixin):
    """
    Pipeline de scikit-learn con StandardScaler + KMeans para segmentación funcional.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        n_init: int = 10,
        random_state: int = 42,
        features: Optional[List[str]] = None,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state
        self.features = features

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        self.features_ = (
            list(self.features) if self.features is not None else list(X.columns)
        )
        self.pipeline_ = Pipeline([
            ("scaler", StandardScaler()),
            (
                "clusterer",
                KMeans(
                    n_clusters=self.n_clusters,
                    n_init=self.n_init,
                    random_state=self.random_state,
                ),
            ),
        ])
        self.pipeline_.fit(X[self.features_])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "pipeline_"):
            raise ValueError("El modelo debe ser entrenado antes de asignar clusters.")
        return self.pipeline_.predict(X[self.features_])

    def fit_predict(
        self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> np.ndarray:
        return self.fit(X, y).predict(X)

    @property
    def labels_(self) -> np.ndarray:
        return self.pipeline_.named_steps["clusterer"].labels_
    @property
    def pipeline(self):
        return getattr(self, "pipeline_", None)
    @property
    def cluster_centers_(self) -> np.ndarray:
        return self.pipeline_.named_steps["clusterer"].cluster_centers_

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ClusterKMeansP3":
        return joblib.load(Path(filepath))


# Retrocompatibilidad
ClusterEstacionesP3 = ClusterKMeansP3