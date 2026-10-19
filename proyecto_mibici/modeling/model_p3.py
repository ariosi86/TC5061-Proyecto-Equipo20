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
    Asigna identificadores numéricos de grupo según el código de municipio.
    """

    def __init__(self, col_municipio: str = "municipio"):
        self.col_municipio = col_municipio
        self.categorias_: Optional[dict] = None

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        if self.col_municipio in X.columns:
            valores = X[self.col_municipio].dropna().unique()
        elif X.index.name == "id" or "origen_id" in X.columns:
            valores = np.array(["GDL", "ZPN", "TLA", "TON"])
        else:
            valores = np.array([0, 1, 2, 3])
        self.categorias_ = {val: idx for idx, val in enumerate(sorted(valores))}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.col_municipio in X.columns and self.categorias_ is not None:
            codigos = X[self.col_municipio].map(self.categorias_).fillna(0).astype(int)
            return codigos.to_numpy()
        return np.zeros(len(X), dtype=int)


class ClusterKMeansP3(BaseEstimator, ClusterMixin):
    """
    Pipeline de scikit-learn con StandardScaler + KMeans para segmentación funcional.
    """

    def __init__(self, n_clusters: int = 4, n_init: int = 10, random_state: int = 42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "clusterer",
                KMeans(
                    n_clusters=n_clusters,
                    n_init=n_init,
                    random_state=random_state,
                ),
            ),
        ])
        self.features_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        self.features_ = list(X.columns)
        self.pipeline.fit(X[self.features_])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.features_ is None:
            raise ValueError("El modelo debe ser entrenado antes de asignar clusters.")
        return self.pipeline.predict(X[self.features_])

    def fit_predict(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None) -> np.ndarray:
        return self.fit(X, y).predict(X)

    @property
    def labels_(self) -> np.ndarray:
        return self.pipeline.named_steps["clusterer"].labels_

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self.pipeline.named_steps["clusterer"].cluster_centers_

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "ClusterKMeansP3":
        return joblib.load(filepath)


# =====================================================================
# ALIAS CRÍTICO: retrocompatibilidad con scripts que importan ClusterEstacionesP3
# =====================================================================
ClusterEstacionesP3 = ClusterKMeansP3