#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clases orientadas a objetos para los modelos de P1 (Demanda Horaria de Viajes).
Incluye baselines ingenuos y el modelo de Regresión Lineal con StandardScaler.
"""

from pathlib import Path
from typing import List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from proyecto_mibici.config import ESTRATEGIAS_BASELINE_P1


class BaselineIngenuoP1(BaseEstimator, RegressorMixin):
    """
    Modelos de referencia heurísticos para la demanda horaria (P1).
    Estrategias soportadas:
      - 'persistencia': Demanda observada en la hora anterior (t-1).
      - 'lag_24h': Demanda observada exactamente hace 24 horas (t-24).
      - 'lag_168h': Demanda observada la misma hora la semana anterior (t-168).
      - 'promedio_24h_168h': Promedio simple entre lag 24h y lag 168h.
    """

    def __init__(self, estrategia: str = "persistencia"):
        self.estrategia = estrategia

        if self.estrategia not in ESTRATEGIAS_BASELINE_P1:
            raise ValueError(
                f"Estrategia '{estrategia}' no válida. Opciones: {ESTRATEGIAS_BASELINE_P1}"
            )
        self.estrategia = estrategia

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.estrategia == "persistencia":
            col = "lag_1h" if "lag_1h" in X.columns else "demand_count"
            pred = X[col].fillna(0).to_numpy()
        elif self.estrategia == "lag_24h":
            col = "lag_24h"
            pred = X[col].fillna(0).to_numpy()
        elif self.estrategia == "lag_168h":
            col = "lag_168h"
            pred = X[col].fillna(0).to_numpy()
        elif self.estrategia == "promedio_24h_168h":
            pred = (X["lag_24h"].fillna(0) + X["lag_168h"].fillna(0)) / 2.0
            pred = pred.to_numpy()
        else:
            pred = np.zeros(len(X))

        return np.clip(pred, 0, None)


class DemandaLinealP1(BaseEstimator, RegressorMixin):
    """
    Modelo de Regresión Lineal para P1 con StandardScaler integrado.
    Expone el atributo .pipeline con el paso 'scaler' requerido por pytest.
    """

    def __init__(self, features: Optional[List[str]] = None):
        self.features = features
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ])

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        self.features_ = (
            list(self.features) if self.features is not None else list(X.columns)
        )
        self.pipeline.fit(X[self.features_], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "features_"):
            cols = list(self.features) if self.features is not None else list(X.columns)
        else:
            cols = self.features_

        pred = self.pipeline.predict(X[cols])
        return np.clip(pred, 0, None)

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DemandaLinealP1":
        return joblib.load(Path(filepath))


class DemandaPoissonP1(BaseEstimator, RegressorMixin):
    """
    Modelo de Regresión Poisson para P1 con StandardScaler integrado.
    Expone el atributo .pipeline con el paso 'scaler' requerido por pytest.
    """

    def __init__(self, features: Optional[List[str]] = None,
                alpha: float = 1e-4, max_iter: int = 300):
        self.features = features
        self.alpha = alpha
        self.max_iter = max_iter
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter)),
        ])

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        self.features_ = (
            list(self.features) if self.features is not None else list(X.columns)
        )
        self.pipeline.fit(X[self.features_], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "features_"):
            cols = list(self.features) if self.features is not None else list(X.columns)
        else:
            cols = self.features_

        pred = self.pipeline.predict(X[cols])
        return np.clip(pred, 0, None)

    @property
    def estimator_(self):
        """Retorna el estimador final para compatibilidad con MLflow."""
        return self.pipeline.named_steps["regressor"]

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DemandaLinealP1":
        return joblib.load(Path(filepath))