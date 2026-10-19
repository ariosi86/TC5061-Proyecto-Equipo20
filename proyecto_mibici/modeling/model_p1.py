#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de modelos orientados a objetos para P1 (Demanda Horaria de Viajes).
Incluye estimadores baseline (reglas fijas) y el pipeline de regresión lineal.
"""

from pathlib import Path
from typing import List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BaselineIngenuoP1(BaseEstimator, RegressorMixin):
    """
    Modelos base basados en heurísticas fijas de series de tiempo.
    Estrategias soportadas:
      - 'persistencia': usa lag_1h (hora previa)
      - 'lag_24h': usa lag_24h (misma hora de ayer)
      - 'lag_168h': usa lag_168h (misma hora de la semana anterior)
      - 'promedio_24h_168h': promedio de ayer y la semana pasada
    """

    def __init__(self, estrategia: str = "promedio_24h_168h"):
        estrategias_validas = {
            "persistencia",
            "lag_24h",
            "lag_168h",
            "promedio_24h_168h",
        }
        if estrategia not in estrategias_validas:
            raise ValueError(f"Estrategia '{estrategia}' no válida. Opciones: {estrategias_validas}")
        self.estrategia = estrategia

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.estrategia == "persistencia":
            preds = X["lag_1h"].to_numpy()
        elif self.estrategia == "lag_24h":
            preds = X["lag_24h"].to_numpy()
        elif self.estrategia == "lag_168h":
            preds = X["lag_168h"].to_numpy()
        elif self.estrategia == "promedio_24h_168h":
            preds = ((X["lag_24h"] + X["lag_168h"]) / 2.0).to_numpy()
        return np.clip(preds, 0, None)


class DemandaLinealP1(BaseEstimator, RegressorMixin):
    """
    Pipeline de regresión lineal con StandardScaler y recorte inferior en cero.
    Soporta cualquier subconjunto de variables predictoras (v1 o v2).
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression(fit_intercept=fit_intercept)),
        ])
        self.features_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        self.features_ = list(X.columns)
        self.pipeline.fit(X[self.features_], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.features_ is None:
            raise ValueError("El modelo debe ser entrenado antes de ejecutar predicciones.")
        y_pred = self.pipeline.predict(X[self.features_])
        return np.clip(y_pred, 0, None)

    @property
    def estimator(self):
        return self.pipeline.named_steps["regressor"]

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "DemandaLinealP1":
        return joblib.load(filepath)