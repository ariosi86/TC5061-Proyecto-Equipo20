#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clases orientadas a objetos para los modelos de P2 (Riesgo de Desbalance).
Incluye la línea base heurística de persistencia y el pipeline de regresión logística.
"""

from pathlib import Path
from typing import List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BaselinePersistenciaP2(BaseEstimator, ClassifierMixin):
    """
    Línea base ingenua para P2 basada en inercia operativa:
    Asume que el estado de la estación en t+1 será idéntico al observado en t (columna 'en_riesgo').
    """

    def __init__(self, col_persistencia: str = "en_riesgo"):
        self.col_persistencia = col_persistencia

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.col_persistencia not in X.columns:
            raise KeyError(
                f"La columna '{self.col_persistencia}' no existe en el conjunto de entrada."
            )
        return X[self.col_persistencia].astype(bool).to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve probabilidades degeneradas (0.0 o 1.0) para compatibilidad con PR-AUC."""
        preds = self.predict(X).astype(float)
        return np.vstack([1.0 - preds, preds]).T


class RiesgoLogisticoP2(BaseEstimator, ClassifierMixin):
    """
    Pipeline de scikit-learn con StandardScaler + LogisticRegression balanceada
    para clasificar el riesgo de desbalance en la hora siguiente.
    """

    def __init__(
        self,
        class_weight: str = "balanced",
        random_state: int = 42,
        max_iter: int = 1000,
    ):
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight=class_weight,
                    random_state=random_state,
                    max_iter=max_iter,
                ),
            ),
        ])
        self.features_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        self.features_ = list(X.columns)
        self.pipeline.fit(X[self.features_], y)
        return self

    def predict(self, X: pd.DataFrame, umbral: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return (prob >= umbral).astype(bool)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.features_ is None:
            raise ValueError("El modelo debe ser entrenado antes de ejecutar predicciones.")
        return self.pipeline.predict_proba(X[self.features_])

    @property
    def estimator(self):
        return self.pipeline.named_steps["classifier"]

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "RiesgoLogisticoP2":
        return joblib.load(filepath)