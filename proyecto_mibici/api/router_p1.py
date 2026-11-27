import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mlflow
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from mlflow.tracking import MlflowClient

from proyecto_mibici.api.schema_p1 import P1Entrada, P1Salida

# Configuración por variables de entorno (en el contenedor las pasaremos con -e)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("P1_MODEL_NAME", "mibici-equipo20-p1-demanda")
MODEL_ALIAS = os.getenv("P1_MODEL_ALIAS", "champion")

#Ruta local del modelo empaquetado si no se encuentra disponible el servidor con el modelo en el Registry
MODEL_URI = os.getenv("P1_MODEL_URI")
MODEL_VERSION = os.getenv("P1_MODEL_VERSION", "1")

router = APIRouter(tags=["P1 - Demanda horaria"])

#Modelo con su version
@dataclass
class ModeloP1:
    modelo: Any
    version: str


@lru_cache(maxsize=1)
def cargar_modelo_p1() -> ModeloP1:
    """Descarga el modelo del Registry una sola vez y lo reutiliza en cada petición."""
    #Si se define el URI del modelo empaquetado, se carga desde ahí
    if MODEL_URI:
        modelo = mlflow.pyfunc.load_model(MODEL_URI)
        return ModeloP1(modelo=modelo, version=MODEL_VERSION)
    #Si no, se carga desde el Registry
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    modelo = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
    return ModeloP1(modelo=modelo, version=str(version))


def get_modelo_p1() -> ModeloP1:
    try:
        return cargar_modelo_p1()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"No se pudo cargar el modelo P1: {e}")


@router.get("/health")
def health(m: ModeloP1 = Depends(get_modelo_p1)):
    return {"status": "ok", "modelo": MODEL_NAME, "alias": MODEL_ALIAS, "version": m.version}


@router.post("/predict", response_model=P1Salida)
def predict(entrada: P1Entrada, m: ModeloP1 = Depends(get_modelo_p1)):
    X = entrada.a_dataframe()
    try:
        y = m.modelo.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

    return P1Salida(
        prediccion_demanda=float(np.asarray(y).ravel()[0]),
        origen_id=entrada.origen_id,
        target_hour=entrada.target_hour,
        modelo=MODEL_NAME,
        version=m.version,
    )