from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
#--------------------
# P1 SCHEMA
#---------------------

FEATURES_P1 = [
    "lag_1h", "lag_24h", "lag_168h",
    "target_hour_sin", "target_hour_cos", "target_is_weekend",
    "rolling_3h", "rolling_168h",
]


class P1Entrada(BaseModel):
    """Datos para predecir la demanda de la hora siguiente en una estación."""
    model_config = ConfigDict(
        extra="forbid",   # rechaza campos que no existen en el esquema
        json_schema_extra={
            "example": {
                "origen_id": 12,
                "target_hour": "2026-11-04T08:00:00",
                "lag_1h": 5,
                "lag_24h": 7.0,
                "lag_168h": 6.0,
                "rolling_3h": 4.67,
                "rolling_168h": 5.2,
            }
        }
    )

    origen_id: Optional[int] = Field(
        None, ge=0, description="ID de la estación (opcional, solo para trazabilidad)."
    )
    target_hour: datetime = Field(
        ..., description="Hora que se quiere predecir, hora local sin zona horaria."
    )
    lag_1h: int = Field(..., ge=0, description="Demanda (entero) de la última hora observada (t).")
    lag_24h: float = Field(..., ge=0, allow_inf_nan=False, description="Demanda 24 h antes de la hora objetivo.")
    lag_168h: float = Field(..., ge=0, allow_inf_nan=False, description="Demanda 168 h (una semana) antes de la hora objetivo.")
    rolling_3h: float = Field(..., ge=0, allow_inf_nan=False, description="Media de demanda de las últimas 3 horas.")
    rolling_168h: float = Field(..., ge=0, allow_inf_nan=False, description="Media de demanda de las últimas 168 horas.")

    @field_validator("target_hour")
    @classmethod
    def sin_zona_horaria(cls, v):
        if v.tzinfo is not None:
            raise ValueError("target_hour debe ir sin zona horaria (hora local)")
        return v
    
    def a_dataframe(self) -> pd.DataFrame:
        """Construye la fila de features con el orden y los dtypes de la signature."""
        h = self.target_hour
        fila = {
            "lag_1h": self.lag_1h,
            "lag_24h": self.lag_24h,
            "lag_168h": self.lag_168h,
            "target_hour_sin": float(np.sin(2 * np.pi * h.hour / 24)),
            "target_hour_cos": float(np.cos(2 * np.pi * h.hour / 24)),
            "target_is_weekend": int(h.weekday() in (5, 6)),
            "rolling_3h": self.rolling_3h,
            "rolling_168h": self.rolling_168h,
        }
        df = pd.DataFrame([fila], columns=FEATURES_P1)
        # dtypes explícitos: long = int64, double = float64 (evita int32 en Windows)
        return df.astype({
            "lag_1h": "int64",
            "target_is_weekend": "int64",
            "lag_24h": "float64",
            "lag_168h": "float64",
            "target_hour_sin": "float64",
            "target_hour_cos": "float64",
            "rolling_3h": "float64",
            "rolling_168h": "float64",
        })


class P1Salida(BaseModel):
    prediccion_demanda: float = Field(..., ge=0, description="Viajes esperados en la hora objetivo.")
    origen_id: Optional[int] = None
    target_hour: datetime
    modelo: str = Field(..., description="Nombre del modelo en el Registry.")
    version: str = Field(..., description="Versión o alias servido.")