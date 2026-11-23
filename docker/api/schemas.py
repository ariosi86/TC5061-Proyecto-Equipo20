from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


FEATURES_P1 = [
    "lag_1h", "lag_24h", "lag_168h",
    "target_hour_sin", "target_hour_cos", "target_is_weekend",
    "rolling_3h", "rolling_168h",
]

# --- P1: demanda horaria ---
class P1Input(BaseModel):
    """Datos para predecir la demanda de la hora siguiente en una estación.

    Las variables de calendario (seno, coseno y fin de semana) NO se envían:
    la API las calcula a partir de `target_hour`.
    """

    model_config = ConfigDict(
        extra="forbid",  # rechaza campos que no existen en el esquema
        json_schema_extra={
            "examples": [
                {
                    "target_hour": "2026-11-04T08:00:00",
                    "lag_1h": 5,
                    "lag_24h": 7.0,
                    "lag_168h": 6.0,
                    "rolling_3h": 4.67,
                    "rolling_168h": 5.2,
                }
            ]
        },
    )

    target_hour: datetime = Field(
        ..., examples=["2026-11-04T08:00:00"],
        description="Hora a predecir. Formato AAAA-MM-DDTHH:MM:SS, hora local, sin zona horaria",
    )
    lag_1h: int = Field(
        ..., ge=0, examples=[5],
        description="Demanda (viajes, entero) de la última hora observada",
    )
    lag_24h: float = Field(
        ..., ge=0, allow_inf_nan=False, examples=[7.0],
        description="Demanda 24 h antes de la hora objetivo",
    )
    lag_168h: float = Field(
        ..., ge=0, allow_inf_nan=False, examples=[6.0],
        description="Demanda 168 h (una semana) antes de la hora objetivo",
    )
    rolling_3h: float = Field(
        ..., ge=0, allow_inf_nan=False, examples=[4.67],
        description="Media de la demanda de las últimas 3 horas",
    )
    rolling_168h: float = Field(
        ..., ge=0, allow_inf_nan=False, examples=[5.2],
        description="Media de la demanda de las últimas 168 horas",
    )

    @field_validator("target_hour")
    @classmethod
    def sin_zona_horaria(cls, v: datetime) -> datetime:
        if v.tzinfo is not None:
            raise ValueError("target_hour debe ir sin zona horaria (hora local)")
        return v

    def to_dataframe(self) -> pd.DataFrame:
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
        return df.astype({
            "lag_1h": "int64", "target_is_weekend": "int64",
            "lag_24h": "float64", "lag_168h": "float64",
            "target_hour_sin": "float64", "target_hour_cos": "float64",
            "rolling_3h": "float64", "rolling_168h": "float64",
        })

class P1Output(BaseModel):
    pregunta: str = "p1"
    modelo: str
    version: str
    run_id: str
    demanda_predicha: float = Field(..., ge=0, description="Viajes esperados en la hora objetivo (≥ 0)")

# --- P2: riesgo de desbalance ---
FEATURES_P2 = [
    "hora_del_dia", "dia_semana", "fin_de_semana", "capacidad",
    "salidas", "llegadas", "balance_acumulado_dia",
    "acumulado_relativo", "balance_ult_3h_relativo",
    "riesgo_vacia", "riesgo_llena",
]

# Mismo valor que P2_UMBRAL en proyecto_mibici/config.py (±50 % de la capacidad)
UMBRAL_P2 = 0.5
# Tolerancia al comparar las variables relativas que calcula quien usa la API
TOLERANCIA_P2 = 1e-3


class P2Input(BaseModel):
    """Estado de una estación en la hora observada, para predecir si estará en riesgo
    (vacía o llena) en la hora siguiente.

    Las variables derivadas se calculan igual que en features_p2.py:
    acumulado_relativo = balance_acumulado_dia / capacidad,
    balance_ult_3h_relativo = balance de las últimas 3 horas / capacidad, y
    riesgo_vacia / riesgo_llena cuando el balance del día cruza ±50 % de la capacidad.
    Se rechazan combinaciones contradictorias entre estas variables.
    """

    model_config = ConfigDict(
        extra="forbid",  # rechaza campos que no existen en el esquema
        json_schema_extra={
            "examples": [
                {
                    "hora_del_dia": 8,
                    "dia_semana": 2,
                    "fin_de_semana": 0,
                    "capacidad": 20,
                    "salidas": 6,
                    "llegadas": 2,
                    "balance_acumulado_dia": -9,
                    "acumulado_relativo": -0.45,
                    "balance_ult_3h_relativo": -0.35,
                    "riesgo_vacia": 0,
                    "riesgo_llena": 0,
                }
            ]
        },
    )

    hora_del_dia: int = Field(
        ..., ge=0, le=23, examples=[8],
        description="Hora observada (0 a 23); se predice la hora siguiente",
    )
    dia_semana: int = Field(
        ..., ge=0, le=6, examples=[2],
        description="Día de la semana (lunes = 0, domingo = 6)",
    )
    fin_de_semana: int = Field(
        ..., ge=0, le=1, examples=[0],
        description="1 si dia_semana es sábado (5) o domingo (6)",
    )
    capacidad: int = Field(
        ..., gt=0, examples=[20],
        description="Número de anclajes de la estación",
    )
    salidas: int = Field(
        ..., ge=0, examples=[6],
        description="Bicicletas que salieron en la hora observada",
    )
    llegadas: int = Field(
        ..., ge=0, examples=[2],
        description="Bicicletas que llegaron en la hora observada",
    )
    balance_acumulado_dia: int = Field(
        ..., examples=[-9],
        description="Llegadas menos salidas desde el inicio del día operativo (3:00)",
    )
    acumulado_relativo: float = Field(
        ..., allow_inf_nan=False, examples=[-0.45],
        description="balance_acumulado_dia / capacidad",
    )
    balance_ult_3h_relativo: float = Field(
        ..., allow_inf_nan=False, examples=[-0.35],
        description="(Llegadas - salidas de las últimas 3 horas, incluida la observada) / capacidad",
    )
    riesgo_vacia: int = Field(
        ..., ge=0, le=1, examples=[0],
        description="1 si balance_acumulado_dia <= -50 % de la capacidad",
    )
    riesgo_llena: int = Field(
        ..., ge=0, le=1, examples=[0],
        description="1 si balance_acumulado_dia >= 50 % de la capacidad",
    )

    @model_validator(mode="after")
    def variables_coherentes(self):
        errores = []
        if self.fin_de_semana != int(self.dia_semana >= 5):
            errores.append(
                f"fin_de_semana={self.fin_de_semana} no corresponde a dia_semana={self.dia_semana}"
            )
        esperado = self.balance_acumulado_dia / self.capacidad
        if abs(self.acumulado_relativo - esperado) > TOLERANCIA_P2:
            errores.append(
                f"acumulado_relativo={self.acumulado_relativo} debería ser "
                f"balance_acumulado_dia / capacidad = {esperado:.4f}"
            )
        if self.riesgo_vacia != int(self.balance_acumulado_dia <= -UMBRAL_P2 * self.capacidad):
            errores.append(
                f"riesgo_vacia={self.riesgo_vacia} no corresponde a balance_acumulado_dia="
                f"{self.balance_acumulado_dia} con capacidad={self.capacidad}"
            )
        if self.riesgo_llena != int(self.balance_acumulado_dia >= UMBRAL_P2 * self.capacidad):
            errores.append(
                f"riesgo_llena={self.riesgo_llena} no corresponde a balance_acumulado_dia="
                f"{self.balance_acumulado_dia} con capacidad={self.capacidad}"
            )
        if errores:
            raise ValueError("; ".join(errores))
        return self

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame([self.model_dump()], columns=FEATURES_P2)
        # Tipos como en el entrenamiento: enteros, decimales y riesgos como booleanos
        return df.astype({
            "hora_del_dia": "int32",
            "dia_semana": "int32",
            "fin_de_semana": "int64",
            "capacidad": "int32",
            "salidas": "int64",
            "llegadas": "int64",
            "balance_acumulado_dia": "int64",
            "acumulado_relativo": "float64",
            "balance_ult_3h_relativo": "float64",
            "riesgo_vacia": "bool",
            "riesgo_llena": "bool",
        })


class P2Output(BaseModel):
    pregunta: str = "p2"
    modelo: str
    version: str
    run_id: str
    prob_riesgo: float = Field(
        ..., ge=0, le=1,
        description="Probabilidad de que la estación esté en riesgo en la hora siguiente",
    )
    pred_riesgo: int = Field(
        ..., ge=0, le=1,
        description="1 si se predice que la estación quedará vacía o llena en la hora siguiente",
    )


# --- P3: clustering de estaciones ---
FEATURES_P3 = [
    "pct_manana", "pct_mediodia", "pct_tarde", "pct_noche",
    "pct_fin_de_semana", "duracion_media", "viajes_dia_log", "balance_neto",
]


class P3Input(BaseModel):
    """Perfil de uso de una estación para asignarle un grupo funcional (KMeans K=4).

    Los porcentajes son fracciones (0 a 1) de las salidas de la estación.
    La madrugada (0 a 6 h) no entra al modelo, por eso las cuatro franjas
    horarias pueden sumar menos de 1, pero nunca más.
    """

    model_config = ConfigDict(
        extra="forbid",  # rechaza campos que no existen en el esquema
        json_schema_extra={
            "examples": [
                {
                    "pct_manana": 0.23,
                    "pct_mediodia": 0.33,
                    "pct_tarde": 0.29,
                    "pct_noche": 0.12,
                    "pct_fin_de_semana": 0.20,
                    "duracion_media": 11.12,
                    "viajes_dia_log": 3.23,
                    "balance_neto": -0.02,
                }
            ]
        },
    )

    pct_manana: float = Field(
        ..., ge=0, le=1, allow_inf_nan=False, examples=[0.23],
        description="Fracción de salidas entre 6 y 10 h",
    )
    pct_mediodia: float = Field(
        ..., ge=0, le=1, allow_inf_nan=False, examples=[0.33],
        description="Fracción de salidas entre 10 y 16 h",
    )
    pct_tarde: float = Field(
        ..., ge=0, le=1, allow_inf_nan=False, examples=[0.29],
        description="Fracción de salidas entre 16 y 20 h",
    )
    pct_noche: float = Field(
        ..., ge=0, le=1, allow_inf_nan=False, examples=[0.12],
        description="Fracción de salidas entre 20 y 24 h",
    )
    pct_fin_de_semana: float = Field(
        ..., ge=0, le=1, allow_inf_nan=False, examples=[0.20],
        description="Fracción de salidas en sábado o domingo",
    )
    duracion_media: float = Field(
        ..., gt=0, le=180, allow_inf_nan=False, examples=[11.12],
        description="Duración media del viaje en minutos (máximo 180, igual que en entrenamiento)",
    )
    viajes_dia_log: float = Field(
        ..., ge=0, allow_inf_nan=False, examples=[3.23],
        description="log(1 + viajes promedio por día que salen de la estación)",
    )
    balance_neto: float = Field(
        ..., ge=-1, le=1, allow_inf_nan=False, examples=[-0.02],
        description="(llegadas - salidas) / (llegadas + salidas)",
    )

    @model_validator(mode="after")
    def franjas_no_pasan_de_1(self):
        suma = self.pct_manana + self.pct_mediodia + self.pct_tarde + self.pct_noche
        if suma > 1 + 1e-6:
            raise ValueError(
                f"pct_manana + pct_mediodia + pct_tarde + pct_noche suman {suma:.3f}; no pueden pasar de 1"
            )
        return self

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()], columns=FEATURES_P3).astype("float64")


class P3Output(BaseModel):
    pregunta: str = "p3"
    modelo: str
    version: str
    run_id: str
    grupo_kmeans: int = Field(..., ge=0, le=3, description="Grupo funcional asignado (KMeans K=4: 0 a 3)")
