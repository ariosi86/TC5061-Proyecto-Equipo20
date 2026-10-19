#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de integración defensiva con MLflow Tracking convertido en clase orientada a objetos.
"""

import hashlib
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd  
from mlflow.models import infer_signature

from proyecto_mibici.config import (
    FIGURES_DIR,
    MANIFEST_PATH,
    PROJ_ROOT,
    TRIPS_PROCESSED_PATH
)

NOMBRES_CONJUNTO = {
    "Train": "train",
    "Validación": "validation",
    "Test": "test",
    "entrenamiento": "train",
    "validacion": "validation",
    "prueba": "test",
}



class MLFlowTracker:
    """Clase para gestionar el tracking y registro de experimentos en MLflow."""

    def __init__(self, tracking_uri: str = None, experiment_name: str = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)

    @staticmethod
    def hash_archivo(ruta, n=12) -> str:
        """Hash SHA256 corto del archivo, para identificar la versión de los datos."""
        h = hashlib.sha256()
        with open(ruta, "rb") as f:
            for bloque in iter(lambda: f.read(1 << 20), b""):
                h.update(bloque)
        return h.hexdigest()[:n]

    @staticmethod
    def git_commit() -> str:
        """Commit actual del repo; agrega '-dirty' si hay cambios sin commitear."""
        try:
            h = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, stderr=subprocess.DEVNULL
            ).decode().strip()
            sucio = subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJ_ROOT
            ).decode().strip() != ""
            return h[:10] + ("-dirty" if sucio else "")
        except Exception:
            return "no_disponible"

    @staticmethod
    def metricas_a_dict(tabla) -> dict:
        """Convierte la tabla a un diccionario con formato {'test_mae': ..., ...}."""
        metricas = {}
        for _, fila in tabla.iterrows():
            prefijo = NOMBRES_CONJUNTO.get(
                fila.get("conjunto"), str(fila.get("conjunto", "metric")).lower()
            )
            for col in tabla.columns:
                if col == "conjunto":
                    continue
                valor = fila[col]
                try:
                    val_float = float(valor)
                    if not (val_float != val_float):  # Filtra np.nan
                        metricas[f"{prefijo}_{col.lower()}"] = val_float
                except (ValueError, TypeError):
                    continue
        return metricas

    def armar_params_tags(
        self,
        pregunta: str,
        splits: dict,
        extra_params: dict = None,
        ruta_datos: Path = TRIPS_PROCESSED_PATH,
        col_tiempo: str = "target_hour",
    ) -> tuple:
        """Parámetros y etiquetas comunes a todos los runs de una pregunta."""
        version_datos = self.hash_archivo(ruta_datos)
        d_train = splits.get("Train") if "Train" in splits else splits.get("entrenamiento")
        d_val = splits.get("Validación") if "Validación" in splits else splits.get("validacion")
        d_test = splits.get("Test") if "Test" in splits else splits.get("prueba")

        params = {
            "pregunta": pregunta,
            "train_inicio": d_train[col_tiempo].min() if d_train is not None else None,
            "train_fin": d_train[col_tiempo].max() if d_train is not None else None,
            "validation_inicio": d_val[col_tiempo].min() if d_val is not None else None,
            "validation_fin": d_val[col_tiempo].max() if d_val is not None else None,
            "test_inicio": d_test[col_tiempo].min() if d_test is not None else None,
            "test_fin": d_test[col_tiempo].max() if d_test is not None else None,
            "datos_version": version_datos,
            **(extra_params or {}),
        }
        tags = {
            "equipo": "20",
            "pregunta": pregunta,
            "git_commit": self.git_commit(),
            "datos_version": version_datos,
        }
        return params, tags

    def registrar_run(
        self,
        run_name: str,
        params: dict,
        tags: dict,
        tabla: pd.DataFrame,
        tabla_est: pd.DataFrame = None,
        modelo=None,
        X_ejemplo=None,
        figuras: dict = None,
    ) -> str:
        """Crea un run con parámetros, métricas, modelo, figuras y tabla por estación."""
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics(self.metricas_a_dict(tabla))
            mlflow.set_tags(tags)

            if modelo is not None:
                signature = None
                if X_ejemplo is not None:
                    try:
                        signature = infer_signature(X_ejemplo, modelo.predict(X_ejemplo))
                    except Exception:
                        signature = None

                mlflow.sklearn.log_model(
                    sk_model=modelo,
                    name="model",
                    signature=signature,
                    input_example=X_ejemplo,
                )

            if figuras:
                FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                for nombre, fig in figuras.items():
                    mlflow.log_figure(fig, f"graficas/{nombre}.png")
                    ruta_local_fig = FIGURES_DIR / f"{nombre}.png"
                    fig.savefig(ruta_local_fig, dpi=150, bbox_inches="tight")

            if tabla_est is not None and not tabla_est.empty:
                mlflow.log_text(tabla_est.to_csv(index=False), "metricas_por_estacion.csv")

            if Path(MANIFEST_PATH).exists():
                mlflow.log_artifact(str(MANIFEST_PATH))

            return run.info.run_id