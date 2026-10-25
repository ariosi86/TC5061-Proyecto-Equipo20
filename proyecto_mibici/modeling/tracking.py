#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de integración defensiva con MLflow Tracking convertido en clase orientada a objetos.
"""

import hashlib
import os
import subprocess
from pathlib import Path

from loguru import logger

import mlflow
import mlflow.sklearn
import pandas as pd  
from mlflow.models import infer_signature
try:
    from mlflow.data.pandas_dataset import from_df  # mlflow < 3
except ImportError:  # mlflow >= 3
    from mlflow.data.pandas_dataset import from_pandas as from_df

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import mlflow.lightgbm
except ImportError:
    mlflow_lightgbm = None
else:
    mlflow_lightgbm = mlflow.lightgbm

from proyecto_mibici.config import (
    FIGURES_DIR,
    MANIFEST_PATH,
    MLFLOW_EXPERIMENT_NAME,
    PROJ_ROOT,
    TRIPS_PROCESSED_PATH
)

CONFIG_PATH = PROJ_ROOT / "proyecto_mibici" / "config.py"

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

    @staticmethod
    def _get_source_tags() -> dict:
        """Obtiene los tags de source del run activo (si existe) para propagarlos a runs hijos."""
        tags = {}
        run_activo = mlflow.active_run()
        if run_activo is not None:
            run_data = mlflow.get_run(run_activo.info.run_id).data
            source_keys = [
                "mlflow.source.type",
                "mlflow.source.name",
                "mlflow.source.git.commit",
                "mlflow.source.git.repoURL",
                "mlflow.source.git.branch",
                "mlflow.source.git.dirty",
                "mlflow.project.entryPoint",
                "mlflow.project.env",
                "mlflow.source.git.repoURL",
            ]
            for k in source_keys:
                v = run_data.tags.get(k)
                if v is not None:
                    tags[k] = v
        return tags

    @staticmethod
    def propagar_source_tags() -> None:
        """Propaga los tags de source del run padre al run activo actual (run hijo)."""
        source_tags = MLFlowTracker._get_source_tags()
        if source_tags:
            mlflow.set_tags(source_tags)

    @staticmethod
    def log_params_nuevos(params: dict) -> None:
        """Registra en la corrida activa solo los parámetros que todavía no tiene.

        Con mlflow run, MLflow registra los -P (modelo, pregunta, n_clusters, etc.) antes
        de ejecutar el script, y no permite cambiar el valor de un parámetro ya registrado.
        En una corrida nueva (por ejemplo, la hija de una búsqueda) se registran todos.
        """
        run_id = mlflow.active_run().info.run_id
        ya_registrados = mlflow.get_run(run_id).data.params
        nuevos = {k: str(v) for k, v in params.items() if k not in ya_registrados}
        if nuevos:
            mlflow.log_params(nuevos)

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
        serialization_format: str = None,
        df_train: pd.DataFrame = None,
        ruta_entrenamiento: str = None,
        nested: bool = False,
    ) -> str:
        """Crea un run con parámetros, métricas, modelo, figuras y tabla por estación.

        Si nested=True, no cierra el run activo previo y crea el run como hijo anidado
        heredando los tags de source (mlflow.source.type=PROJECT, etc.) del run padre.
        """
        # Etiqueta question automática a partir del parámetro pregunta
        if "question" not in tags and "pregunta" in params:
            tags = {**tags, "question": params["pregunta"]}

        # Asegurar que usamos el experimento correcto
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        # Propagar tags de source (mlflow.source.type=PROJECT, etc.) del run activo
        source_tags = self._get_source_tags()
        if source_tags:
            tags = {**tags, **source_tags}

        # Terminar run activo previo SOLO si NO es anidado
        if not nested:
            try:
                mlflow.end_run()
            except Exception:
                pass

        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            self.log_params_nuevos(params)
            mlflow.log_metrics(self.metricas_a_dict(tabla))
            mlflow.set_tags(tags)

            # Log del dataset de entrenamiento como input (versión de datos)
            if df_train is not None:
                try:
                    training_dataset = from_df(
                        df_train,
                        source=ruta_entrenamiento or str(TRIPS_PROCESSED_PATH),
                        name="train_set",
                    )
                    mlflow.log_input(training_dataset, context="training")
                except Exception as e:
                    logger.warning(f"No se pudo registrar mlflow.log_input: {e}")

            if modelo is not None:
                signature = None
                if X_ejemplo is not None:
                    try:
                        signature = infer_signature(X_ejemplo, modelo.predict(X_ejemplo))
                    except Exception:
                        signature = None

                try:
                    # Usar el sabor nativo de MLflow para LightGBM si corresponde
                    if lgb is not None and isinstance(modelo, lgb.LGBMRegressor):
                        if mlflow_lightgbm is not None:
                            mlflow_lightgbm.log_model(
                                lgb_model=modelo,
                                artifact_path="model",
                                signature=signature,
                                input_example=X_ejemplo,
                            )
                        else:
                            logger.warning("mlflow.lightgbm no disponible; usando mlflow.sklearn.log_model")
                            kwargs_modelo = {}
                            if serialization_format is not None:
                                kwargs_modelo["serialization_format"] = serialization_format
                            mlflow.sklearn.log_model(
                                sk_model=modelo,
                                artifact_path="model",
                                signature=signature,
                                input_example=X_ejemplo,
                                **kwargs_modelo,
                            )
                    else:
                        kwargs_modelo = {}
                        if serialization_format is not None:
                            kwargs_modelo["serialization_format"] = serialization_format
                        mlflow.sklearn.log_model(
                            sk_model=modelo,
                            artifact_path="model",
                            signature=signature,
                            input_example=X_ejemplo,
                            **kwargs_modelo,
                        )
                except Exception as e:
                    logger.warning(f"No se pudo loggear el modelo en MLflow: {e}")

            if figuras:
                FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                for nombre, fig in figuras.items():
                    mlflow.log_figure(fig, f"plots/{nombre}.png")
                    ruta_local_fig = FIGURES_DIR / f"{nombre}.png"
                    fig.savefig(ruta_local_fig, dpi=150, bbox_inches="tight")

            if tabla_est is not None and not tabla_est.empty:
                mlflow.log_text(tabla_est.to_csv(index=False), "data/metricas_por_estacion.csv")

            if Path(MANIFEST_PATH).exists():
                mlflow.log_artifact(str(MANIFEST_PATH), artifact_path="data")

            if Path(CONFIG_PATH).exists():
                mlflow.log_artifact(str(CONFIG_PATH), artifact_path="config")

            return run.info.run_id