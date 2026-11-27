import os
import threading
import time
import joblib
import mlflow
from mlflow.tracking import MlflowClient

_CACHE_LOCK = threading.Lock()
_CACHE: dict = {"ctx": None, "expires_at": 0.0}


def _cargar_desde_mlflow():
    """Descarga y deserializa el modelo y sus metadatos desde MLflow Model Registry."""
    model_uri = os.environ.get("MODELO_REGISTRY")
    if not model_uri:
        raise ValueError("La variable de entorno MODELO_REGISTRY no está definida.")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # model_uri tiene formato: models:/nombre_modelo@alias
    name_part = model_uri.replace("models:/", "")
    if "@" in name_part:
        model_name, alias = name_part.split("@", 1)
    else:
        model_name = name_part
        alias = None

    # Resolver versión por alias
    if alias:
        mv = client.get_model_version_by_alias(model_name, alias)
    else:
        mv = client.get_latest_versions(model_name)
        if not mv:
            raise ValueError(f"No se encontraron versiones para el modelo {model_name}")
        mv = mv[0]

    run_id = mv.run_id
    version = mv.version

    # Descargar el artifact del modelo al contenedor local
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model",
        dst_path=f"/tmp/mlflow_model_{run_id}",
    )

    # Preferir el artefacto crudo guardado con joblib (modelo sklearn/LightGBM/KMeans)
    # en lugar del model.pkl del wrapper _MiModeloPyFunc de mlflow.pyfunc.log_model.
    model_file = None
    candidate_crudo = os.path.join(local_path, "artifacts", "modelo.pkl")
    if os.path.isfile(candidate_crudo):
        model_file = candidate_crudo
    else:
        for root, _, files in os.walk(local_path):
            for f in files:
                if f.endswith(".pkl"):
                    model_file = os.path.join(root, f)
                    break
            if model_file:
                break

    if model_file is None:
        raise FileNotFoundError(f"No se encontró archivo .pkl en {local_path}")

    modelo = joblib.load(model_file)

    # Obtener parámetros y tags del run de entrenamiento
    run = client.get_run(run_id)
    params = dict(run.data.params)
    tags = dict(run.data.tags)

    return {
        "modelo": modelo,
        "model_name": model_name,
        "alias": alias,
        "version": version,
        "run_id": run_id,
        "params": params,
        "tags": tags,
        "model_uri": model_uri,
    }


def load_model_and_params():
    """
    Carga un modelo del MLflow Model Registry usando caché en memoria con TTL.
    Controlado por MODEL_CACHE_TTL (segundos, default 60; <= 0 desactiva el caché).
    """
    ttl = float(os.environ.get("MODEL_CACHE_TTL", "60"))
    if ttl <= 0:
        return _cargar_desde_mlflow()

    now = time.monotonic()
    if _CACHE["ctx"] is not None and now < _CACHE["expires_at"]:
        return _CACHE["ctx"]

    with _CACHE_LOCK:
        now = time.monotonic()
        if _CACHE["ctx"] is not None and now < _CACHE["expires_at"]:
            return _CACHE["ctx"]
        ctx = _cargar_desde_mlflow()
        _CACHE["ctx"] = ctx
        _CACHE["expires_at"] = time.monotonic() + ttl
        return ctx
