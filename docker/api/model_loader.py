import os
import mlflow
import joblib
from mlflow.tracking import MlflowClient


def load_model_and_params():
    """
    Carga un modelo del MLflow Model Registry usando el alias @champion.
    Descarga el artefacto .pkl desde el servidor MLflow al contenedor local
    y lo carga con joblib, ya que mlflow.pyfunc.load_model no funciona
    correctamente con file store local entre contenedores Docker.
    """
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
        dst_path=f"/tmp/mlflow_model_{run_id}"
    )

    import os as _os
    # Preferir el artefacto crudo guardado con joblib (modelo sklearn/LightGBM/KMeans)
    # en lugar del model.pkl del wrapper _MiModeloPyFunc de mlflow.pyfunc.log_model.
    model_file = None
    candidate_crudo = _os.path.join(local_path, "artifacts", "modelo.pkl")
    if _os.path.isfile(candidate_crudo):
        model_file = candidate_crudo
    else:
        for root, _, files in _os.walk(local_path):
            for f in files:
                if f.endswith(".pkl"):
                    model_file = _os.path.join(root, f)
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
