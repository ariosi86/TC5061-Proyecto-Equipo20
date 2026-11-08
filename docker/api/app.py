import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from dotenv import load_dotenv
from fastapi.exceptions import RequestValidationError

from model_loader import load_model_and_params
from schemas import (
    P1Input, P1Output,
    P2Input, P2Output,
    P3Input, P3Output,
)

load_dotenv()

app = FastAPI(title="MiBici API", version="1.0.0")

PREGUNTA = os.environ.get("PREGUNTA", "p1")
MODELO_REGISTRY = os.environ.get("MODELO_REGISTRY")

# Esquemas por pregunta: FastAPI valida la entrada y documenta entrada y salida en /docs.
# Para sumar o quitar una pregunta basta con editar estos dos diccionarios.
# Si PREGUNTA no existe, se cae a dict (sin esquema) y /predict responde 400 como antes.
MODELS_INPUT = {"p1": P1Input, "p2": P2Input, "p3": P3Input}
MODELS_OUTPUT = {"p1": P1Output, "p2": P2Output, "p3": P3Output}
EntradaModel = MODELS_INPUT.get(PREGUNTA, dict)
SalidaModel = MODELS_OUTPUT.get(PREGUNTA)

# Cache ligero del modelo cargado por petición (evita recargar run_id/version en cada predict interno)
modelo_name_cache = {}


def obtener_modelo_desde_registry():
    """
    Carga el modelo desde MLflow Model Registry en cada petición.
    Diseño intencional: el modelo no va empaquetado en la imagen; se descarga
    bajo demanda desde MLflow usando download_artifacts + joblib.
    """
    try:
        return load_model_and_params()
    except Exception as exc:  # noqa: BLE001
        print(f"No se pudo cargar el modelo {MODELO_REGISTRY}: {exc}")
        traceback.print_exc()
        return None


@app.get("/health", tags=["health"])
def health():
    """
    Estado del servicio. Intenta cargar el campeón desde MLflow.
    Devuelve 200 si el modelo se lee correctamente, 503 en caso contrario.
    """
    ctx = obtener_modelo_desde_registry()
    if not ctx:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "pregunta": PREGUNTA,
                "modelo_cargado": False,
                "error": "Modelo no disponible",
            },
        )

    return {
        "status": "ok",
        "pregunta": PREGUNTA,
        "modelo_cargado": True,
        "modelo": ctx["model_name"],
        "version": ctx["version"],
        "run_id": ctx["run_id"],
        "alias": ctx["alias"],
        "model_uri": ctx["model_uri"],
        "parametros_entrenamiento": ctx["params"],
        "tags": ctx["tags"],
    }


def _predecir_p1(modelo_obj, datos: dict):
    entrada = P1Input(**datos)
    df = entrada.to_dataframe()
    pred = modelo_obj.predict(df)
    return P1Output(
        modelo=modelo_name_cache["model_name"],
        version=modelo_name_cache["version"],
        run_id=modelo_name_cache["run_id"],
        demanda_predicha=float(pred[0]),
    )


def _predecir_p2(modelo_obj, datos: dict):
    entrada = P2Input(**datos)
    df = entrada.to_dataframe()
    pred = modelo_obj.predict(df)
    try:
        proba = modelo_obj.predict_proba(df)[:, 1]
        prob_riesgo = float(proba[0])
    except Exception:
        prob_riesgo = float(pred[0])
    return P2Output(
        modelo=modelo_name_cache["model_name"],
        version=modelo_name_cache["version"],
        run_id=modelo_name_cache["run_id"],
        prob_riesgo=prob_riesgo,
        pred_riesgo=int(pred[0]),
    )


def _predecir_p3(modelo_obj, datos: dict):
    entrada = P3Input(**datos)
    df = entrada.to_dataframe()
    pred = modelo_obj.predict(df)
    return P3Output(
        modelo=modelo_name_cache["model_name"],
        version=modelo_name_cache["version"],
        run_id=modelo_name_cache["run_id"],
        grupo_kmeans=int(pred[0]),
    )


@app.post("/predict", response_model=SalidaModel, tags=["predict"])
def predict(datos: EntradaModel): # pyright: ignore[reportInvalidTypeForm]  # esquema elegido por PREGUNTA al arrancar
    """
    Recibe un JSON con las features de la pregunta configurada (PREGUNTA),
    ejecuta el modelo cargado desde MLflow y devuelve la predicción en JSON.
    El esquema de entrada y salida mostrado en /docs corresponde a esa pregunta.
    """
    ctx = obtener_modelo_desde_registry()
    if not ctx:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Modelo no cargado. Verifica MLFLOW_TRACKING_URI y MODELO_REGISTRY.",
                "pregunta": PREGUNTA,
            },
        )

    try:
        modelo_obj = ctx["modelo"]
        modelo_name_cache.update(ctx)
        
        # Con esquema, FastAPI ya validó y entrega un objeto Pydantic; sin esquema llega un dict
        datos_dict = datos.model_dump() if hasattr(datos, "model_dump") else datos


        if PREGUNTA == "p1":
            resultado = _predecir_p1(modelo_obj, datos_dict)
        elif PREGUNTA == "p2":
            resultado = _predecir_p2(modelo_obj, datos_dict)
        elif PREGUNTA == "p3":
            resultado = _predecir_p3(modelo_obj, datos_dict)
        else:
            raise HTTPException(
                status_code=400,
                detail={"error": f"Pregunta desconocida: {PREGUNTA}"},
            )

        return resultado.model_dump()

    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "Datos de entrada inválidos", "detalle": exc.errors()},
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail={"error": str(exc)})

@app.exception_handler(RequestValidationError)
async def validacion_handler(request, exc: RequestValidationError):
    # Se omite "input" y "ctx": pueden traer NaN/Infinity, que no son JSON válido
    errores = [
        {"type": e["type"], "loc": e["loc"], "msg": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errores})