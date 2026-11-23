# Plan: despliegue automatizado de MiBici en contenedores (Tarea 4)

Referencia: `tarea4.md` — API REST con FastAPI + contenedores (Podman/Docker), modelo recuperado del Model Registry de MLflow, documento en README.

## Arquitectura de contenedores (flujo automatizado)

```
Red: mibici-net

1. mlflow-server        (puerto 5000)  — servidor de tracking + Model Registry
2. ingest               (job, termina) — python -m proyecto_mibici.dataset
3. train-p1             (job, termina) — entrenamiento campeón P1 (LightGBM Tweedie)
4. train-p2             (job, termina) — entrenamiento campeón P2 (GBM)
5. train-p3             (job, termina) — entrenamiento campeón P3 (KMeans K=4)
6. register-champions   (job, termina) — python run_mejor_modelo.py
7. api-p1               (puerto 8001)  — API que carga mibici-equipo20-p1-demanda@champion
8. api-p2               (puerto 8002)  — API que carga mibici-equipo20-p2-riesgo@champion
9. api-p3               (puerto 8003)  — API que carga mibici-equipo20-p3-grupos@champion
```

**Orden de ejecución:** `mlflow-server` → `ingest` → `train-p1/p2/p3` → `register-champions` → `api-p1/p2/p3`.

Cada servicio de entrenamiento y registro es un **job** (contenedor que se ejecuta y termina), no un servicio permanente. Solo `mlflow-server` y las 3 APIs permanecen levantados.

---

## ¿Qué código vive en cada contenedor?

| Contenedor | Código dentro de la imagen | Origen del código |
|------------|---------------------------|-------------------|
| `mlflow-server` | Solo `mlflow server` (imagen oficial/slim). No contiene código del proyecto. | Construida desde `deployment/mlflow/Containerfile` |
| `ingest` | Repo completo montado/copiado; ejecuta `python -m proyecto_mibici.dataset` | `deployment/ingest/` |
| `train-p1/p2/p3` | Repo completo montado/copiado; ejecuta `train_p*.py` | `deployment/train/` |
| `register-champions` | Repo completo montado/copiado; ejecuta `run_mejor_modelo.py` | `deployment/register/` |
| `api-p1/p2/p3` | Solo lo necesario para servir: `app.py`, `model_loader.py`, `schemas.py`, `predict_p*.py`, `requirements.txt` | `deployment/api/` (imagen ligera, no todo el repo) |

> **Nota:** el código de entrenamiento y ETL (`train_p*.py`, `dataset.py`, `run_train.py`, etc.) **no va dentro de las imágenes de API**. Las imágenes de API solo contienen el código de inferencia. El entrenamiento se ejecuta desde contenedores separados que montan el repo local.

---

## Estructura de archivos propuesta

```
deployment/
├── docker-compose.yml          # orquesta todo el flujo
├── .env.example                # variables de entorno
├── mlflow/
│   └── Containerfile           # servidor MLflow (python:3.10-slim + mlflow)
├── ingest/
│   ├── Containerfile           # ejecuta python -m proyecto_mibici.dataset
│   └── entrypoint.sh           # espera a MLflow y lanza la ingesta
├── train/
│   ├── Containerfile           # imagen base con repo + dependencias
│   └── entrypoint.sh           # recibe PREGUNTA y MODELO por variables de entorno
├── register/
│   ├── Containerfile           # ejecuta python run_mejor_modelo.py
│   └── entrypoint.sh           # espera a MLflow y registra campeones
└── api/
    ├── Containerfile           # imagen ligera de FastAPI
    ├── requirements.txt        # fastapi, uvicorn, pydantic, mlflow, scikit-learn, lightgbm, pandas, numpy, python-dotenv
    ├── app.py                  # uvicorn + lee PREGUNTA y MODELO_REGISTRY
    ├── model_loader.py         # mlflow.pyfunc.load_model("models:/...@champion")
    ├── schemas.py              # Pydantic: P1Input/Output, P2Input/Output, P3Input/Output
    ├── predict_p1.py
    ├── predict_p2.py
    └── predict_p3.py
```

---

## Imagen base recomendada

**`python:3.10-slim`** (Debian-based, glibc).

- Coincide con `.python-version` (3.10.22).
- Las ruedas de `scikit-learn<1.6`, `pandas`, `numpy` y `lightgbm` se instalan directamente desde PyPI sin compilar.
- `libgomp1` (OpenMP para LightGBM) se instala con `apt-get install -y libgomp1`.
- **No usar Alpine:** LightGBM y scikit-learn tienen problemas con musl libc y obligan a compilar desde fuente.

---

## Dockerfiles (esqueletos)

### `deployment/mlflow/Containerfile`
```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir mlflow[extras]>=2.15.0
EXPOSE 5000
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "./mlruns", \
     "--host", "0.0.0.0", "--port", "5000"]
```

### `deployment/ingest/Containerfile`
```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "proyecto_mibici.dataset"]
```

### `deployment/train/Containerfile` (genérico para p1, p2, p3)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# El comando se sobrescribe en docker-compose.yml según la pregunta
CMD ["python", "-m", "proyecto_mibici.modeling.train_p1", "--model", "baseline"]
```

### `deployment/register/Containerfile`
```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "run_mejor_modelo.py"]
```

### `deployment/api/Containerfile`
```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## docker-compose.yml (esqueleto)

```yaml
services:
  mlflow-server:
    build:
      context: ..
      dockerfile: deployment/mlflow/Containerfile
    container_name: mlflow-server
    ports: ["5000:5000"]
    volumes:
      - ./deployment/mlflow/mlflow.db:/app/mlflow.db
      - ./deployment/mlflow/mlruns:/app/mlruns
    networks: [mibici-net]
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:5000"]
      interval: 10s
      timeout: 5s
      retries: 10

  ingest:
    build:
      context: ..
      dockerfile: deployment/ingest/Containerfile
    container_name: ingest
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow-server:5000
    volumes:
      - .:/app:ro
    depends_on:
      mlflow-server:
        condition: service_healthy
    networks: [mibici-net]

  train-p1:
    build:
      context: ..
      dockerfile: deployment/train/Containerfile
    container_name: train-p1
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow-server:5000
    volumes:
      - .:/app:ro
    command: ["python", "-m", "proyecto_mibici.modeling.train_p1", "--model", "lgbm_tweedie"]
    depends_on:
      ingest:
        condition: service_completed_successfully
    networks: [mibici-net]

  train-p2:
    build:
      context: ..
      dockerfile: deployment/train/Containerfile
    container_name: train-p2
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow-server:5000
    volumes:
      - .:/app:ro
    command: ["python", "-m", "proyecto_mibici.modeling.train_p2", "--modelo", "gbm"]
    depends_on:
      ingest:
        condition: service_completed_successfully
    networks: [mibici-net]

  train-p3:
    build:
      context: ..
      dockerfile: deployment/train/Containerfile
    container_name: train-p3
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow-server:5000
    volumes:
      - .:/app:ro
    command: ["python", "-m", "proyecto_mibici.modeling.train_p3", "--modelo", "kmeans", "--n-clusters", "4", "--n-init", "10"]
    depends_on:
      ingest:
        condition: service_completed_successfully
    networks: [mibici-net]

  register-champions:
    build:
      context: ..
      dockerfile: deployment/register/Containerfile
    container_name: register-champions
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow-server:5000
    volumes:
      - .:/app:ro
    depends_on:
      train-p1:
        condition: service_completed_successfully
      train-p2:
        condition: service_completed_successfully
      train-p3:
        condition: service_completed_successfully
    networks: [mibici-net]

  api-p1:
    build:
      context: ..
      dockerfile: deployment/api/Containerfile
    container_name: api-p1
    ports: ["8001:8000"]
    environment:
      - PREGUNTA=p1
      - MODELO_REGISTRY=mibici-equipo20-p1-demanda@champion
      - MLFLOW_TRACKING_URI=http://mlflow-server:5000
    depends_on:
      register-champions:
        condition: service_completed_successfully
    networks: [mibici-net]

  api-p2:
    build:
      context: ..
      dockerfile: deployment/api/Containerfile
    container_name: api-p2
    ports: ["8002:8000"]
    environment:
      - PREGUNTA=p2
      - MODELO_REGISTRY=mibici-equipo20-p2-riesgo@champion
      - MLFLOW_TRACKING_URI=http://mlflow-server:5000
    depends_on:
      register-champions:
        condition: service_completed_successfully
    networks: [mibici-net]

  api-p3:
    build:
      context: ..
      dockerfile: deployment/api/Containerfile
    container_name: api-p3
    ports: ["8003:8000"]
    environment:
      - PREGUNTA=p3
      - MODELO_REGISTRY=mibici-equipo20-p3-grupos@champion
      - MLFLOW_TRACKING_URI=http://mlflow-server:5000
    depends_on:
      register-champions:
        condition: service_completed_successfully
    networks: [mibici-net]

networks:
  mibici-net:
    driver: bridge
```

---

## Variables de entorno (`.env.example`)

```
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_EXPERIMENT_NAME=mibici-equipo20
```

> En los contenedores, `MLFLOW_TRACKING_URI` apunta al nombre del servicio `mlflow-server` dentro de la red Docker/Podman, no a `localhost`.

---

## Plan de ejecución

```bash
# 1. Construir todas las imágenes
docker-compose build   # o podman-compose build

# 2. Levantar todo el flujo (MLflow, ingesta, entrenamientos, registro, APIs)
docker-compose up -d   # o podman-compose up -d

# 3. Verificar que MLflow esté arriba
curl -sf http://localhost:5000

# 4. Probar cada API
curl http://localhost:8001/health   # P1
curl http://localhost:8002/health   # P2
curl http://localhost:8003/health   # P3

# 5. Ejemplo de predicción P1
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"lag_1h":2,"lag_24h":3,"lag_168h":1,"rolling_3h":2.5,"rolling_168h":2.0,"target_hour_sin":0.5,"target_hour_cos":0.8,"target_is_weekend":0}'
```

---

## Entregables de tarea4.md cubiertos

1. ✅ Recuperar modelo desde Model Registry por nombre y alias `@champion` → `model_loader.py` + `MLFLOW_TRACKING_URI`.
2. ✅ API REST con FastAPI, endpoint `/predict`, validación Pydantic → `app.py` + `schemas.py`.
3. ✅ Containerfile OCI (Podman/Docker) → `deployment/*/Containerfile`.
4. ✅ Construir y probar contenedores localmente → pasos de ejecución arriba.
5. ✅ Documentar en README (build, run, ejemplo petición/respuesta, justificación) → `deployment/README_DEPLOY.md` + actualizar `README.md` raíz.
6. ✅ Captura de pantalla del contenedor respondiendo → después del paso 4.
7. ✅ No eliminar ni reemplazar entregas previas → `deployment/` es aditivo al repo existente.

---

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Los contenedores de job (ingest, train, register) fallan y las APIs no arrancan | `condition: service_completed_successfully` en `depends_on` evita que las APIs se levanten sin campeones registrados. |
| `mlflow run` intenta crear entornos aislados con pyenv | No usamos `mlflow run`; ejecutamos directamente `python -m proyecto_mibici...` dentro de los contenedores. |
| LightGBM necesita OpenMP | `apt-get install -y libgomp1` en cada `Containerfile` que use lightgbm/scikit-learn. |
| Los Parquet de `data/processed/` no existen en el contenedor | `ingest` se ejecuta antes que `train` y genera los Parquet dentro del contenedor (montando el volumen del repo). |
| Puerto 5000 o 8001-8003 en uso local | Configurables en `docker-compose.yml` y `.env`. |
