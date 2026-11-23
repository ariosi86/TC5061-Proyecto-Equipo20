# AGENTS.md — TC5061 Proyecto MiBici (Equipo 20)

Repositorio de Machine Learning (POO + scikit-learn pipelines + MLflow) con tres problemas sobre MiBici (Guadalajara): **P1** demanda horaria (regresión), **P2** riesgo de desbalance de estaciones (clasificación), **P3** clustering funcional de estaciones (no supervisado). Paquete principal: `proyecto_mibici/`. Entrenamiento/modelado en `proyecto_mibici/modeling/`. Despliegue con contenedores en `docker/`.

---

## Estructura del repositorio

```
TC5061-Proyecto-Equipo20-prev/
├── proyecto_mibici/          # Paquete principal Python (ETL, features, modelado, tracking)
├── proyecto_mibici/modeling/ # Entrenamiento y modelos P1/P2/P3
├── proyecto_mibici/modeling/tracking.py  # Integración MLflow (joblib + log_artifact)
├── docker/                   # Contenedores (MLflow, prep, API P1/P2/P3)
│   ├── docker-compose.yml    # Orquestación de todos los servicios
│   ├── mlflow/Containerfile  # Servidor MLflow (puerto 5000)
│   ├── prep/Containerfile    # Job de ingesta + entrenamiento + registro
│   ├── prep/entrypoint.sh    # Script de ejecución de prep
│   ├── api/Containerfile     # Imagen base de API FastAPI
│   ├── api/app.py            # Endpoints /health y /predict (Flask)
│   ├── api/model_loader.py   # Carga modelo desde MLflow (download_artifacts + joblib)
│   ├── api/schemas.py        # Esquemas Pydantic P1/P2/P3
│   ├── api/predict_p*.py     # Lógica de predicción por pregunta
│   └── api/requirements.txt  # Dependencias de la API
├── data/                     # Datos (no se versionan)
├── models/                   # Modelos serializados (no se versionan)
├── mlruns/                   # Artefactos MLflow (no se versionan)
├── reports/                  # Reportes (no se versionan)
├── notebooks/                # Jupyter notebooks
├── tests/                    # Pruebas pytest
├── run_mejor_modelo.py       # Registro de campeones en Model Registry
├── run_train.py              # Orquestador de entrenamientos
├── pipeline.py               # Orquestador end-to-end
├── Makefile                  # Comandos Make (lint, format, data, etc.)
├── pyproject.toml            # Configuración ruff, paquete, línea 99
├── requirements.txt          # Dependencias del proyecto
├── MLproject                 # Entry points MLflow Projects
├── python_env.yaml           # Entorno MLflow Project
└── .env                      # Variables de entorno (no se versiona)
```

---

## Entorno local (hacer esto primero)

- Python 3.10 (`.python-version` = `3.10.22`). Crear y activar venv:
  ```bash
  python -m venv .venv
  source .venv/bin/activate   # macOS/Linux
  ```
- Instalar dependencias (`requirements.txt` incluye `-e .`):
  ```bash
  pip install -U pip && pip install -r requirements.txt
  ```
- Jupyter (notebooks/`notebooks/*.ipynb`): registrar kernel:
  ```bash
  python -m ipykernel install --user --name=mibici-venv --display-name="Python (MiBici .venv)"
  ```

---

## Datos y artefactos (no se versionan)

`data/raw/`, `data/interim/`, `data/processed/`, `data/manifest.json`, `models/*.pkl`, `mlruns/`, `mlflow.db`, `reports/` y `.env` están en `.gitignore`. Se generan localmente y nunca se comiteen.

- Generar/procesar datos (descarga CSVs de mibici.net; necesita internet):
  ```bash
  python -m proyecto_mibici.dataset
  ```
  Salidas: `data/processed/train/trips_train_2025.parquet`, `data/processed/test/trips_test_2026_04_09.parquet`, `data/manifest.json`.
- Sin los Parquets de `data/processed/`, `pytest` salta las pruebas de DeepChecks sobre datos reales y `test_pipeline.py`.

---

## MLflow (requerido para entrenar)

El servidor MLflow corre en contenedor Docker (ver sección Despliegue con contenedores). URI de tracking: `http://mlflow:5000` dentro de la red Docker, `http://localhost:5001` desde el host.

- Para entrenar localmente sin Docker:
  ```bash
  mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
  export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
  python -m proyecto_mibici.modeling.train_p1 --model lgbm_tweedie   # P1
  python -m proyecto_mibici.modeling.train_p2 --modelo gbm           # P2
  python -m proyecto_mibici.modeling.train_p3 --modelo kmeans --n-clusters 4 --n-init 10   # P3
  ```
- Registrar campeones en MLflow Model Registry (idempotente): `python run_mejor_modelo.py`

---

## Comandos principales

- Entrenar un modelo suelto (venv activo, MLflow arriba):
  ```bash
  python -m proyecto_mibici.modeling.train_p1 --model lgbm_tweedie   # P1: LightGBM Tweedie (campeón)
  python -m proyecto_mibici.modeling.train_p2 --modelo gbm           # P2: GBM (campeón)
  python -m proyecto_mibici.modeling.train_p3 --modelo kmeans --n-clusters 4 --n-init 10   # P3: KMeans K=4 (campeón)
  ```
- Todas las versiones candidatas de una pregunta: `python run_train.py train --pregunta p1|p2|p3 [--versiones base,gbm]`
- Registrar campeones en MLflow Model Registry (idempotente): `python run_mejor_modelo.py`
- Inferencia (Typer, carga modelos desde MLflow registry):
  ```bash
  python -m proyecto_mibici.modeling.predict_p1 --input reports/predicciones_p1_test.parquet [--output out.parquet]
  python -m proyecto_mibici.modeling.predict_p2 --input ... --umbral 0.35 --output out.csv
  python -m proyecto_mibici.modeling.predict_p3 --input reports/predicciones_p3_test.csv [--output out.csv]
  ```
- Orquestador end-to-end: `python -m proyecto_mibici.pipeline [--skip-data | --skip-tests]`

---

## Despliegue con contenedores (Docker Compose)

Archivos en `docker/`. Levanta MLflow + 3 contenedores de API (uno por campeón).

### Comandos

```bash
# Desde la raíz del repositorio:
cp docker/.env.example .env   # si no existe .env

# Construir todas las imágenes
docker compose -f docker/docker-compose.yml build

# Levantar todo el flujo (MLflow → prep → APIs)
docker compose -f docker/docker-compose.yml up -d

# Ver estado de contenedores
docker compose -f docker/docker-compose.yml ps

# Ver logs de un servicio
docker compose -f docker/docker-compose.yml logs prep
docker compose -f docker/docker-compose.yml logs api-p1

# Detener todo
docker compose -f docker/docker-compose.yml down

# Limpiar todo incluyendo volúmenes (borra datos de MLflow)
docker compose -f docker/docker-compose.yml down -v
```

### Endpoints expuestos

| Servicio | Puerto host | Endpoint | Descripción |
|----------|-------------|----------|-------------|
| MLflow | 5001 | `GET /` | UI de MLflow y API REST |
| API P1 | 8001 | `GET /health`, `POST /predict` | Demanda horaria (LightGBM Tweedie) |
| API P2 | 8002 | `GET /health`, `POST /predict` | Riesgo de desbalance (GBM) |
| API P3 | 8003 | `GET /health`, `POST /predict` | Clustering (KMeans K=4) |

**Ejemplo de predicción P1:**
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"lag_1h":2,"lag_24h":3,"lag_168h":1,"target_hour_sin":0.5,"target_hour_cos":0.8,"target_is_weekend":0,"rolling_3h":2.5,"rolling_168h":2.0}'
```

### Notas importantes de contenedores

- Los campeones se registran en MLflow con alias `@champion` durante la ejecución de `prep`.
- La API carga el modelo en **cada petición** usando `mlflow.artifacts.download_artifacts()` + `joblib.load()` para evitar problemas de caché de modelo.
- Las pruebas `depends_on` esperan a que `prep` termine (`service_completed_successfully`) antes de iniciar las APIs.
- El volumen `mlruns` comparte los artefactos entre MLflow y las APIs.

---

## Pruebas

- Suite completa: `pytest -v` (resultado esperado: ~152 passed, 3 skipped; ~40 s).
- Por pregunta: `pytest -v -k p1` / `-k p2` / `-k p3`. Un solo archivo: `pytest tests/test_features_p1.py -v`.
- Las unitarias usan fixtures sintéticos de `tests/conftest.py`: no necesitan datos reales ni MLflow, corren en segundos.
- `tests/test_deepchecks_p*.py` leen datos reales (`data/interim/p1_modelado.parquet`, `data/processed/*`); si faltan, se saltan. Generarlos primero con `python -m proyecto_mibici.dataset`.
- DeepChecks 0.19.1 es incompatible con scikit-learn >= 1.6 y requiere `setuptools < 81` (ver `requirements.txt`). No actualizar `setuptools` >= 81 ni `scikit-learn` >= 1.6 sin comprobar deepchecks.
- Algunas pruebas de P1 (campeón) necesitan `models/p1_champion.pkl`; si no existe, se saltan.

---

## Lint, formato y seguridad

- Ruff (`pyproject.toml`: line-length 99, isort con `proyecto_mibici` como first-party):
  ```bash
  ruff format --check && ruff check     # (make lint)
  ruff check --fix && ruff format        # (make format)
  ```
- Seguridad opcional: `bandit -r proyecto_mibici/` y `pip-audit`.

---

## Quirk de macOS

- LightGBM requiere OpenMP. Si falla al importar: `brew install libomp`.
- `.python-version` especifica Python 3.10.22.

---

## Campeones (Model Registry)

P1: `mibici-equipo20-p1-demanda@champion` (LightGBM Tweedie); P2: `mibici-equipo20-p2-riesgo@champion` (GBM, umbral 0.5); P3: `mibici-equipo20-p3-grupos@champion` (KMeans K=4). Los `.pkl` locales son generados: no se comiteen.

---

## Archivos modificados recientemente

- `proyecto_mibici/modeling/tracking.py`: `registrar_run()` ahora usa `joblib.dump()` + `mlflow.log_artifact()` en lugar de `mlflow.sklearn.log_model()` para soportar wrappers POO no compatibles con sklearn.
- `docker/api/model_loader.py`: Ahora usa `mlflow.artifacts.download_artifacts()` + `joblib.load()` en lugar de `mlflow.pyfunc.load_model()` para cargar campeones desde MLflow.
- `docker/mlflow/Containerfile`: Agregado `--env MLFLOW_TRACKING_URI=http://mlflow:5000`.
- `docker/prep/entrypoint.sh`: Ahora apunta a `http://mlflow:5000` (red interna Docker) en lugar de `host.docker.internal:5001`.
- `docker/docker-compose.yml`: Las 3 APIs montan volumen `mlruns:/app/mlruns` y apuntan a `http://mlflow:5000`.

REVISA SIEMPRE DOCUMENTACIÓN ACTUALIZADA EN INTERNET Y EN CONTEXT7 PARA RESOLVER LOS PROBLEMAS

PUEDES USAR LA SHELL, CONTEXT7 Y BUSQUEDA EN INTERNET
TIENES QUE ESPERAR A QUE LEVANTEN LAS IMAGENES ALGUNAS TARDAN
JAMAS RECREES LAS IMAGENES SI UNA DEMORA MUCHO,
SI TARDA MUCHO  MONITORE CON EL LOG Y REVISA LA BOTACORA DE LA IMAGEN MAS TARDADA HASTA QUE TERMINE
PARA RECONSTRUIR LAS IMAGENES HAS EL COMPOSE DOWN Y LUEGO EL BUILD UP Y ESPERA A QUE TERMINE 