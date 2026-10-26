# TC5061 - Proyecto MiBici (Equipo 20)

Solución integral de Machine Learning para el análisis, modelado predictivo y segmentación funcional del sistema de bicicleta pública compartida **MiBici** (Área Metropolitana de Guadalajara). El proyecto resuelve tres problemáticas operativas centrales combinando regresión, clasificación y clustering bajo estándares de MLOps: programación orientada a objetos (POO), Pipelines formales de scikit-learn, versionado criptográfico de datos, serialización defensiva, interfaces CLI con Typer y seguimiento de experimentos con MLflow.

---

## 1. Alcance de las Preguntas del Proyecto

| Pregunta | Tipo de Problema | Objetivo de Negocio / Técnico | Algoritmo y Pipeline |
|---|---|---|---|
| **P1** | Regresión Temporal | Estimar el volumen de demanda horaria de viajes por estación para la hora inmediata ($t+1$). | `StandardScaler` + `LinearRegression` con cota inferior recortada en $\ge 0$. |
| **P2** | Clasificación Binaria | Predecir si una estación entrará en estado de desbalance crítico (vacía o saturada) en $t+1$. | `StandardScaler` + `LogisticRegression` balanceada con umbral de decisión operativo calibrable. |
| **P3** | Clustering No Supervisado | Identificar perfiles funcionales de uso en estaciones y contrastar el agrupamiento contra la división municipal (línea base). | `StandardScaler` + `KMeans` ($K=4$) evaluado con coeficientes de Silhouette y Davies-Bouldin. |

---

## 2. Organización del Repositorio

La arquitectura del proyecto sigue la estructura estándar de **Cookiecutter Data Science** con modularización independiente por pregunta:

```text
├── LICENSE                      <- Licencia de código abierto del proyecto
├── Makefile                     <- Comandos automatizados de apoyo (make data, make train)
├── README.md                    <- Documentación principal del repositorio
├── pyproject.toml               <- Metadatos del paquete local y configuración de herramientas
├── requirements.txt             <- Librerías y versiones fijadas del entorno de ejecución
├── setup.cfg                    <- Configuración de linters de código (flake8, black)
│
├── data/
│   ├── external/                <- Datos provenientes de fuentes externas secundarias
│   ├── interim/                 <- Datos intermedios en proceso de transformación
│   ├── processed/               <- Parquets procesados canónicos (trips.parquet y stations.parquet)
│   │   ├── train/               <- Partición de entrenamiento (Año 2025 completo)
│   │   └── test/                <- Partición de prueba (Abril a Septiembre de 2026)
│   ├── raw/                     <- Archivos CSV mensuales originales de MiBici (inmutables)
│   └── manifest.json            <- Registro criptográfico SHA-256 para validación de datos
│
├── docs/                        <- Documentación técnica extendida, model cards y guías metodológicas
├── models/                      <- Modelos y pipelines serializados en formato .pkl
├── notebooks/                   <- Jupyter Notebooks de análisis exploratorio (P1, P2 y P3)
├── references/                  <- Diccionarios de datos y manuales explicativos
├── reports/                     <- Tablas comparativas de métricas (.csv) y reportes generados
│   └── figures/                 <- Gráficas exportadas para reportes
│
├── tests/                       <- Suite de pruebas de integración y validación con pytest
│   └── test_pipeline.py         <- Pruebas unitarias de artefactos, modelos y pipelines
│
└── proyecto_mibici/             <- Paquete principal de Python
    ├── __init__.py              <- Metadatos del módulo principal (__version__, __author__)
    ├── config.py                <- Rutas resueltas con pathlib, constantes, URLs y semillas
    ├── dataset.py               <- Pipeline CLI para descarga, limpieza inicial y exportación a Parquet
    ├── features.py              <- Fachada unificada que expone la API pública de características
    ├── features_p1.py           <- Malla horaria, rezagos (1h, 24h, 168h), ventanas móviles y variables cíclicas
    ├── features_p2.py           <- Balance por día operativo (inicio 03:00 hrs) y estrés relativo
    ├── features_p3.py           <- Perfiles anuales de uso, distribuciones horarias y balance neto
    ├── pipeline.py              <- Orquestador maestro End-to-End (Datos + Train + Predict + Testing)
    ├── plots.py                 <- Visualizaciones generadas sobre backend headless 'Agg'
    └── modeling/
        ├── __init__.py          <- Exposición de las clases POO de modelado
        ├── model_p1.py          <- Clase POO DemandaLinealP1 (StandardScaler + LinearRegression)
        ├── model_p2.py          <- Clase POO RiesgoLogisticoP2 (StandardScaler + LogisticRegression)
        ├── model_p3.py          <- Clase POO ClusterEstacionesP3 (StandardScaler + KMeans)
        ├── train_p1.py          <- Script CLI de entrenamiento, evaluación y registro en MLflow (P1)
        ├── train_p2.py          <- Script CLI de entrenamiento, evaluación y registro en MLflow (P2)
        ├── train_p3.py          <- Script CLI de entrenamiento, contraste baseline y MLflow (P3)
        ├── predict_p1.py        <- Script CLI para inferencia sobre demanda horaria (P1)
        ├── predict_p2.py        <- Script CLI para inferencia sobre riesgo de desbalance (P2)
        ├── predict_p3.py        <- Script CLI para inferencia de clusters funcionales (P3)
        └── tracking.py          <- Módulo de integración defensiva con MLflow Tracking
```

---

## 3. Descripción de los Módulos de Código (`proyecto_mibici/`)

- **`config.py`**: Centraliza rutas absolutas mediante `pathlib.Path`, URLs de descarga de datos abiertos, semillas aleatorias (`RANDOM_SEED`), hiperparámetros base (`P2_UMBRAL`, `P3_K_ELEGIDO`), nombres de columnas predictoras y configuración de conexión a MLflow (`URI_MLFLOW_TRACKING`, `MLFLOW_EXPERIMENT_NAME`).
- **`dataset.py`**: Ejecuta el flujo ETL primario. Descarga el catálogo de estaciones y los registros históricos de viajes, maneja codificaciones conflictivas (`utf-8` / `latin-1`), normaliza nombres de columnas, tipa identificadores y fechas, exporta los archivos maestros `trips.parquet`, `stations.parquet` junto con las particiones específicas de entrenamiento (`trips_train_2025.parquet`) y prueba (`trips_test_2026_04_09.parquet`), y construye el archivo `data/manifest.json` con hashes SHA-256.
- **`features.py` (Fachada) y `features_p1.py`, `features_p2.py`, `features_p3.py`**: Implementan transformadores orientados a objetos compatibles con scikit-learn (`BaseEstimator`, `TransformerMixin`) junto con sus funciones orquestadoras:
  * **`features_p1.py`**: `LimpiadorFiltroEstacionesP1` y `GeneradorDemandaLagsP1` construyen la serie horaria completa rellenando horas sin viajes con cero, calculan rezagos (`lag_1h`, `lag_24h`, `lag_168h`), promedios móviles (`rolling_3h`, `rolling_168h`) y codificación trigonométrica del ciclo diario (`sin`/`cos`).
  * **`features_p2.py`**: `LimpiadorFlujoP2` y `ConstructorSerieRiesgoP2` eliminan viajes anómalos (<10 segundos entre estaciones distintas), estructuran el día operativo con corte a las 03:00 hrs, calculan el balance acumulado respecto a la capacidad instalada (`dpcapacity`) y etiquetan el estado de riesgo para la hora $t+1$.
  * **`features_p3.py`**: `LimpiadorPerfilP3` depura duraciones fuera de rango (>180 min) y sintetiza el comportamiento anual de cada estación en proporciones de uso por bloque horario, porcentaje de viajes en fin de semana, duración media, volumen diario en escala logarítmica y balance neto relativo.
- **`pipeline.py`**: Orquestador maestro que automatiza todo el ciclo End-to-End del proyecto en secuencia (Ingesta ETL, entrenamiento de P1, P2 y P3 con registro en MLflow, inferencia CLI de prueba y suite de validación con `pytest`).
- **`plots.py`**: Módulo de visualización configurado con backend `Agg` (sin requerir monitor gráfico) que genera las gráficas de evaluación para MLflow: series temporales y dispersión real vs. predicho (P1), matrices de confusión y curvas Precision-Recall (P2), y mapas geográficos de estaciones coloreados por clúster o municipio (P3).
- **`modeling/model_p1.py`, `model_p2.py`, `model_p3.py`**: Clases POO (`DemandaLinealP1`, `RiesgoLogisticoP2`, `ClusterEstacionesP3`) que encapsulan objetos `sklearn.pipeline.Pipeline` uniendo el escalamiento `StandardScaler` con el estimador final, ofreciendo métodos estandarizados `.fit()`, `.predict()`, `.save()` y `.load()`.
- **`modeling/tracking.py`**: Gestiona la trazabilidad en MLflow registrando el hash SHA-256 de los datos, el hash del commit de Git, hiperparámetros, métricas desglosadas por partición (`Train`, `Validación`, `Test`), artefactos de figuras, tablas CSV y el modelo con su firma (`infer_signature`), guardando simultáneamente copias locales de las figuras en `reports/figures/`.

---

## 4. Instalación y Configuración del Entorno

### Requisitos previos
- **Python 3.10 o 3.11** instalado en el sistema operativo.
- **Git** configurado en la terminal.

### Paso 1: Clonar el proyecto
```bash
git clone https://github.com/ariosi86/TC5061-Proyecto-Equipo20.git
cd TC5061-Proyecto-Equipo20
```

### Paso 2: Crear el entorno virtual
```bash
python -m venv .venv
```

### Paso 3: Activar el entorno virtual

- **macOS y Linux:**
  ```bash
  source .venv/bin/activate
  ```

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
  *(En caso de bloqueo por políticas de ejecución en PowerShell, ejecutar antes: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

- **Windows (Símbolo del sistema / CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

### Paso 4: Instalar dependencias y montar el paquete local
```bash
# 1. Actualizar pip
pip install --upgrade pip

# 2. Instalar dependencias fijadas en requirements.txt
pip install -r requirements.txt

# 3. Instalar el paquete local proyecto_mibici en modo editable
pip install -e .

# 4. Registrar el kernel para uso en Jupyter Notebooks
python -m ipykernel install --user --name=mibici-venv --display-name="Python (MiBici .venv)"
```

---

## 5. Descarga y Procesamiento de Datos (Ingesta Reproducible)

Con el entorno virtual activo `(.venv)`, ejecuta el módulo de ingesta desde la raíz del proyecto:

```bash
python -m proyecto_mibici.dataset
```

**Salidas generadas automáticamente:**
1. Descarga de los archivos CSV originales en `data/raw/`.
2. Generación de los conjuntos procesados en formato Apache Parquet:
   * Maestro general: `data/processed/trips.parquet`
   * Catálogo de estaciones: `data/processed/stations.parquet`
   * Entrenamiento (2025): `data/processed/train/trips_train_2025.parquet`
   * Prueba (Abr-Sep 2026): `data/processed/test/trips_test_2026_04_09.parquet`
3. Creación de `data/manifest.json` con los hashes criptográficos SHA-256, conteo total de registros y rango de fechas.

---

## 6. Entrenamiento y Registro en MLflow (`train_p*.py`)

Cada script de entrenamiento construye su tabla de características desde los archivos Parquet procesados, aplica la partición temporal correspondiente para evitar fuga de información (*data leakage*), ajusta el Pipeline, guarda el artefacto `.pkl` en `models/` y registra la corrida completa en MLflow.

### Requisitos de MLflow
Antes de ejecutar los entrenamientos, inicia el servidor local de MLflow en una terminal secundaria con el entorno virtual activo:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
```

*La URI de seguimiento está definida en `proyecto_mibici/config.py` mediante `URI_MLFLOW_TRACKING` (`http://localhost:5000` o `sqlite:///mlflow.db`). Puedes consultar los experimentos en el navegador ingresando a `http://localhost:5000`.*

---

### P1: Entrenamiento de Demanda Horaria (Regresión)
```bash
python -m proyecto_mibici.modeling.train_p1
```

- **Parámetros registrados en MLflow:**
  * `pregunta`: `"P1"`
  * `modelo`: `"DemandaLinealP1"`
  * `features`: Lista de rezagos (`lag_1h`, `lag_24h`, `lag_168h`), medias móviles (`rolling_3h`, `rolling_168h`) y variables de calendario (`target_hour_sin`, `target_hour_cos`, `target_is_weekend`).
  * `train_inicio`, `train_fin`, `validation_inicio`, `validation_fin`, `test_inicio`, `test_fin`: Límites cronológicos exactos de cada partición.
  * `datos_version`: Hash SHA-256 corto del dataset procesado.
- **Métricas registradas:** `train_mae`, `train_rmse`, `test_mae`, `test_rmse`.
- **Artefactos generados:**
  * Local: `models/p1_linear_model.pkl`
  * MLflow: Pipeline scikit-learn con firma de entrada/salida, `manifest.json`, gráfica de serie temporal (`penitenciaria_real_vs_pred_v2_lineal_rolling.png`) y gráfico de dispersión global (`scatter_test_v2_lineal_rolling.png`).

---

### P2: Entrenamiento de Riesgo de Desbalance (Clasificación)
```bash
python -m proyecto_mibici.modeling.train_p2
```

- **Parámetros registrados en MLflow:**
  * `pregunta`: `"P2"`
  * `modelo`: `"RiesgoLogisticoP2"`
  * `umbral`: `0.5` (umbral de saturación/vaciamiento respecto a la capacidad de la estación).
  * `variables`: Variables predictoras (`hora_del_dia`, `fin_de_semana`, `acumulado_relativo`, `balance_ult_3h_relativo`).
  * `train_inicio`, `train_fin`, `validation_inicio`, `validation_fin`, `test_inicio`, `test_fin` y `datos_version`.
- **Métricas registradas:** `train_precision`, `train_recall`, `train_f1_score`, `train_pr_auc`, `test_precision`, `test_recall`, `test_f1_score`, `test_pr_auc`.
- **Artefactos generados:**
  * Local: `models/p2_logistic_model.pkl`
  * MLflow: Pipeline scikit-learn con firma de datos, `manifest.json`, matriz de confusión (`matriz_confusion_v1_logistic.png`) y curva Precision-Recall (`curva_pr_v1_logistic.png`).

---

### P3: Entrenamiento de Clustering de Estaciones (No Supervisado)
```bash
python -m proyecto_mibici.modeling.train_p3
```

- **Parámetros registrados en MLflow:**
  * `pregunta`: `"P3"`
  * `modelo`: Registra dos corridas para contraste directo: `"AgrupacionMunicipio"` (Línea Base geográfica) y `"ClusterEstacionesP3"` (KMeans funcional).
  * `n_clusters`: `4` ($K$ elegido).
  * `n_init`: `10`.
  * `semilla`: `RANDOM_SEED`.
  * `variables`: Proporciones por bloque horario (`pct_madrugada`, `pct_manana`, `pct_mediodia`, `pct_tarde`, `pct_noche`), `pct_fin_de_semana`, `duracion_media`, `viajes_dia_log` y `balance_neto`.
  * `min_viajes_estacion`: `100`.
  * `duracion_max_min`: `180`.
- **Métricas registradas:** `train_silhouette`, `train_davies_bouldin`, `test_silhouette`, `test_davies_bouldin` en ambas corridas.
- **Artefactos generados:**
  * Local: `models/p3_kmeans_model.pkl` y tabla comparativa `reports/p3_comparacion.csv`.
  * MLflow: Pipeline scikit-learn con firma de datos, mapa de estaciones por clúster KMeans (`mapa_grupos_kmeans.png`) y mapa de la línea base municipal (`mapa_grupos_baseline.png`).

---

## 7. Inferencia por Línea de Comandos (`predict_p*.py`)

Los scripts de inferencia cargan los Pipelines serializados desde `models/` (aplicando internamente el `StandardScaler` ajustado en entrenamiento) y generan predicciones sobre nuevos datos mediante una interfaz CLI construida con **Typer**.

### Parámetros de los Scripts de Inferencia

| Opción CLI | Tipo | Obligatorio | Aplica a | Descripción |
|---|---|---|---|---|
| `--input`, `-i` | `PATH` | **Sí** | P1, P2, P3 | Ruta al archivo `.parquet` o `.csv` con las variables de entrada del modelo. |
| `--output`, `-o` | `PATH` | No | P1, P2, P3 | Ruta de salida (`.parquet` o `.csv`) para guardar el DataFrame con las predicciones anexadas. Si se omite, imprime una vista previa de 10 filas en consola. |
| `--umbral`, `-u` | `FLOAT` | No | Solo P2 | Umbral de decisión entre `0.0` y `1.0` para clasificar una estación en riesgo a partir de la probabilidad predicha (valor por defecto: `0.5`). |

---

### Ejemplos de Ejecución de Inferencia

#### 1. Inferencia de Demanda Horaria (P1)
Anexa la columna `demanda_predicha` sobre un archivo con la matriz de características horarias (rezagos y componentes temporales):

```bash
# Guardar resultados en archivo Parquet
python -m proyecto_mibici.modeling.predict_p1 \
  --input reports/predicciones_p1_test.parquet

# Mostrar vista previa en terminal sin guardar archivo
python -m proyecto_mibici.modeling.predict_p1 \
  --input reports/predicciones_p1_test.parquet \
  --output reports/mis_predicciones_p1.parquet



#### 2. Inferencia de Riesgo de Desbalance (P2)
Anexa las columnas `prob_riesgo` (probabilidad continua) y `pred_riesgo` (etiqueta booleana evaluada contra `--umbral`):

```bash
# Inferencia con umbral estándar (0.50) y vista previa en terminal
python -m proyecto_mibici.modeling.predict_p2 \
  --input reports/predicciones_p2_test.parquet \
  --umbral 0.5

# Inferencia con umbral sensible (0.35) exportando a CSV
python -m proyecto_mibici.modeling.predict_p2 \
  --input reports/predicciones_p2_test.parquet \
  --umbral 0.35 \
  --output reports/predicciones_p2_sensible.csv
```

#### 3. Inferencia de Clusters Funcionales (P3)
Escala las características y asigna la etiqueta de clúster a partir del catálogo procesado de estaciones:

```bash
# Asignar grupos funcionales y mostrar distribución en terminal
python -m proyecto_mibici.modeling.predict_p3 \
  --input reports/predicciones_p3_test.csv

# Exportar asignación final de grupos a una ruta personalizada
python -m proyecto_mibici.modeling.predict_p3 \
  --input reports/predicciones_p3_test.csv \
  --output reports/estaciones_clusterizadas.csv
```

---

## 8. Orquestador End-to-End e Integración de Pruebas (`pipeline.py`)

El proyecto incluye un script orquestador maestro que automatiza todo el ciclo de vida del sistema en una sola ejecución. Se encarga de coordinar la ingesta ETL, los tres entrenamientos con registro en MLflow, las pruebas de inferencia CLI y la validación con `pytest`.

Este script se creo para ejecutar la descarga, los entrenamientos, los tests y la generación de reportes y figuras, además de enviar a mlflow las configuraciones e hiperparametros de los 3 modelos.

Se genero de manera única de probar los modelos y que todo el proyecto funcione con metadatos fijos desde el script

### Parámetros del Orquestador Maestro

| Bandera CLI | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `--skip-data` | Booleano | `False` | Omite la descarga y el procesamiento inicial de datos si los archivos Parquet ya existen en `data/processed/`. |
| `--skip-tests` | Booleano | `False` | Omite la ejecución automática de las pruebas de inferencia y la suite de `pytest`. |

### Comandos de Ejecución Integral

* **Ejecutar todo el flujo desde cero (Ingesta + Entrenamiento + Inferencia de prueba + Pytest):**
  ```bash
  python -m proyecto_mibici.pipeline
  ```


* **Ejecutar únicamente ingesta y entrenamiento (forzando regenerar los archivos parquet):**
  ```bash
  python -m proyecto_mibici.pipeline --force-data
  ```

* **Ejecutar únicamente entrenamiento, inferencia y tests (omitiendo la descarga de datos):**
  ```bash
  python -m proyecto_mibici.pipeline --skip-data
  ```

* **Ejecutar únicamente ingesta y entrenamiento (omitiendo las pruebas finales):**
  ```bash
  python -m proyecto_mibici.pipeline --skip-tests
  ```


## 9. Entrega 2: Nuevos modelos, MLflow Project y selección del mejor modelo

La Entrega 2 extiende la Entrega 1 con modelos adicionales para P1 (Poisson, LightGBM), búsquedas de hiperparámetros organizadas como nested runs en MLflow, el proyecto MLflow para reproducción completa, el orquestador `run_train.py` y el registro de los mejores modelos en el MLflow Model Registry.

---

### 9.1 Nuevos modelos candidatos en P1 (Demanda Horaria)

Además del baseline de Regresión Lineal, P1 incluye candidatos de Regresión de Poisson y LightGBM (Poisson y Tweedie), con búsquedas anidadas de hiperparámetros.

Ejecuta un modelo individual:

```bash
python -m proyecto_mibici.modeling.train_p1 --model <nombre> --lr <learning_rate> --alpha <alpha> --max_iter <max_iter>
```

| Nombre modelo | `--model` | Descripción |
|---|---|---|
| Baseline (lineal v2) | `baseline` | Regresión Lineal con medias móviles |
| Poisson lineal (candidato) | `poisson` | Regresión de Poisson con `lr=0.05`, `alpha=0.0001`, `max_iter=300` |
| LightGBM-Poisson (candidato) | `lgbm_poisson` | LightGBM con objective=poisson |
| LightGBM-Poisson (nested search) | `lgbm_p_search` | Búsqueda anidada de hiperparámetros (objective=poisson) |
| LightGBM-Tweedie (candidato) | `lgbm_tweedie` | LightGBM con objective=tweedie |
| LightGBM-Tweedie (nested_search) | `lgbm_t_search` | Búsqueda anidada de hiperparámetros (objective=tweedie) |

Tags registrados en MLflow por modelo:

| Modelo | `model_family` | `stage_of_work` | Extra |
|---|---|---|---|
| baseline | linear | baseline | `baseline=true` |
| poisson | poisson_regression | candidato | |
| lgbm_poisson | candidato | lightgbm | |
| lgbm_p_search | search | lightgbm | |
| lgbm_tweedie | candidato | lightgbm | |
| lgbm_t_search | search | lightgbm | |

> **macOS:** la librería `lightgbm` requiere OpenMP para soporte multihilo. Instala con Homebrew:
> ```bash
> brew install libomp
> ```

---

### 9.2 MLflow Project y tracking avanzado

El proyecto está empaquetado como un **MLflow Project** para reproducir la ingesta y los entrenamientos desde cualquier entorno.

**Archivos del proyecto MLflow:**
- `MLproject`: define los entry points `ingest`, `train_p1`, `train_p2`, `train_p3` y `train`, con sus parámetros y variables de entorno.
- `python_env.yaml`: definición del entorno Python con las dependencias fijas.
- `run_train.py`: orquestador CLI que lanza todas las versiones de una pregunta mediante `mlflow run`.

#### 9.2.1 Levantar el servidor de MLflow

Desde la raíz del repositorio, con el entorno virtual activado:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
```

Accede a la interfaz en `http://localhost:5000`. En la terminal de trabajo define la URI:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

> Los hiperparámetros siempre se pasan como `-P clave=valor`. MLflow no lee la sección `variables` del `MLproject` para la URI de tracking; los scripts usan `URI_MLFLOW_TRACKING` desde `config.py`.

#### 9.2.2 Reproducir la ingesta de datos

```bash
mlflow run . -e ingest --env-manager local --experiment-name mibici-equipo20
```

Equivalente a `python -m proyecto_mibici.dataset`.

#### 9.2.3 Entrenar modelos por pregunta

Hay dos formas equivalentes: con `run_train.py` o con `mlflow run` directamente. Ambos quedan registrados en MLflow con `source_type=PROJECT`.

**P1 — Demanda horaria (regresión):**

```bash
mlflow run . -e train_p1 --env-manager local --experiment-name mibici-equipo20 -P modelo=baseline
mlflow run . -e train_p1 --env-manager local --experiment-name mibici-equipo20 -P modelo=poisson -P lr=0.05 -P alpha=0.0001 -P max_iter=300
mlflow run . -e train_p1 --env-manager local --experiment-name mibici-equipo20 -P modelo=lgbm_tweedie
mlflow run . -e train_p1 --env-manager local --experiment-name mibici-equipo20 -P modelo=lgbm_t_search
mlflow run . -e train_p1 --env-manager local --experiment-name mibici-equipo20 -P modelo=lgbm_p_search
```

**P2 — Riesgo de desbalance (clasificación):**

```bash
mlflow run . -e train_p2 --env-manager local --experiment-name mibici-equipo20 -P modelo=baseline
mlflow run . -e train_p2 --env-manager local --experiment-name mibici-equipo20 -P modelo=logistic -P umbral=0.5 -P class_weight=balanced -P max_iter=1000
mlflow run . -e train_p2 --env-manager local --experiment-name mibici-equipo20 -P modelo=logistic -P umbral=0.5 -P class_weight=None -P max_iter=1000
mlflow run . -e train_p2 --env-manager local --experiment-name mibici-equipo20 -P modelo=logistic_search
mlflow run . -e train_p2 --env-manager local --experiment-name mibici-equipo20 -P modelo=gbm
mlflow run . -e train_p2 --env-manager local --experiment-name mibici-equipo20 -P modelo=gbm_search
```

**P3 — Clustering de estaciones (no supervisado):**

```bash
mlflow run . -e train_p3 --env-manager local --experiment-name mibici-equipo20 -P modelo=baseline
mlflow run . -e train_p3 --env-manager local --experiment-name mibici-equipo20 -P modelo=kmeans -P n_clusters=4 -P n_init=10
mlflow run . -e train_p3 --env-manager local --experiment-name mibici-equipo20 -P modelo=kmeans_search -P n_init=10
```

#### 9.2.4 Flujo completo recomendado

```bash
# 1. Levantar MLflow (terminal aparte)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000

# 2. En la terminal de trabajo
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# 3. Ingesta de datos
mlflow run . -e ingest --env-manager local --experiment-name mibici-equipo20

# 4. Entrenar modelos para producción
python run_train.py train --pregunta p1 --versiones baseline,lgbm_tweedie
python run_train.py train --pregunta p2 --versiones baseline,logistic_sin_peso
python run_train.py train --pregunta p3 --versiones baseline,kmeans_k4
```

Cada run queda registrado en MLflow con sus parámetros, métricas, etiquetas (git commit, versión de datos, equipo) y artefactos (modelo con firma, figuras, `manifest.json`). El campo `Source` es consistente (`source_type=PROJECT`) en todas las corridas lanzadas vía `mlflow run`.

---

### 9.3 Orquestador `run_train.py`

`run_train.py` es un **atajo CLI (Typer)** para ejecutar todos los entrenamientos de una pregunta en un solo comando, en lugar de lanzar cada `mlflow run` uno a uno. **No contiene lógica de entrenamiento**: solo despacha cada versión candidata como un proyecto MLflow independiente mediante `mlflow run . -e train_pX -P ...`.

El entry point de MLflow (`train_p1`, `train_p2`, `train_p3`) invoca directamente al módulo de entrenamiento correspondiente, evitando recursión.

**Comando:**

```bash
python run_train.py train --pregunta <p1|p2|p3> [--versiones <lista>]
```

`--versiones` acepta `todos` (por defecto) o una lista separada por comas.

**Versiones candidatas por pregunta:**

| Pregunta | Versiones |
|---|---|
| P1 | baseline, poisson, lgbm_poisson, lgbm_poisson_search, lgbm_tweedie, lgbm_tweedie_search |
| P2 | baseline, logistic_balanced, logistic_sin_peso, logistic_umbral_sensible, logistic_search, gbm, gbm_search |
| P3 | baseline, kmeans_search, kmeans_k3, kmeans_k4, kmeans_k5, kmeans_k6 |

**Ejemplos:**

```bash
# Todas las versiones de P2
python run_train.py train --pregunta p2

# Solo baseline y el mejor candidato de P2
python run_train.py train --pregunta p2 --versiones baseline,logistic_sin_peso

# Búsqueda anidada de P2
python run_train.py train --pregunta p2 --versiones logistic_search
```

**Requisitos previos:**
1. Servidor de MLflow levantado (terminal aparte).
2. Datos procesados en `data/processed/` (generados con `python -m proyecto_mibici.dataset` o `mlflow run . -e ingest`).
3. Dependencias instaladas (`pip install -r requirements.txt`).
4. Todos los comandos `mlflow run` requieren `--env-manager local --experiment-name mibici-equipo20` porque MLflow 3.x intenta crear un entorno aislado con pyenv; estas flags hacen que use el `.venv` existente.

---

### 9.4 Registro en el Model Registry (`run_mejor_modelo.py`)

`run_mejor_modelo.py` registra en el MLflow Model Registry el modelo elegido de cada pregunta, conservando su linaje (experimento, corrida de origen, commit y versión de datos) y agregando una descripción a cada modelo.

**Modelos registrados:**

| Pregunta | Nombre en el Registry | Modelo elegido |
|---|---|---|
| P1 | `mibici-equipo20-p1-demanda` | LightGBM Regresión Tweedie |
| P2 | `mibici-equipo20-p2-riesgo` | Regresión Logística sin peso, umbral 0.5 |
| P3 | `mibici-equipo20-p3-grupos` | KMeans funcional K=4 |

**Ejecución:**

```bash
python run_mejor_modelo.py
```

El script es idempotente: si ya existe una versión para la misma corrida de origen, la reusa y actualiza su descripción y tags en lugar de duplicarla.

Todos los modelos se promueven al stage `Production`. Quedan accesibles en MLflow (sección Models) y pueden cargarse para inferencia con:

```python
import mlflow
modelo = mlflow.pyfunc.load_model("models:/mibici-equipo20-p1-demanda@champion")
```


