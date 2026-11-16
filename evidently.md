# Monitoreo de drift con Evidently — `EvidentlyMonitor`

Documentación de integración para el **Entregable 8** (monitoreo, drift y A/B testing) del proyecto MiBici (Equipo 20, TC5061).

Este archivo explica cómo usar la clase `EvidentlyMonitor` definida en
`proyecto_mibici/monitoring/evidently_monitor.py` para generar reportes de
data drift entre un conjunto de **referencia** (entrenamiento) y un conjunto
**actual** (producción / periodo posterior), y cómo integrarla en los
testings que cada integrante elabore.

> **No modifica ni reemplaza** las entregas anteriores. Se integra al paquete
> `proyecto_mibici/monitoring/` junto a `drift_report.py`, `ab_test_p1.py` y
> `umbral_ab_p1.py`.

---

## 1. ¿Qué resuelve esta clase?

- Compara dos datasets (referencia vs. actual) con **Evidently** y produce:
  - Un reporte HTML interactivo (data drift + data summary).
  - Una tabla CSV de scores complementaria (medias, std, Wasserstein normalizado y bandera de drift).
- Registra opcionalmente el experimento en **MLflow** (`mibici-equipo20-p1-monitoreo`).
- Es **genérica**: sirve para P1 (regresión), P2 (clasificación), P3 (clustering) o cualquier otro dataset que pases con sus columnas.

---

## 1.1. API de Evidently Platform en `http://127.0.0.1:8000/`

La clase `EvidentlyMonitor` se conecta al servidor de Evidently UI levantado en `http://127.0.0.1:8000/` mediante `RemoteWorkspace` (de `evidently.ui.workspace`). Esta es la **API real de Evidently Platform**, que acepta escritura:

| Operación | Endpoint usado por `RemoteWorkspace` |
|-----------|--------------------------------------|
| Verificar servidor | `GET /api/version` |
| Crear proyecto | `POST /api/v2/projects` |
| Buscar proyectos | `GET /api/projects/search/{name}` |
| Listar proyectos | `GET /api/v2/projects` |
| Subir dataset | `POST /api/datasets/upload` |
| Subir reporte (run) | `POST /api/v2/snapshots/{project_id}` |

Si el servidor está disponible, la clase registra **proyecto, datasets y reportes directamente en Evidently Platform**. Cada reporte subido obtiene una URL para verlo en la consola web (ej. `http://127.0.0.1:8000/projects/{id}/reports/{snapshot_id}`).

Si el servidor **no está disponible**, la clase hace fallback automático a registro local JSON bajo `reports/drift/registro/` para trazabilidad y para el reporte PDF del entregable.

Para abrir automáticamente el HTML tras generar el reporte, usa el parámetro `abrir_navegador=True`:

```python
monitor = EvidentlyMonitor(
    numerical_columns=["edad", "saldo"],
    categorical_columns=["sexo"],
    nombre="mi_drift_test",
    abrir_navegador=True,  # abre el HTML en el navegador al terminar
)
resumen = monitor.comparar(df_ref, df_cur, log_mlflow=False)
```

O bien, abre el archivo manualmente:

```bash
open reports/drift/drift_mi_drift_test.html
```

Si el servidor de Evidently UI no está levantado, inícialo con:

```bash
evidently ui  # inicia el servicio de UI de Evidently en el puerto 8000
```

Para desactivar la conexión a la API y usar solo registro local, pasa `usar_api_evidently=False`:

```python
monitor = EvidentlyMonitor(
    numerical_columns=["edad", "saldo"],
    usar_api_evidently=False,  # solo registro local JSON
)
```

---

## 2. Instalación y dependencias

La dependencia ya está en `requirements.txt`:

```text
evidently>=0.7.23
```

Si faltara, instalar:

```bash
pip install evidently>=0.7.23
```

El proyecto ya usa `mlflow`, `pandas`, `numpy`, `scipy` y `loguru`, que son las demás dependencias de la clase.

---

## 3. Importación básica

```python
from proyecto_mibici.monitoring import EvidentlyMonitor
```

La clase se exporta desde `proyecto_mibici/monitoring/__init__.py`.

---

## 4. Instanciación

```python
monitor = EvidentlyMonitor(
    numerical_columns=[...],      # lista de columnas numéricas
    categorical_columns=[...],    # lista opcional de columnas categóricas/binarias
    num_method="wasserstein",     # método numérico de Evidently
    cat_method="jensenshannon",   # método categórico de Evidently
    num_threshold=0.1,            # umbral de drift numérico
    cat_threshold=0.1,            # umbral de drift categórico
    n_muestra=100_000,            # filas máximas por conjunto (Evidently es costoso)
    random_seed=42,               # semilla para muestreo reproducible
    out_dir=None,                 # carpeta de salida (por defecto reports/drift/)
)
```

### Parámetros más usados

| Parámetro | Significado | Valor por defecto |
|-----------|-------------|-------------------|
| `numerical_columns` | Columnas numéricas a comparar | — (requerido) |
| `categorical_columns` | Columnas categóricas/binarias | `[]` |
| `num_threshold` | Umbral para marcar drift numérico | `0.1` |
| `cat_threshold` | Umbral para marcar drift categórico (Evidently) | `0.1` |
| `n_muestra` | Filas máximas por conjunto antes de muestrear | `100 000` |
| `out_dir` | Carpeta donde se guardan HTML y CSV | `reports/drift/` |
| `abrir_navegador` | Abrir el HTML en el navegador tras generar el reporte | `False` |

| `n_muestra` | Filas máximas por conjunto antes de muestrear | `100 000` |
| `out_dir` | Carpeta donde se guardan HTML y CSV | `reports/drift/` |
| `abrir_navegador` | Abrir el HTML en el navegador tras generar el reporte | `False` |
| `registrar_local` | Guardar metadatos de proyecto/datasets/reportes como JSON | `True` |
| `usar_api_evidently` | Registrar en la API de Evidently Platform en `http://127.0.0.1:8000` | `True` |

### Parámetros adicionales de `comparar()`

Además de los parámetros del constructor, el método `comparar()` acepta:

| Parámetro | Significado | Valor por defecto |
|-----------|-------------|-------------------|
| `mlflow_run_name` | Nombre del run en MLflow | `drift_<nombre>` |
| `log_mlflow` | Si se registra en MLflow | `True` |
| `abrir_navegador` | Sobrescribe `self.abrir_navegador` para esta llamada | `None` (usa el del constructor) |
| `registrar_proyecto_nombre` | Nombre del proyecto a registrar (opcional) | `None` |
| `registrar_proyecto_descripcion` | Descripción del proyecto | `None` |

---

## 5. Ejecutar la comparación con registro

```python
resumen = monitor.comparar(
    referencia=df_ref,
    actual=df_cur,
    nombre="p1_ref2025_vs_oct2026",
    mlflow_run_name="p1_drift_oct2026",  # opcional
    log_mlflow=True,                      # opcional
    registrar_proyecto_nombre="p1-demanda-horaria",  # opcional
    registrar_proyecto_descripcion="Monitoreo P1: referencia 2025 vs octubre 2026",
)
```

`comparar` devuelve un `dict` con las siguientes claves:

```text
{
    "nombre": str,
    "n_ref": int,
    "n_actual": int,
    "vars_con_drift": int,
    "html": str,        # ruta al .html de Evidently
    "scores_csv": str,  # ruta al .csv de scores
    "proyecto_id": str | None,
    "proyecto_nombre": str | None,
    "dataset_referencia_id": str,
    "dataset_actual_id": str,
    "reporte_id": str,
}
```

También puedes obtener la tabla de scores por separado:

```python
scores = monitor.tabla_scores(df_ref, df_cur)
print(scores)
```

---

## 5.1. Registro de proyecto, datasets y reportes

La clase incluye una capa de registro que documenta todo el flujo para trazabilidad y para el reporte PDF del entregable. Los registros se guardan como archivos JSON bajo `out_dir / "registro/"`:

```
reports/drift/registro/
├── proyectos/          # Un JSON por proyecto
├── datasets/           # Un JSON por dataset (referencia y actual)
└── reportes/           # Un JSON por reporte generado
```

### Registro de proyecto

Se llama automáticamente si pasas `registrar_proyecto_nombre` a `comparar()`:

```python
resumen = monitor.comparar(
    ref, cur,
    nombre="p1_ref2025_vs_oct2026",
    registrar_proyecto_nombre="p1-demanda-horaria",
    registrar_proyecto_descripcion="Monitoreo P1: referencia 2025 vs octubre 2026",
    log_mlflow=False,
)
```

Esto crea `reports/drift/registro/proyectos/p1-demanda-horaria.json` con metadatos como nombre, descripción, columnas, umbrales, fecha y si la API de Evidently Platform estaba disponible.

También puedes registrar un proyecto explícitamente:

```python
proyecto_id = monitor.registrar_proyecto(
    "mi-proyecto",
    descripcion="Descripción del proyecto",
)
```

### Registro de datasets

Se llaman automáticamente dentro de `comparar()` para referencia y actual. Cada dataset se registra con sus estadísticas básicas (filas, medias, std, min, max para numéricas; valores únicos para categóricas):

```python
# Dentro de comparar(), se registra automáticamente:
#   dataset_referencia_id = monitor.registrar_dataset(ref, "ref-<nombre>", tipo="referencia")
#   dataset_actual_id     = monitor.registrar_dataset(cur, "cur-<nombre>", tipo="actual")
```

También puedes registrar datasets manualmente:

```python
dataset_id = monitor.registrar_dataset(
    df_referencia,
    nombre="ref-2025",
    tipo="referencia",
)
```

### Registro de reportes

Se llama automáticamente al final de `comparar()`. Guarda rutas al HTML y CSV, número de variables con drift, filas usadas y fecha:

```python
# Dentro de comparar(), se registra automáticamente:
#   reporte_id = monitor.registrar_reporte(
#       nombre="p1_ref2025_vs_oct2026",
#       html_path=html_path,
#       scores_csv=scores_csv,
#       vars_con_drift=3,
#       n_ref=150,
#       n_actual=150,
#   )
```

### API de Evidently Platform (`http://127.0.0.1:8000/`)

La clase se conecta al servidor de Evidently UI en `http://127.0.0.1:8000/` mediante `RemoteWorkspace` para registrar proyectos, datasets y reportes directamente en la plataforma. Cada reporte subido obtiene una URL para verlo en la consola web.

Si el servidor no está disponible, la clase hace fallback a registro local JSON. El campo `registrado_en_evidently` en cada JSON indica si el registro se realizó en la API de Evidently Platform o solo localmente.

---

## 6. Ejemplos por pregunta del proyecto

A continuación, ejemplos concretos usando las configuraciones de `proyecto_mibici/config.py`.

### 6.1 P1 — Demanda horaria (regresión)

```python
import pandas as pd
from proyecto_mibici.config import (
    P1_ACTUAL_FIN, P1_FEATURES_V2, P1_TARGET, P1_TEST_FIN, P1_VAL_INICIO,
)
from proyecto_mibici.features import construir_dataset_p1
from proyecto_mibici.monitoring import EvidentlyMonitor

# 1. Cargar datos de modelado P1
df_p1, _ = construir_dataset_p1()
t = df_p1["target_hour"]

# 2. Definir referencia y actual
df_ref = df_p1[t < pd.Timestamp(P1_VAL_INICIO)]          # entrenamiento (2025)
df_cur = df_p1[(t >= pd.Timestamp(P1_TEST_FIN)) &        # octubre 2026 (producción)
               (t < pd.Timestamp(P1_ACTUAL_FIN))]

# 3. Columnas a monitorear (todas excepto la booleana)
numericas = [c for c in P1_FEATURES_V2 if c != "target_is_weekend"]
categoricas = ["target_is_weekend"]

# 4. Instanciar y comparar
monitor = EvidentlyMonitor(
    numerical_columns=numericas,
    categorical_columns=categoricas,
    num_threshold=0.1,
    cat_threshold=0.1,
    n_muestra=100_000,
    nombre="p1",
)
resumen = monitor.comparar(
    df_ref, df_cur,
    nombre="p1_ref2025_vs_oct2026",
    mlflow_run_name="p1_drift_oct2026",
    log_mlflow=True,
)

print(resumen)
# Salida esperada (ejemplo):
# {'nombre': 'p1_ref2025_vs_oct2026', 'n_ref': 100000, 'n_actual': 100000,
#  'vars_con_drift': 2, 'html': 'reports/drift/drift_p1_ref2025_vs_oct2026.html',
#  'scores_csv': 'reports/drift/drift_p1_ref2025_vs_oct2026_scores.csv'}
```

### 6.2 P2 — Riesgo de desbalance (clasificación)

```python
import pandas as pd
from proyecto_mibici.config import P2_VARIABLES, P2_OBJETIVO
from proyecto_mibici.monitoring import EvidentlyMonitor

# df_ref y df_cur deben venir de tu fuente de datos P2 (interim/processed)
# Ejemplo genérico; ajusta los filtros de fecha según tu periodo.

numericas = [c for c in P2_VARIABLES if c not in ("fin_de_semana",)]
categoricas = ["fin_de_semana"]

monitor = EvidentlyMonitor(
    numerical_columns=numericas,
    categorical_columns=categoricas,
    num_threshold=0.15,
    cat_threshold=0.10,
    n_muestra=50_000,
    nombre="p2",
)
resumen = monitor.comparar(
    df_ref, df_cur,
    nombre="p2_ref_vs_actual",
    mlflow_run_name="p2_drift_ab_test",
    log_mlflow=True,
)
```

### 6.3 P3 — Clustering funcional de estaciones (no supervisado)

```python
from proyecto_mibici.config import P3_VARIABLES
from proyecto_mibici.monitoring import EvidentlyMonitor

monitor = EvidentlyMonitor(
    numerical_columns=P3_VARIABLES,   # todas numéricas en P3
    categorical_columns=[],
    num_threshold=0.12,
    cat_threshold=0.10,
    n_muestra=50_000,
    nombre="p3",
)
resumen = monitor.comparar(
    df_ref, df_cur,
    nombre="p3_ref_vs_actual",
    mlflow_run_name="p3_drift_clusters",
    log_mlflow=True,
)
```

---

## 7. Interpretar los resultados

### 7.1 Reporte HTML de Evidently

El archivo `drift_<nombre>.html` contiene:

- **Data Drift Preset**: qué variables presentan drift y con qué severidad, según el método y umbral elegidos.
- **Data Summary Preset**: distribuciones, estadísticos y valores faltantes de referencia vs. actual.

Abrilo en el navegador:

```bash
open reports/drift/drift_p1_ref2025_vs_oct2026.html
```

### 7.2 Tabla de scores (CSV)

Generada por `tabla_scores`. Columnas principales:

| Columna | Significado |
|---------|-------------|
| `variable` | Nombre de la variable |
| `tipo` | `numerica` o `categorica` |
| `media_ref` / `media_actual` | Media en referencia y actual |
| `std_ref` / `std_actual` | Desviación estándar |
| `wasserstein_norm` | Distancia de Wasserstein normalizada (solo numéricas) |
| `drift` | `True` si `wasserstein_norm > num_threshold` |

Una variable con `drift=True` indica que su distribución cambió respecto al entrenamiento y puede afectar el desempeño del modelo.

### 7.3 Métricas en MLflow

Si `log_mlflow=True`, se registra en el experimento `mibici-equipo20-p1-monitoreo`:

- Parámetros: `n_muestra`, `num_method`, `cat_method`, `num_threshold`, `cat_threshold`, `n_ref`, `n_actual`, `nombre`.
- Métrica: `vars_con_drift` (cuántas variables numéricas superaron el umbral).
- Artefactos: carpeta `drift_<nombre>/` con el HTML y el CSV.

---

## 8. Integración con A/B testing y Continuous Training

Esta clase cubre la parte de **detección de drift** de la actividad. Se complementa con:

- `proyecto_mibici/monitoring/ab_test_p1.py` — compara champion vs challenger sobre el mismo conjunto de evaluación.
- `proyecto_mibici/monitoring/umbral_ab_p1.py` — calcula un umbral de mejora razonable a partir de la variación mensual del WAPE.

Flujo sugerido para los testings del equipo:

1. **Definir periodos**: referencia = ventana de entrenamiento; actual = periodo posterior (ej. octubre 2026).
2. **Ejecutar `EvidentlyMonitor.comparar`** → obtener HTML de drift y tabla de scores.
3. **Identificar variables con drift** → anotar en el reporte PDF cuáles son y qué implican para el modelo.
4. **Entrenar versión alternativa (challenger)** y registrarla en MLflow Model Registry.
5. **Ejecutar A/B testing** (`ab_test_p1.py`) comparando champion vs challenger sobre el conjunto actual.
6. **Decidir**: promover challenger a champion o mantener el vigente, con evidencia de métricas y drift.

---

## 9. Buenas prácticas para el equipo

1. **No hardcodear rutas**: usa `REPORTS_DIR` o deja `out_dir=None` para que se guarde en `reports/drift/`.
2. **Muestreo consistente**: `n_muestra` y `random_seed` controlan el muestreo; úsalos igual para referencia y actual para comparar de forma justa.
3. **Nombres descriptivos**: el parámetro `nombre` se usa en los archivos de salida y en MLflow (ej. `p1_ref2025_vs_oct2026`).
4. **Documentar los periodos elegidos**: en el reporte PDF, indica qué ventana de referencia y qué periodo actual usaste y por qué (requisito del entregable).
5. **No tocar lo existente**: esta clase es aditiva; no edites `drift_report.py`, `ab_test_p1.py` ni `umbral_ab_p1.py` salvo que el equipo acuerde un cambio consensuado.
6. **Evidencia en Git**: cada integrante debe commitear su propio script de testing y los reportes generados (o al menos los scripts que los generan), para que se vea la contribución individual.

---

## 10. Estructura de salida generada

```
reports/drift/
├── drift_<nombre>.html                 # Reporte interactivo de Evidently
├── drift_<nombre>_scores.csv           # Tabla complementaria de scores
└── registro/
    ├── proyectos/
    │   └── <nombre_proyecto>.json      # Metadatos del proyecto
    ├── datasets/
    │   └── <dataset_id>.json           # Metadatos y estadísticas de cada dataset
    └── reportes/
        └── <reporte_id>.json           # Metadatos de cada reporte generado
```

Y en MLflow (experimento `mibici-equipo20-p1-monitoreo`):

```
run: drift_<nombre>
  params: n_muestra, num_method, cat_method, num_threshold, cat_threshold, n_ref, n_actual, nombre
  metrics: vars_con_drift
  artifacts: drift_<nombre>/ (html + csv)
```

---

## 11. Ejecución rápida (CLI opcional)

El módulo incluye un comando de ejemplo para P1 (requiere los datos de modelado generados):

```bash
python -m proyecto_mibici.monitoring.evidently_monitor ejemplo-p1
```

---

## Referencias

- Documentación oficial de Evidently: https://docs.evidentlyai.com/introduction
- Ejemplo oficial de data y ML checks: https://docs.evidentlyai.com/
- Actividad del curso: `acividadMonitoreo.md` (Entregable 8).
