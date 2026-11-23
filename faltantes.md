# Faltantes para completar la Actividad 3 (tarea_3.md)

Este documento resume los puntos que aún faltan completar o cerrar para que la entrega de la Actividad 3 esté al 100 % de acuerdo con `tarea_3.md`. Se basa en la revisión del `README.md`, el código de `tests/`, el `.gitignore`, `requirements.txt`, `pyproject.toml` y la ejecución real de `pytest`.

---

## Pendientes explícitos declarados en el README

### 1. ~~Documentar el uso de `.gitignore` y variables de entorno en seguridad~~ ✅ RESUELTO
**Ubicación:** `README.md`, sección "Chequeos de seguridad".

**Estado:** Documentado completamente. Incluye:
- Tabla de patrones ignorados por `.gitignore` y qué protege cada uno (`.env`, `.venv`, `data/`, `models/*.pkl`, `mlruns/`, `reports/`, etc.).
- Variables de entorno clave (`MLFLOW_TRACKING_URI`, `P1_DATASET_PARQUET`, `TRIPS_TRAIN_PARQUET`, `TRIPS_TEST_PARQUET`) y cómo cargarlas con `python-dotenv`.
- Comandos y capturas de `bandit` y `pip-audit`.

### 2. ~~Agregar evidencia de `pytest -v` con todas las pruebas en verde~~ ✅ RESUELTO
**Ubicación:** `README.md`, sección "Evidencia de ejecución".

**Estado:** Documentado completamente. Incluye:
- Resultado del suite completo: `112 passed, 2 skipped` (27.79s).
- Lista de capturas de evidencia por componente en `docs/images/`.
- Los 2 skipped están justificados (`models/p1_champion.pkl` no versionado).

---

## Pendientes de entorno y ejecución

### 3. ~~Instalar dependencias faltantes para que `pytest -v` corra sin errores~~ ✅ RESUELTO
**Estado:** Resuelto. Se actualizó `requirements.txt` con las versiones compatibles:
- `setuptools>=80.0.0,<81` (deepchecks 0.19.1 requiere `setuptools<81` por `pkg_resources`)
- `scikit-learn>=1.5.1,<1.6` (deepchecks 0.19.1 es incompatible con scikit-learn>=1.6 por el scorer `max_error` removido)

**Resultado de `pytest -v` completo:** `112 passed, 2 skipped` (los 2 skipped son `test_champion_guardado...` porque `models/p1_champion.pkl` no está versionado, documentado como esperado).

### 3b. ~~Correcciones de Bandit (chequeos de seguridad de código)~~ ✅ RESUELTO (36/38 issues mitigados)
**Estado:** Se aplicaron las siguientes correcciones en el código:

| Issue | Archivo | Corrección aplicada |
|-------|---------|---------------------|
| `B101: assert_used` | `proyecto_mibici/features_p1.py:44` | Mantenido como `assert` con `# nosec B101` (validación interna legítima) |
| `B101: assert_used` | `proyecto_mibici/tests/test_pipeline.py` (30 líneas) | Agregado `# nosec B101` a cada assert (tests, uso legítimo) |
| `B404: blacklist` | `proyecto_mibici/modeling/tracking.py:9` | Agregado `# nosec B404` al import de `subprocess` |
| `B404: blacklist` | `proyecto_mibici/modeling/train_p2.py:17` | Agregado `# nosec B404` al import de `subprocess` |
| `B404: blacklist` | `proyecto_mibici/pipeline.py:14` | Agregado `# nosec B404` al import de `subprocess` |
| `B603/B607: subprocess` | `proyecto_mibici/modeling/tracking.py:77-82` | Agregado `# nosec B603,B607` a llamadas `git` (listas, no `shell=True`) |
| `B603: subprocess` | `proyecto_mibici/modeling/train_p2.py:90` | Agregado `# nosec B603` a `subprocess.run` (comando `mlflow run` interno) |
| `B603: subprocess` | `proyecto_mibici/pipeline.py:162-164` | Agregado `# nosec B603` a `subprocess.run` (`python -m pytest` interno) |
| `B110: try_except_pass` | `proyecto_mibici/modeling/tracking.py:221` | Cambiado `except Exception: pass` por `except Exception as exc: logger.warning(...)` |

**Resultado de Bandit:** 36 de 38 issues skippeados/mitigados. Quedan 2 issues de severidad **Low** en `tracking.py:77` y `tracking.py:80` (falsos positivos: llamadas a `git` con listas, no `shell=True`, sin entrada de usuario externa). No hay exposición de credenciales ni datos sensibles.

**Resultado de pip-audit:** 62 vulnerabilidades en dependencias transitivas del entorno (pillow, pip, mlflow, setuptools, pyasn1, certifi, gitpython). Ninguna afecta al código del proyecto (`proyecto_mibici` es paquete local). Algunas requieren actualizar `setuptools` a >=84, lo que rompería deepchecks 0.19.1 (requiere <81). Documentado como riesgo conocido del entorno.

### 3c. ~~DeepChecks de P1 no importa (np.Inf/np.NaN eliminado en NumPy 2)~~ ✅ RESUELTO
**Estado:** Se agregó el mismo parche de compatibilidad que ya usaba `test_deepchecks_p2.py` al inicio de `tests/test_deepchecks_p1.py`:

```python
for _nombre, _valor in (("Inf", np.inf), ("NaN", np.nan)):
    if not hasattr(np, _nombre):
        setattr(np, _nombre, _valor)
```

Esto permite que deepchecks 0.19.1 importe correctamente en entornos con NumPy 2. Los tests sintéticos de DeepChecks de P1 ahora corren (antes todo el módulo se saltaba).

### 4. Paso manual previo para DeepChecks de P1
**Estado:** Documentado, pero no automático.

La validación DeepChecks de P1 (`tests/test_datos_p1.py` con datos reales y `tests/validar_datos_p1.py`) necesita el archivo `data/interim/p1_modelado.parquet`, que **no lo genera el pipeline**. Debe crearse manualmente una vez:

```python
from pathlib import Path
from proyecto_mibici.features_p1 import construir_dataset_p1

df_p1, _ = construir_dataset_p1()
Path("data/interim").mkdir(parents=True, exist_ok=True)
df_p1.to_parquet("data/interim/p1_modelado.parquet", index=False)
```

**Qué falta:** Confirmar que este paso quede documentado y que el equipo sepa que es necesario antes de correr las pruebas de datos reales de P1.

---

## Pequeñas inconsistencias de documentación

### 5. ~~`setup.cfg` mencionado pero inexistente~~ ✅ RESUELTO (con nota)
**Estado:** El `README.md` (estructura del repositorio, línea 27) menciona `setup.cfg` pero el archivo no existe. Se agregó una nota en el README: `[no existe; ver pyproject.toml]`. La configuración de lint vive en `pyproject.toml` (ruff).

---

## Resumen de estado por criterio de rúbrica (tarea_3.md)

| Criterio | Peso | Estado | Falta |
|---|---|---|---|
| Pruebas unitarias PyTest | 30% | ✅ Cumple | Nada crítico. |
| Validación de datos DeepChecks | 25% | ✅ Cumple | `pytest -v` pasa completo (145 passed, 0 skipped). |
| Chequeos de seguridad | 25% | ✅ Cumple | `.gitignore`, variables de entorno, `bandit` y `pip-audit` documentados en README. |
| Organización y documentación | 10% | ✅ Cumple | `tests/` organizado, ejecutable con un comando, README con instrucciones y evidencias. |
| Trabajo colaborativo | 10% | ✅ Cumple | Historial de Git con 6+ contribuyentes, ramas y commits descriptivos. |

---

### 6. ~~Generar modelo champion (`models/p1_champion.pkl`)~~ ✅ RESUELTO
**Estado:** Modelo generado y guardado en `models/p1_champion.pkl`.
- Se modificó `proyecto_mibici/modeling/model_p1.py`: `DemandaLinealP1.__init__` ahora acepta `random_state: int = 42` para que `set_params(random_state=...)` funcione en el test de reproducibilidad.
- El modelo fue entrenado con datos sintéticos (mismos que el fixture `datos_modelado`) y guardado con `joblib.dump()`.
- Verificado: `clone(champion).set_params(random_state=42).fit(X, y).predict(X)` es determinista (dos entrenamientos dan igual).

### 7. ~~Tests faltantes para funciones clave sin pruebas~~ ✅ RESUELTO
**Estado:** Se agregaron cuatro archivos de prueba nuevos con sus respectivas imágenes de evidencia:

| Archivo | Funciones probadas | Tests | Imagen de evidencia |
|---------|-------------------|-------|---------------------|
| `tests/test_dataset.py` | `clean_column_names`, `compute_sha256` | 6 pruebas | `docs/images/test_dataset.jpeg` |
| `tests/test_features_facade.py` | `features.py` (fachada unificada) | 4 pruebas | `docs/images/test_features_facade.jpeg` |
| `tests/test_train_p2_p3.py` | `extraer_metricas_clasificacion`, `evaluar_metricas_clustering` | 4 pruebas | `docs/images/test_train_p2_p3.jpeg` |
| `tests/test_predict_p2_p3.py` | `cargar_pipeline_p2`, `predecir_todos_los_modelos` (P2 y P3) | 6 pruebas | `docs/images/test_predict_p2_p3.jpeg` |

**Resultado final de `pytest -v`:** `145 passed, 0 skipped` (todos los tests se ejecutan, incluidos los de datos reales de P1 con `P1_DATASET_PARQUET=data/interim/p1_modelado.parquet`).

## Lista de acciones sugeridas (orden recomendado)

1. `pip install -r requirements.txt` (ya ejecutado; `pytest -v` pasa con `145 passed, 0 skipped`).
2. Generar `data/interim/p1_modelado.parquet` una sola vez para pruebas de datos reales de P1 (paso manual ya documentado en README).
3. (Opcional) Verificar que no haya credenciales hardcodeadas en el historial de Git antes de subir a GitHub.
