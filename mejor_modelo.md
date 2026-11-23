# Selección del Mejor Modelo por Pregunta — MiBici Equipo 20

Este documento resume la comparación de modelos ejecutada con `run_train.py` (todas las preguntas P1, P2, P3), registrada en la interfaz de MLflow (`http://127.0.0.1:5000`, experimento `mibici-equipo20`), y justifica el modelo que pasará a producción por cada pregunta de negocio.

## 1. Preguntas del proyecto integrador

| # | Pregunta de negocio | Tipo de tarea | Quién la usaría |
|---|---------------------|---------------|-----------------|
| P1 | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | Operador del sistema, para planear la flota. |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro? | Clasificación | Equipo de rebalanceo de bicicletas. |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad? | Clustering | Planeación urbana y expansión del sistema. |

## 2. Modelo seleccionado por pregunta (para producción)

| Pregunta | Modelo elegido | Nombre en Model Registry | Métrica de decisión (Test) | Baseline (Test) | Mejora vs. baseline |
|----------|----------------|--------------------------|----------------------------|-----------------|---------------------|
| P1 | LightGBM Tweedie (champion; mejor modelo de la búsqueda anidada). Challenger por comparación: LightGBM Poisson (no registrado) | `mibici-equipo20-p1-demanda` | Val MAE **0.816** (criterio) · Test MAE 0.789 (reportado) | 0.853 (lineal + rolling) | −7.5 % MAE / −8.8 % RMSE |
| P2 | Gradient boosting `HistGradientBoostingClassifier` con valores por defecto, umbral 0.5 (`gbm`) | `mibici-equipo20-p2-riesgo` | Test F1 **0.865** / PR-AUC **0.942** | F1 0.833 / PR-AUC 0.712 (persistencia) | +3.2 pts F1 / +23.0 pts PR-AUC |
| P3 | KMeans funcional con K=4 | `mibici-equipo20-p3-grupos` | Test Silhouette **0.182** / DB **1.535** | Silhouette 0.019 / DB 3.71 (municipio) | +0.163 Silhouette / −2.18 DB |

---

## 3. Tablas comparativas por pregunta

### 3.1 P1 — Demanda horaria (Regresión)

Se selecciona por menor **MAE de validación**; la prueba se reporta una sola vez.
WAPE, RMSE y sesgo son complementarias. Sesgo = media(pred − real).
Métrica de negocio: subestimar la demanda deja estaciones sin bicicletas; sobreestimarla genera rebalanceos innecesarios.

| Modelo | Hiperparámetros clave | Val MAE | Val WAPE | Test MAE | Test RMSE | Test WAPE | Test sesgo | Mejora MAE test |
|---|---|---|---|---|---|---|---|---|
| Baseline (lineal + rolling, v2) | StandardScaler + LinearRegression, features V2 | 0.885 | 0.663 | 0.853 | 1.545 | 0.681 | +0.004 | referencia |
| Regresión de Poisson | PoissonRegressor(alpha=1e-4, max_iter=300) | 1.165 | 0.872 | 1.181 | 4.341 | 0.943 | +0.137 | −38.5 % |
| LightGBM Poisson (challenger) | objective=poisson, lr=0.1, num_leaves=127, n_estimators=600 | 0.819 | 0.614 | 0.792 | 1.422 | 0.632 | −0.018 | +7.2 % |
| LightGBM Tweedie (champion) | objective=tweedie, power=1.8, lr=0.1, num_leaves=127, n_estimators=600 | 0.816 | 0.611 | 0.789 | 1.409 | 0.630 | −0.027 | +7.5 % |

### 3.2 P2 — Riesgo de desbalance (Clasificación)

Métrica de negocio: el equipo de rebalanceo actúa sobre la predicción de riesgo. Un **falso negativo** (no predecir riesgo) deja usuarios sin bici o sin anclaje (impacto directo al servicio); un **falso positivo** genera un traslado innecesario (costo operativo). Se usa **F1** como resumen del equilibrio entre **Precisión** y **Recall** con el umbral de decisión de 0.5, y **PR-AUC** para medir qué tan bien ordena el modelo los casos de riesgo, independientemente del umbral. La exactitud no se usa porque solo cerca del 10 % de las horas-estación están en riesgo.

PR-AUC se calcula como *average precision* (`average_precision_score`), igual que en la entrega anterior. Una versión previa de este documento usaba interpolación lineal de la curva, que inflaba el PR-AUC de la persistencia (0.842 en lugar de 0.712) porque ese baseline solo produce dos valores (0 o 1).

| Modelo | Hiperparámetros clave | Val F1 | Test Precisión | Test Recall | Test F1 | Test PR-AUC | Mejora F1 vs baseline | Mejora PR-AUC vs baseline |
|--------|----------------------|--------|----------------|-------------|---------|-------------|-----------------------|---------------------------|
| **Baseline (persistencia)** | Regla `en_riesgo_t` → `en_riesgo_t+1` | 0.824 | 0.833 | 0.833 | 0.833 | 0.712 | — (referencia) | — (referencia) |
| Logística balanceada | `class_weight=balanced`, `C=1.0`, `max_iter=1000`, umbral 0.5 | 0.781 | 0.745 | 0.850 | 0.794 | 0.799 | −3.9 pts | +8.7 pts |
| Logística sin peso | `class_weight=None`, `C=1.0`, `max_iter=1000`, umbral 0.5 | 0.823 | 0.831 | 0.832 | 0.832 | 0.806 | −0.1 pts | +9.4 pts |
| Logística umbral 0.3 | `class_weight=balanced`, `C=1.0`, `max_iter=1000`, umbral 0.3 | 0.694 | 0.589 | 0.881 | 0.706 | 0.799 | −12.7 pts | +8.7 pts |
| Logística afinada (`logistic_search_tuned`) | Búsqueda anidada de 16 combinaciones (`C`, `max_iter`, `class_weight`; criterio F1 de validación) → `C=10`, `max_iter=500`, `class_weight=None` | 0.823 | 0.831 | 0.832 | 0.832 | 0.806 | −0.1 pts | +9.4 pts |
| **Gradient boosting (`gbm`, ELEGIDO)** | `HistGradientBoostingClassifier` con valores por defecto: `learning_rate=0.1`, `max_iter=100`, `max_leaf_nodes=31`, `min_samples_leaf=20` | **0.856** | **0.896** | 0.837 | **0.865** | **0.942** | **+3.2 pts** | **+23.0 pts** |
| Gradient boosting afinado (`gbm_tuned`) | Búsqueda anidada de 8 combinaciones (`learning_rate`, `max_leaf_nodes`, `min_samples_leaf`; criterio PR-AUC de validación) → `learning_rate=0.1`, `max_leaf_nodes=63`, `min_samples_leaf=100`, `max_iter=500` (se detuvo en 373) | 0.856 | 0.899 | 0.833 | 0.865 | 0.943 | +3.2 pts | +23.1 pts |

Todas las versiones usan la misma partición temporal que la entrega anterior (entrenamiento: 2025; validación: enero a marzo de 2026; prueba: abril a junio de 2026, sin estaciones nuevas) y las mismas 12 variables.

### 3.3 P3 — Clustering de estaciones (No supervisado)

Métrica de negocio: identificar grupos funcionales de estaciones para planeación urbana y expansión. Se valida la calidad de la partición con **Silhouette Score** (mayor = mejor separación) y **Davies-Bouldin** (menor = mejor). El baseline es la agrupación administrativa por municipio.

| Modelo | Hiperparámetros clave | Train Silhouette | Train Davies-Bouldin | Test Silhouette | Test Davies-Bouldin | Mejora Silhouette vs baseline |
|--------|----------------------|------------------|----------------------|-----------------|---------------------|-------------------------------|
| Baseline (municipio) | Agrupación por `municipality_code` | 0.050 | 3.348 | 0.019 | 3.712 | — (referencia) |
| KMeans K=3 | `n_clusters=3`, `n_init=10`, StandardScaler | — | — | 0.167 | 1.817 | +0.148 |
| KMeans K=4 (ELEGIDO) | `n_clusters=4`, `n_init=10`, StandardScaler | — | — | **0.182** | **1.535** | **+0.163** |
| KMeans K=5 | `n_clusters=5`, `n_init=10`, StandardScaler | — | — | 0.149 | 1.801 | +0.130 |
| KMeans K=6 | `n_clusters=6`, `n_init=10`, StandardScaler | — | — | 0.141 | 1.780 | +0.122 |

---

## 4. Análisis de la comparación y justificación del modelo elegido

### P1 → LightGBM Tweedie (champion)
- **Regla:** menor MAE de validación (0.816 vs 0.819 de Poisson). Empate práctico; Poisson queda como challenger por menor sesgo (−1.4 % vs −2.1 %) y un hiperparámetro menos.
- **Desempeño:** MAE de prueba −7.5 % y RMSE −8.8 % frente al baseline.
- **Sesgo:** ambos subestiman levemente y de forma estable entre validación y prueba; Tweedie subestima más que Poisson (≈300 vs ≈200 viajes diarios en la red). El Poisson lineal sobreestima +11 %.
- **Estabilidad:** mejora en validación (7.7 %) y prueba (7.5 %) casi idéntica; el MAE es estable entre entrenamiento, validación y prueba (0.820, 0.816, 0.789).
- **Costo:** 17.7s por corrida completa, comparable al baseline (14.3s).

### P2 — Riesgo de desbalance → **Gradient boosting (`gbm`)**
- **Desempeño vs. baseline:** La persistencia es un baseline exigente (F1 0.833 en prueba), porque el estado de riesgo cambia poco de una hora a la siguiente. Ninguna de las cuatro versiones logísticas lo supera en F1 (la mejor llega a 0.832), aunque sí ordenan mejor los casos de riesgo (PR-AUC 0.806 contra 0.712). La búsqueda anidada sobre la logística (16 combinaciones) no cambia el resultado: su límite está en ser un modelo lineal, no en sus hiperparámetros. El gradient boosting sí supera al baseline: F1 0.865, PR-AUC 0.942 y precisión 0.896, con un recall similar (0.837 contra 0.833).
- **Métrica ligada a la decisión:** Para el equipo de rebalanceo, el gbm detecta la misma proporción de estaciones en riesgo que la persistencia, pero con menos falsas alarmas (la precisión sube de 0.833 a 0.896), lo que significa menos traslados innecesarios. Bajar el umbral de la logística a 0.3 aumenta el recall a 0.881, pero la precisión cae a 0.589: alrededor de 4 de cada 10 alertas serían falsas, lo que no es operativamente viable.
- **Estabilidad:** El F1 del gbm es prácticamente igual en entrenamiento (0.856), validación (0.856) y prueba (0.865), sin indicios de sobreajuste. Con semilla fija, al volver a correrlo se obtienen exactamente los mismos resultados.
- **Costo:** El gbm entrena en unos 27 segundos sobre 3.2 millones de horas-estación (100 árboles de 31 hojas). La versión afinada por búsqueda rinde igual (F1 0.865, PR-AUC 0.943), pero necesita 373 árboles de 63 hojas y unos 64 segundos, y es más pesada al predecir. Ante un desempeño equivalente, se elige el modelo más simple. La búsqueda documenta además que los valores por defecto ya estaban prácticamente en el óptimo, lo que resuelve la limitación señalada en la entrega anterior ("los hiperparámetros del gradient boosting no se optimizaron").

### P3 — Clustering → **KMeans funcional con K=4**
- **Desempeño vs. baseline:** K=4 alcanza Test Silhouette 0.182 y Davies-Bouldin 1.535, muy superior al baseline municipal (Silhouette 0.019, DB 3.712), que refleja que la división administrativa no captura los patrones funcionales de uso.
- **Métrica ligada a la decisión:** Planeación urbana necesita grupos homogéneos en comportamiento (mañana/tarde/noche, fin de semana, duración, balance neto). K=4 es el que mejor separa estos perfiles (mayor Silhouette, menor DB) entre K=3,4,5,6.
- **Estabilidad:** Con `n_init=10` y semilla fija (42) el clustering es reproducible; K=4 es el valor configurado en `config.py` (`P3_K_ELEGIDO`).
- **Costo:** KMeans sobre ~371 estaciones de entrenamiento es computacionalmente barato; la asignación de nuevas estaciones es una predicción de proximidad a centroides, de bajo costo en producción.

---

## 5. Esquema de ciclo de vida utilizado y justificación

**Esquema:** Desarrollo y seguimiento con **MLflow Tracking** (experimento único `mibici-equipo20`) + registro y promoción en **MLflow Model Registry**.

Etapas del ciclo:
1. **Ingesta y versionado de datos:** `dataset.py` genera los Parquet procesados y `data/manifest.json` con hash SHA-256; cada run registra `datos_version` (hash corto de los datos de entrenamiento).
2. **Experimentos (Tracking):** Una corrida por modelo/versión, registrando parámetros, métricas (Train/Validación/Test), etiquetas (`question`, `model_family`, `stage_of_work`, `equipo=20`, `git_commit`, `datos_version`), artefactos (modelo con firma, figuras, manifiesto).
3. **Selección:** Comparación en la interfaz de MLflow por métrica de negocio, estabilidad Train/Test y costo.
4. **Registro (Model Registry):** Cada modelo elegido se registra con nombre consistente (`mibici-equipo20-p1-demanda`, `mibici-equipo20-p2-riesgo`, `mibici-equipo20-p3-grupos`), conservando el linaje (experimento, corrida de origen, commit de Git, versión de datos) y una descripción.
5. **Promoción:** Las versiones registradas se promueven al alias/stage `Production` para inferencia (`predict_p*.py`).

**Justificación:** MLflow Tracking da trazabilidad completa de cada corrida (qué datos, qué código/commit, qué hiperparámetros, qué métricas). El Model Registry separa la experimentación de la producción, permite versionar modelos, hacer rollback y asociar cada versión a su run de origen y a su versión de datos, cumpliendo requisitos de reproducibilidad y auditoría MLOps.

---

## 6. Modelo a producción (resumen)

| Pregunta | Modelo en producción | Nombre en Registry | Run de origen | Commit | Versión de datos |
|----------|----------------------|--------------------|---------------|--------|------------------|
| P1 | LightGBM Tweedie | `mibici-equipo20-p1-demanda` | `P1_lgbm_tweedie` | `4dc84407e4` | `35a3ecf437d8` |
| P2 | Gradient boosting (`HistGradientBoostingClassifier`, umbral 0.5) | `mibici-equipo20-p2-riesgo` | `p2-gbm` | `4f8dbb1542` | `35a3ecf437d8` |
| P3 | KMeans K=4 | `mibici-equipo20-p3-grupos` | `p3-kmeans-k4` (n_clusters=4) | `c87cc1bd81` / `4dc84407e4` | `35a3ecf437d8` |

El registro de estos tres modelos en el Model Registry se ejecuta con `run_mejor_modelo.py`.
