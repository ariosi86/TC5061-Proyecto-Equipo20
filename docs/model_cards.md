# Model Cards: MiBici, Equipo 20

Fichas de los modelos de P1, P2 y P3. Estructura basada en *Model Cards for Model Reporting* (Mitchell et al., 2019), verificadas contra el Model Registry (`@champion` en MLflow) y los reportes de monitoreo.

**Aplica a las tres**
- **Responsables:** Equipo 20, TC5061.
- **Fuente:** datos abiertos de viajes y catálogo de estaciones de MiBici (Área Metropolitana de Guadalajara).
- **Versión de datos vigente (`datos_version`):** `23510f42a0a0` (`data/manifest.json`, compartida por los tres `@champion` activos).
- **Estado:** proyecto académico. No se ha validado en una operación real.
- **Reversión:** reasignar el alias `champion` a la versión anterior. Con caché, la API toma el cambio al vencer `CACHE_TTL_SEG` (60 s por defecto). Revisar que `prep` con `FORZAR_ENTRENAMIENTO=1` no reemplace la elección.

---

## P1: Demanda horaria por estación

### Detalles
| Campo | Valor |
|---|---|
| Registry | `mibici-equipo20-p1-demanda@champion` |
| Versión | `1` (run `03ba9410107d4e6487a2619d6602ce6d`) |
| Algoritmo | LightGBM, objetivo Tweedie (power 1.8, learning_rate 0.1, num_leaves 127, n_estimators 600) |
| Linaje | Run `P1_lgbm_tweedie`, repositorio `15c7add219` |

### Uso previsto
Estimar los viajes que iniciarán en una estación en la hora siguiente, como apoyo al operador para planear la flota.

### Usos no recomendados
- Pronósticos a varios días u horizontes distintos de una hora.
- Estaciones sin historial suficiente para calcular los rezagos, u otras ciudades.
- Decisiones automáticas de rebalanceo sin supervisión.

### Datos
- **Entrenamiento:** 2025 (`2025-01-08` a `2025-12-31`). **Validación:** enero a abril de 2026 (`2026-01-01` a `2026-04-30`). **Prueba:** mayo a septiembre de 2026 (`2026-05-01` a `2026-09-30`).
- **Variables:** `lag_1h`, `lag_24h`, `lag_168h`, `rolling_3h`, `rolling_168h`, `target_hour_sin`, `target_hour_cos`, `target_is_weekend`.

### Métricas
| Modelo | MAE validación | MAE prueba | RMSE prueba | WAPE prueba | Sesgo prueba |
|---|---|---|---|---|---|
| Baseline lineal | 0.885 | 0.853 | 1.545 | 0.681 | +0.004 |
| Champion Tweedie | 0.816 | 0.789 | 1.409 | 0.630 | -0.027 |

Frente al baseline, en prueba el MAE baja 7.5 % y el RMSE 8.8 %.

### Limitaciones y riesgos
- Tiende a subestimar (cerca de 300 viajes diarios en toda la red), lo que puede dejar estaciones sin bicicletas.
- El WAPE de 0.63 indica un error grande respecto al volumen de demanda por estación y hora.
- La demanda tiene muchos ceros: 55.96 % de las horas de prueba en la validación con DeepChecks.
- La API depende de que el cliente envíe rezagos y medias móviles correctos.

### Consideraciones éticas
Los errores pueden no repartirse igual entre estaciones o zonas. Antes de usarlo para asignar bicicletas, revisar las métricas por estación y zona para no atender sistemáticamente peor a algunas (análisis desagregado por zona pendiente antes de operación real).

### Monitoreo y gobernanza
- **Drift:** referencia 2025 y octubre 2025 contra octubre 2026 (`drift_report.py`); 6 de 9 variables (objetivo, rezagos y medias móviles).
- **A/B en octubre 2026** (champion v1 contra challenger reentrenado en el experimento `mibici-equipo20-p1-monitoreo`):

| Métrica | Champion | Challenger |
|---|---|---|
| MAE | 0.9825 | 0.9804 |
| RMSE | 1.7559 | 1.7277 |
| Deviance Tweedie | 11.30 | 14.98 |

- **Decisión:** mantener el champion. La mejora en MAE (0.21 %) no alcanza el umbral de 7.8 %, la deviance empeora y el challenger gana en solo 29.9 % de las estaciones.
- **Nota sobre el umbral:** se calibró con la variación del champion de mayo a octubre de 2026, que incluye el mes del A/B. En una siguiente iteración conviene calibrarlo solo con meses anteriores.
- **Cuándo revisar o reentrenar (propuesta):** cada mes, o antes si aparece drift junto con un deterioro del error o del sesgo. Un reentrenamiento solo se promueve si cumple los criterios del A/B.

---

## P2: Riesgo de desbalance de estaciones

### Detalles
| Campo | Valor |
|---|---|
| Registry | `mibici-equipo20-p2-riesgo@champion` |
| Versión | `1` (run `025f42b7df7c4b5a91146ae3d2697f6c`, vigente en el despliegue Docker) |
| Algoritmo | `HistGradientBoostingClassifier` (learning_rate 0.1, max_iter 100, max_leaf_nodes 31, min_samples_leaf 20), umbral 0.5 |
| Linaje del modelo original | Run `p2-gbm`, repositorio `15c7add219` |

### Uso previsto
Alertar al equipo de rebalanceo si una estación estará en riesgo de quedar vacía o llena en la hora siguiente.

### Usos no recomendados
- Interpretar la alerta como el inventario real de bicicletas o anclajes.
- Estaciones sin capacidad registrada.
- Despachar rebalanceos sin validar el estado de la estación.

### Datos
- **Original:** entrenamiento 2025, validación enero a marzo de 2026, prueba abril a junio de 2026, sin estaciones nuevas.
- **Challenger:** entrenamiento enero 2025 a junio 2026; evaluación julio a septiembre de 2026, sin estaciones que iniciaron desde abril de 2026.
- **Variables (11):** `hora_del_dia`, `dia_semana`, `fin_de_semana`, `capacidad`, `salidas`, `llegadas`, `balance_acumulado_dia`, `acumulado_relativo`, `balance_ult_3h_relativo`, `riesgo_vacia`, `riesgo_llena`.

### Métricas (prueba, abril a junio de 2026)
| Modelo | Precisión | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Persistencia | 0.833 | 0.833 | 0.833 | 0.712 |
| GBM original | 0.896 | 0.837 | 0.865 | 0.942 |

### Limitaciones y riesgos
- El riesgo se deriva del balance acumulado desde las 3:00 respecto a la capacidad; es una aproximación, no una medición del inventario.
- Un falso negativo puede dejar usuarios sin bicicleta o sin anclaje; un falso positivo genera un traslado innecesario.
- La proporción de horas en riesgo cambia en el tiempo (13.0 % a 9.9 % entre referencia y julio a septiembre de 2026).

### Consideraciones éticas
Revisar precisión y recall por estación y zona (análisis pendiente antes de operación real), para evitar que algunas reciban menos alertas útiles.

### Monitoreo y gobernanza
- **Drift:** sin drift en 12 columnas (11 variables y el objetivo); puntaje máximo 0.087. La proporción de riesgo de estación vacía bajó de 6.8 % a 4.3 %, así que conviene vigilar también el cambio relativo.
- **A/B, julio a septiembre de 2026:** F1 0.8477 contra 0.8468 y PR-AUC 0.9284 contra 0.9279 (challenger contra champion). El IC 95 % de la diferencia en F1 es +0.0004 a +0.0013; McNemar p = 0.00024. Según el reporte, ambos intervalos quedan sobre cero.
- **Decisión:** la regla recomienda promover al challenger. La ganancia es pequeña, cerca de 83 riesgos más detectados en tres meses. Estado del alias: en el despliegue Docker actual `@champion` apunta a la versión `1` (`p2-gbm`); al ejecutar `ab_test_p2.py --promover` se actualiza el alias al challenger.
- **Cuándo revisar o reentrenar (propuesta):** cada trimestre, o antes si bajan F1 o PR-AUC o cambia mucho la proporción de riesgo.

---

## P3: Grupos funcionales de estaciones

### Detalles
| Campo | Valor |
|---|---|
| Registry | `mibici-equipo20-p3-grupos@champion` |
| Versión | `1` (run `66dc1ab7c9ac4a9f806e53564413ef4a`) |
| Algoritmo | `StandardScaler` + `KMeans` (K=4, n_init 10, semilla 42) |
| Linaje | Run `p3-kmeans-k4`, repositorio `15c7add219` |

### Uso previsto
Exploración y planeación urbana: comparar patrones de uso entre estaciones y asignar grupo a una estación nueva o actualizada.

### Usos no recomendados
- Tratar los grupos como categorías reales o permanentes.
- Decidir por sí solo dónde abrir, cerrar o reducir estaciones.
- Describir a las personas usuarias o a la población de una zona.

### Datos
- **Entrenamiento:** 2025, estaciones con al menos 100 viajes. **Validación:** enero a abril de 2026. **Prueba:** mayo a septiembre de 2026.
- **Variables (8):** `pct_manana`, `pct_mediodia`, `pct_tarde`, `pct_noche`, `pct_fin_de_semana`, `duracion_media`, `viajes_dia_log`, `balance_neto`.

### Métricas (prueba)
| Modelo | Silhouette (mayor es mejor) | Davies-Bouldin (menor es mejor) |
|---|---|---|
| Baseline por municipio | 0.019 | 3.712 |
| KMeans K=4 | 0.182 | 1.535 |

### Limitaciones y riesgos
- Un silhouette de 0.182 indica grupos con separación débil; supera al baseline, pero hay traslape.
- No hay etiquetas reales para validar los grupos.
- Una estación con pocos viajes puede tener un perfil inestable.

### Consideraciones éticas
Usar los grupos junto con el contexto de cada zona. Una demanda baja puede reflejar poca oferta o mal acceso, no falta de necesidad.

### Monitoreo y gobernanza
- **Drift en octubre 2026:** 6 de 8 variables contra 2025 y 7 de 8 contra octubre 2025. Destacan `duracion_media`, `balance_neto` y `viajes_dia_log`.
- **Efecto:** el silhouette bajó a 0.163 y 33 % de las estaciones cambiaron de grupo respecto a 2025. El grupo de alta demanda pasó de 134 a 251 estaciones.
- **A/B:** challenger 0.166 contra champion 0.163 en silhouette, IC 95 % de la diferencia de -0.006 a +0.010; Davies-Bouldin mejora 0.04.
- **Decisión:** mantener el champion, porque el intervalo del silhouette incluye cero. Mantenerlo no elimina el drift: hay que seguir vigilándolo e interpretar los grupos con cautela.
- **Cuándo revisar o reentrenar (propuesta):** cada trimestre, o antes si baja el silhouette o muchas estaciones cambian de grupo.
