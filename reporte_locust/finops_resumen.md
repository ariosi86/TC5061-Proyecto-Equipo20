# Resumen de Desempeño (Locust) y Análisis FinOps — Equipo 20

## 1. Escenario de Prueba de Carga (Locust)

- **Configuración**: `30` usuarios concurrentes, rampa de `5 usuarios/s`, duración `60 s` (`locustfile.py`).
- **Distribución de tráfico**:
  - `MiBiciP1User` (`weight=3`): `POST /predict` (demanda horaria, LightGBM Tweedie) + `GET /health`.
  - `MiBiciP2User` (`weight=2`): `POST /predict` (riesgo de desbalance, HistGradientBoosting) + `GET /health`.
  - `MiBiciP3User` (`weight=1`): `POST /predict` (clustering funcional, KMeans K=4) + `GET /health`.
- **Resultados globales (`mibici_stats.csv`)**:
  - **Total de peticiones**: `1,286` (CSV) / `1,305` (resumen final consola) — **0 fallos (0.00%)**.
  - **Throughput agregado**: `21.74 req/s` (`18.11 pred/s` en `/predict` + `3.63 req/s` en `/health`).
  - **Latencia agregada**: Mediana `6 ms`, Promedio `12.09 ms`, p95 `30 ms`, Máximo `732 ms` (primer hit de caché).

| Endpoint | Peticiones | Fallos | Mediana (p50) | Promedio | p95 | Máximo | Req/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `P1 POST /predict` | 583 | 0 (0.00%) | 5 ms | 6.47 ms | 11 ms | 275 ms | 9.86 |
| `P2 POST /predict` | 374 | 0 (0.00%) | 23 ms | 27.76 ms | 33 ms | 732 ms | 6.32 |
| `P3 POST /predict` | 114 | 0 (0.00%) | 3 ms | 5.20 ms | 9 ms | 127 ms | 1.93 |
| `GET /health` (P1/P2/P3) | 215 | 0 (0.00%) | 2 ms | 2.3–4.6 ms | 5–6 ms | 272 ms | 3.63 |

---

## 2. Consumo de Recursos (`docker stats` durante la carga)

Muestreo durante la ventana activa de carga (`05:02:31Z` – `05:03:08Z`):

| Contenedor | CPU % (rango en carga) | Memoria observada | Rol |
|---|---:|---:|---|
| `mlflow` | `1.39% – 5.92%` | `583.1 MiB` | Tracking & Model Registry |
| `api-p1` | `4.54% – 6.81%` | `229.6 MiB` | Microservicio P1 (LightGBM) |
| `api-p2` | `18.88% – 20.60%` | `189.7 MiB` | Microservicio P2 (GBM) |
| `api-p3` | `0.11% – 1.00%` | `182.8 MiB` | Microservicio P3 (KMeans) |
| **Total Stack** | **~25% – 34% (de 1 vCPU)** | **~1,185 MiB (~1.16 GiB)** | Sistema completo en inferencia |

---

## 3. Análisis FinOps: Costo por Inferencia

### Instancia de referencia en la nube (AWS EC2 `us-east-1`, Linux On-Demand)
Dado que el stack en inferencia consume **~1.16 GiB de RAM** y **~0.34 vCPU pico** bajo 30 usuarios concurrentes:

1. **Escenario ajustado — `t3.small` (2 vCPU, 2 GiB RAM)**: **`$0.0208 USD/hora`** (`$15.18 USD/mes` a 730 h).
2. **Escenario recomendado con margen — `t3.medium` (2 vCPU, 4 GiB RAM)**: **`$0.0416 USD/hora`** (`$30.37 USD/mes` a 730 h).

### Cálculo de costo por inferencia (`/predict`)
Con el throughput medido en `/predict` de **`18.11 inferencias/s`** (`65,196 inferencias/hora`):

| Métrica | `t3.small` (`$0.0208/h`) | `t3.medium` (`$0.0416/h`) |
|---|---:|---:|
| **Costo por hora** | `$0.0208 USD` | `$0.0416 USD` |
| **Inferencias/hora (carga medida 18.11 req/s)** | `65,196` | `65,196` |
| **Costo por inferencia (carga medida)** | `$3.19 × 10⁻⁷ USD` | `$6.38 × 10⁻⁷ USD` |
| **Costo por 1,000,000 de inferencias (carga medida)** | **`$0.32 USD`** | **`$0.64 USD`** |
| **Escenario valle (10% uso = ~6,520 inf/h)** | `$3.19 USD / 1M inf` | `$6.38 USD / 1M inf` |
| **Capacidad máx. estimada (70% de 2 vCPU ≈ 75 inf/s)** | `~$0.077 USD / 1M inf` | `~$0.154 USD / 1M inf` |

---

## 4. Optimizaciones FinOps Propuestas

1. **Caché en memoria con TTL (`MODEL_CACHE_TTL=60s`, ya implementado en `model_loader.py`)**:
   Evitó descargar el `.pkl` desde MLflow en cada petición, manteniendo `mlflow` debajo del `6%` de CPU y latencias p50 de `3–5 ms` en P3 y P1.
2. **Desacoplar `prep` (entrenamiento) del entorno productivo**:
   El entrenamiento inicial (`prep`) requiere ~`7.75 GiB` de RAM por los Parquets históricos, mientras que la operación de las APIs consume solo **`1.16 GiB`**. Ejecutar `prep` como job efímero (instancia Spot o CI) y correr producción en `t3.small`/`t3.medium` reduce el costo fijo mensual en >50% frente a dimensionar la instancia permanente por el pico de entrenamiento.
3. **Optimización de `api-p2`**:
   `api-p2` concentra ~`60–70%` del CPU de inferencia (`20.6%` CPU, `27.8 ms` promedio) frente a `api-p1` (`6.8%` CPU, `6.5 ms`). Reducir el costo computacional de `HistGradientBoostingClassifier` (por ejemplo exportando a ONNX o habilitando inferencia por lotes cuando aplique) aumentaría el techo de RPS por instancia sin subir de `t3.small`/`t3.medium`.
4. **Compromisos de uso (Savings Plans / Reserved Instances 1 año)**:
   Para un servicio continuo 24/7, aplicar Savings Plans reduce el costo de `$0.0416/h` a `~$0.026–$0.029/h` (~30–37% de ahorro directo).
