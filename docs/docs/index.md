# TC5061 - Proyecto MiBici (Documentación Técnica)

Bienvenido a la documentación oficial del proyecto del **Equipo 20** para el sistema de bicicleta pública compartida **MiBici** (Área Metropolitana de Guadalajara).

## Descripción del Proyecto

Diseñar, como equipo, una propuesta de solución de Machine Learning para un conjunto de preguntas operativas sobre el sistema de bicicleta pública MiBici, que combina modelos de regresión, clasificación y clustering bajo estándares estrictos de MLOps, Pipelines de scikit-learn y programación orientada a objetos (POO).

---

## Secciones Principales

* **[Getting Started](getting-started.md)**: Guía detallada de clonación, configuración de entorno virtual, descarga de datos e instalación del paquete local.
* **Modelos y Pipelines**: Documentación de la arquitectura de modelado (P1: Demanda Horaria, P2: Riesgo de Desbalance, P3: Clustering de Estaciones).
* **Tracking y MLOps**: Integración con MLflow, control de versiones criptográfico (`manifest.json`) y artefactos exportados.

---

## Comandos Rápidos del Proyecto

El proyecto incluye un orquestador End-to-End (`pipeline.py`) y un `Makefile` para automatizar las tareas comunes:

* Ejecutar el pipeline completo (Datos + Entrenamiento + Inferencia + Pytest):
  ```bash
  python -m proyecto_mibici.pipeline
  ```
* Iniciar el servidor local de MLflow:
  ```bash
  mlflow server --port 5000
  ```

