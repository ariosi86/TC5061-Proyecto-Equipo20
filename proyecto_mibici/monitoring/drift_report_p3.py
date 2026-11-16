#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reporte de drift de P3 con Evidently (Entregable 8).

Compara el perfil de las estaciones del periodo actual (octubre 2026) contra:
  1. la referencia de entrenamiento del champion (2025 completo), y
  2. octubre 2025, para separar el efecto de la temporada.

Usa la clase EvidentlyMonitor.comparar() para el cálculo central.
"""

from pathlib import Path

import typer
from loguru import logger

from proyecto_mibici.config import P3_VARIABLES, REPORTS_DIR, URI_MLFLOW_TRACKING
from proyecto_mibici.monitoring import EvidentlyMonitor
from proyecto_mibici.monitoring.datos_p3 import PERIODOS_P3, cargar_conjunto

SALIDA = REPORTS_DIR / "drift" / "p3"
UMBRAL = 0.1

COMPARACIONES = {
    "ref2025_vs_oct2026": "referencia",
    "oct2025_vs_oct2026": "referencia_octubre",
}

app = typer.Typer(help="Reporte de drift de P3 con Evidently.")


@app.command()
def main(
    log_mlflow: bool = typer.Option(True, "--log-mlflow/--sin-mlflow", help="Registrar en MLflow."),
    usar_api_evidently: bool = typer.Option(
        False, "--api-evidently/--sin-api-evidently",
        help="Subir los reportes a Evidently UI (requiere `evidently ui` en el puerto 8000).",
    ),
    salida: str = typer.Option(str(SALIDA), "--salida", help="Carpeta de salida."),
):
    salida_path = Path(salida)
    salida_path.mkdir(parents=True, exist_ok=True)

    actual = cargar_conjunto("actual")

    monitor = EvidentlyMonitor(
        numerical_columns=P3_VARIABLES,
        categorical_columns=[],
        num_threshold=UMBRAL,
        out_dir=salida_path,
        usar_api_evidently=usar_api_evidently,
    )

    for nombre, conjunto_ref in COMPARACIONES.items():
        referencia = cargar_conjunto(conjunto_ref)
        resumen = monitor.comparar(
            referencia,
            actual,
            nombre=f"p3_{nombre}",
            log_mlflow=False,
            registrar_proyecto_nombre="mibici-equipo20-p3" if usar_api_evidently else None,
        )

        logger.info(
            f"--- {nombre}: {conjunto_ref} {PERIODOS_P3[conjunto_ref]} vs "
            f"actual {PERIODOS_P3['actual']}"
        )
        logger.info(f"Variables con drift: {resumen['vars_con_drift']} de {len(P3_VARIABLES)}")
        logger.success(f"Reporte HTML -> {resumen['html']}")

    if log_mlflow:
        import mlflow

        mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
        mlflow.set_experiment("mibici-equipo20-p3-monitoreo")
        with mlflow.start_run(run_name="p3_drift_oct2026"):
            mlflow.set_tags({"question": "P3", "stage_of_work": "monitoring"})
            mlflow.log_params({
                "num_method": monitor.num_method,
                "num_threshold": UMBRAL,
                "variables": ", ".join(P3_VARIABLES),
            })
            mlflow.log_artifacts(str(salida_path), artifact_path="drift_p3")
        logger.success("Corrida p3_drift_oct2026 registrada en MLflow.")


if __name__ == "__main__":
    app()
