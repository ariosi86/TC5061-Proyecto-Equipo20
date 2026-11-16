#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monitoreo de drift de P2 con Evidently (Entregable 8).

Referencia: entrenamiento del champion (2025).
Actual: julio a septiembre de 2026 (ver datos_p2.py).

Usa la clase EvidentlyMonitor.comparar() para el cálculo central.
No modifica modelos ni el Registry.
"""

import typer
from loguru import logger

from proyecto_mibici.config import (
    P2_OBJETIVO,
    P2_VARIABLES,
    REPORTS_DIR,
)
from proyecto_mibici.monitoring import EvidentlyMonitor
from proyecto_mibici.monitoring.datos_p2 import (
    construir_serie as construir_serie_p2,
    separar,
    INICIO_ACTUAL as P2_INICIO_ACTUAL,
    FIN_ACTUAL as P2_FIN_ACTUAL,
)

UMBRAL_NUM = 0.1
UMBRAL_CAT = 0.1
N_MUESTRA = 100_000

# Columnas categóricas según la definición original de drift_report_p2.py
CAT = [
    "hora_del_dia",
    "dia_semana",
    "fin_de_semana",
    "riesgo_vacia",
    "riesgo_llena",
    P2_OBJETIVO,
]

app = typer.Typer(help="Reportes de drift de P2 con Evidently.")


@app.command()
def main(
    n_muestra: int = typer.Option(N_MUESTRA, "--n-muestra", help="Filas por conjunto."),
    registrar: bool = typer.Option(True, "--mlflow/--sin-mlflow", help="Registrar en MLflow."),
):
    # Construir la serie completa hasta septiembre 2026
    serie = construir_serie_p2(fin_periodo=P2_FIN_ACTUAL)

    # Separar en referencia (entrenamiento 2025) y actual (jul-sep 2026)
    ref, cur, nuevas = separar(serie, P2_INICIO_ACTUAL, P2_FIN_ACTUAL)

    logger.info(
        f"Referencia: {len(ref):,} filas | Actual: {len(cur):,} filas | "
        f"Estaciones nuevas excluidas: {nuevas}"
    )

    numericas = [c for c in P2_VARIABLES if c not in CAT]

    monitor = EvidentlyMonitor(
        numerical_columns=numericas,
        categorical_columns=CAT,
        num_threshold=UMBRAL_NUM,
        cat_threshold=UMBRAL_CAT,
        n_muestra=n_muestra,
        nombre_proyecto="p2-riesgo-desequilibrio",
        descripcion_proyecto="Monitoreo de drift P2: referencia 2025 vs julio-septiembre 2026",
    )

    resumen = monitor.comparar(
        ref,
        cur,
        nombre="p2_ref2025_vs_jul_sep2026",
        registrar_proyecto_nombre="p2-riesgo-desequilibrio",
        log_mlflow=registrar,
        mlflow_run_name="P2_drift_jul_sep2026",
        experimento="mibici-equipo20-p2-monitoreo",
    )

    logger.info(f"Variables con drift: {resumen['vars_con_drift']}")
    logger.success(f"Reporte HTML -> {resumen['html']}")


if __name__ == "__main__":
    app()
