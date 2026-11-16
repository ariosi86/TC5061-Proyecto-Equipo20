#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monitoreo de drift de P1 con Evidently (Entregable 8).

Referencia vs octubre 2026. No modifica modelos ni el Registry.
Usa la clase EvidentlyMonitor.comparar() para el cálculo central.
"""

import pandas as pd
import typer
from loguru import logger

from proyecto_mibici.config import (
    P1_ACTUAL_FIN,
    P1_FEATURES_V2,
    P1_TARGET,
    P1_TEST_FIN,
    P1_VAL_INICIO,
)
from proyecto_mibici.features import construir_dataset_p1
from proyecto_mibici.monitoring import EvidentlyMonitor

app = typer.Typer(help="Reportes de drift de P1 con Evidently.")

NUMERICAS = [c for c in P1_FEATURES_V2 if c != "target_is_weekend"]
CATEGORICAS = ["target_is_weekend"]


@app.command()
def main(n_muestra: int = typer.Option(100_000, "--n-muestra", help="Filas por conjunto.")):
    df_p1, _ = construir_dataset_p1()
    df_p1 = df_p1.dropna(subset=P1_FEATURES_V2 + [P1_TARGET])
    t = df_p1["target_hour"]

    actual = df_p1[(t >= pd.Timestamp(P1_TEST_FIN)) & (t < pd.Timestamp(P1_ACTUAL_FIN))]
    comparaciones = {
        "ref2025_vs_oct2026": df_p1[t < pd.Timestamp(P1_VAL_INICIO)],
        "oct2025_vs_oct2026": df_p1[
            (t >= pd.Timestamp("2025-10-01")) & (t < pd.Timestamp("2025-11-01"))
        ],
    }
    if actual.empty:
        raise ValueError("No hay filas de octubre 2026 en la tabla de modelado.")

    out_dir = None  # Se usa reports/drift/ por defecto desde EvidentlyMonitor

    for nombre, ref in comparaciones.items():
        logger.info(
            f"=== {nombre}: ref {len(ref):,} filas | actual {len(actual):,} filas ==="
        )

        monitor = EvidentlyMonitor(
            numerical_columns=NUMERICAS,
            categorical_columns=CATEGORICAS,
            n_muestra=n_muestra,
            nombre_proyecto="p1-demanda-horaria",
            descripcion_proyecto="Monitoreo de drift P1: referencia 2025 vs octubre 2026",
        )
        resumen = monitor.comparar(
            ref,
            actual,
            nombre=f"p1_{nombre}",
            registrar_proyecto_nombre="p1-demanda-horaria",
        )

        logger.info(f"Variables con drift: {resumen['vars_con_drift']}")
        logger.success(f"Reporte HTML -> {resumen['html']}")


if __name__ == "__main__":
    app()
