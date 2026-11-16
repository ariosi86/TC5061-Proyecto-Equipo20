#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Umbral del A/B de P1 a partir de la variación mensual de v1.

Calcula el WAPE de la versión vigente (v1) mes a mes, de mayo a octubre de 2026
(meses que v1 no usó para entrenar), y toma la variación relativa entre meses
(desviación estándar / media) como umbral de mejora: una mejora menor que lo que
el error de v1 varía normalmente de un mes a otro no se distingue de ese vaivén.

Como el A/B se evalúa sobre el mismo conjunto, el cambio relativo del WAPE es
igual al del MAE, así que el umbral se usa directo en ab_test_p1:

    python -m proyecto_mibici.monitoring.ab_test_p1 --min-mejora-mae <UMBRAL>

Uso:
    python -m proyecto_mibici.monitoring.umbral_ab_p1
"""

import typer
from loguru import logger

from proyecto_mibici.monitoring import EvidentlyMonitor

app = typer.Typer(help="Umbral del A/B a partir de la variación mensual del WAPE de v1.")


@app.command()
def main(
    champion_version: int | None = typer.Option(
        None, "--champion-version", help="Por defecto, la versión del alias 'champion'."),
):
    monitor = EvidentlyMonitor(
        numerical_columns=[],
        categorical_columns=[],
        usar_api_evidently=False,
    )
    resumen = monitor.calcular_umbral_ab(champion_version=champion_version)
    logger.success(
        f"Umbral propuesto: {resumen['umbral_propuesto']}% "
        f"(WAPE medio: {resumen['wape_medio']:.4f})"
    )


if __name__ == "__main__":
    app()
