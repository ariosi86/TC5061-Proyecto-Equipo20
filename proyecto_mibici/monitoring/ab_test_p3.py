#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A/B testing de P3 (Entregable 8): champion contra challenger en octubre 2026.

Usa la clase EvidentlyMonitor.ab_test_p3() para el cálculo central.
"""

import typer
from loguru import logger

from proyecto_mibici.monitoring import EvidentlyMonitor

app = typer.Typer(help="A/B testing de P3: champion vs challenger.")


@app.command()
def main(
    champion_version: int | None = typer.Option(
        None, "--champion-version", help="Por defecto, la versión del alias 'champion'."),
    challenger_version: int | None = typer.Option(
        None, "--challenger-version", help="Por defecto, la versión del alias 'challenger'."),
    n_boot: int = typer.Option(1000, "--n-boot", help="Repeticiones del bootstrap por estación."),
    promover: bool = typer.Option(
        False, "--promover", help="Mueve el alias 'champion' al challenger si cumple."),
):
    monitor = EvidentlyMonitor(
        numerical_columns=[],
        categorical_columns=[],
        usar_api_evidently=False,
    )
    resumen = monitor.ab_test_p3(
        champion_version=champion_version,
        challenger_version=challenger_version,
        n_boot=n_boot,
        promover=promover,
    )

    logger.info(f"Champion v{resumen['champion_version']} vs Challenger v{resumen['challenger_version']}")
    logger.info(f"Acuerdo ARI: {resumen['acuerdo_ari']:.3f}")
    logger.info(f"B mejor en silhouette: {resumen['gana_b_pct']:.1f}% de repeticiones")
    logger.info(f"Veredicto: {resumen['veredicto']}")
    for nombre, ok in resumen["criterios"].items():
        logger.info(f"  [{'OK' if ok else 'NO'}] {nombre}")
    if resumen["cumple"]:
        logger.success("VEREDICTO: promover challenger")
    else:
        logger.success("VEREDICTO: mantener champion")
    monitor.guardar_ab("P3", resumen, "mibici-equipo20-p3-monitoreo")


if __name__ == "__main__":
    app()
