#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A/B testing offline de P1 (Entregable 8): champion vs challenger sobre octubre 2026.

Usa la clase EvidentlyMonitor.ab_test_p1() para el cálculo central.
No modifica el Registry salvo con --promover.
"""

import typer
from loguru import logger

from proyecto_mibici.monitoring import EvidentlyMonitor

app = typer.Typer(help="A/B testing offline champion vs challenger (P1, octubre 2026).")


@app.command()
def main(
    champion_version: int | None = typer.Option(
        None, "--champion-version", help="Por defecto, la versión del alias 'champion'."),
    challenger_version: int | None = typer.Option(
        None, "--challenger-version", help="Por defecto, la versión del alias 'challenger'."),
    experimento: str = typer.Option(
        "mibici-equipo20-p1-monitoreo", "--experimento",
        help="Experimento de MLflow del A/B."),
    min_mejora_mae: float = typer.Option(
        1.0, "--min-mejora-mae", help="Mejora mínima de MAE (%) para promover."),
    promover: bool = typer.Option(
        False, "--promover", help="Mueve el alias 'champion' al challenger si cumple."),
):
    monitor = EvidentlyMonitor(
        numerical_columns=[],
        categorical_columns=[],
        usar_api_evidently=False,
    )
    resumen = monitor.ab_test_p1(
        champion_version=champion_version,
        challenger_version=challenger_version,
        min_mejora_mae_pct=min_mejora_mae,
        promover=promover,
        experimento=experimento,
    )

    logger.info(f"Champion v{resumen['champion_version']} vs Challenger v{resumen['challenger_version']}")
    logger.info(f"Mejora MAE: {resumen['mejora_mae_pct']:.2f}%")
    logger.info(f"IC95%: [{resumen['ic95_inf']:.4f}, {resumen['ic95_sup']:.4f}]")
    logger.info(f"Veredicto: {resumen['veredicto']}")
    for nombre, ok in resumen["criterios"].items():
        logger.info(f"  [{'OK' if ok else 'NO'}] {nombre}")
    if resumen["cumple"]:
        logger.success("VEREDICTO: reemplazar champion")
    else:
        logger.success("VEREDICTO: mantener champion")
    monitor.guardar_ab("P1", resumen, experimento)


if __name__ == "__main__":
    app()
