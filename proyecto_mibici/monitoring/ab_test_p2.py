#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A/B testing de P2 (Entregable 8): champion contra challenger.

Usa la clase EvidentlyMonitor.ab_test_p2() para el cálculo central.
No modifica el Registry salvo con --promover.
"""

import typer
from loguru import logger

from proyecto_mibici.monitoring import EvidentlyMonitor

app = typer.Typer(help="A/B testing de P2: champion vs challenger.")


@app.command()
def main(
    champion_version: int | None = typer.Option(
        None, "--champion-version", help="Por defecto, la versión del alias 'champion'."),
    challenger_version: int | None = typer.Option(
        None, "--challenger-version", help="Por defecto, la versión del alias 'challenger'."),
    n_boot: int = typer.Option(300, "--n-boot", help="Repeticiones del bootstrap."),
    promover: bool = typer.Option(
        False, "--promover", help="Mueve el alias 'champion' al challenger si cumple."),
):
    monitor = EvidentlyMonitor(
        numerical_columns=[],
        categorical_columns=[],
        usar_api_evidently=False,
    )
    resumen = monitor.ab_test_p2(
        champion_version=champion_version,
        challenger_version=challenger_version,
        n_boot=n_boot,
        promover=promover,
    )

    logger.info(
        f"Champion v{resumen['champion_version']} vs "
        f"Challenger v{resumen['challenger_version']}"
    )
    logger.info(f"F1: {resumen['metricas_champion']['f1_score']:.4f} -> "
                f"{resumen['metricas_challenger']['f1_score']:.4f}")
    logger.info(f"PR-AUC: {resumen['metricas_champion']['pr_auc']:.4f} -> "
                f"{resumen['metricas_challenger']['pr_auc']:.4f}")
    logger.info(f"IC95 F1: [{resumen['ic95']['f1'][0]:+.4f}, "
                f"{resumen['ic95']['f1'][1]:+.4f}]")
    logger.info(f"IC95 PR-AUC: [{resumen['ic95']['pr_auc'][0]:+.4f}, "
                f"{resumen['ic95']['pr_auc'][1]:+.4f}]")
    logger.info(f"Veredicto: {resumen['veredicto']}")
    for nombre, ok in resumen["criterios"].items():
        logger.info(f"  [{'OK' if ok else 'NO'}] {nombre}")
    if resumen["cumple"]:
        logger.success("VEREDICTO: promover challenger")
    else:
        logger.success("VEREDICTO: mantener champion")
    monitor.guardar_ab("P2", resumen, "mibici-equipo20-p2-monitoreo")


if __name__ == "__main__":
    app()
