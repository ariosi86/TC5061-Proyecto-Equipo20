#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Orquestador de entrenamientos MLflow Projects para MiBici (P1/P2/P3).

Atajo para lanzar con ``mlflow run`` todas las versiones candidatas de una
pregunta en un solo comando, en lugar de lanzarlas una a una. Cada versión tiene sus
hiperparámetros hardcodeados.

No recibe parámetros de ejecución sueltos: solo elige la pregunta y, opcional,
quiénes de las versiones candidatas se lanzan (--versiones).
"""

import os
import subprocess
import sys

import typer

from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    P1_ALPHA_DEFAULT,
    P1_LR_DEFAULT,
    P1_MAX_ITER_DEFAULT,
    P2_CLASS_WEIGHT_DEFAULT,
    P2_UMBRAL,
    P2_MAX_ITER_DEFAULT,
    P3_K_ELEGIDO,
    P3_N_INIT_DEFAULT,
    RANDOM_SEED,
    URI_MLFLOW_TRACKING,
)

app = typer.Typer(help="Orquestador de entrenamientos MLflow Projects (P1/P2/P3).")


def _ejecutar(cmd: list[str], etiqueta: str) -> None:
    typer.secho(f"[Entrenamiento] {etiqueta}", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  $ {' '.join(cmd)}")
    env = os.environ.copy()
    # mlflow run crea la corrida antes de lanzar el script; ambos deben usar el mismo servidor
    env.setdefault("MLFLOW_TRACKING_URI", URI_MLFLOW_TRACKING)
    subprocess.run(cmd, check=True, env=env)


def _mlflow_run_cmd(entry_point: str, extra_params: list[str]) -> list[str]:
    """Construye el comando mlflow run para una versión dada."""
    cmd = [
        "mlflow", "run", ".", "-e", entry_point,
        "--env-manager", "local",
        "--experiment-name", MLFLOW_EXPERIMENT_NAME,
    ]
    for param in extra_params:
        clave, valor = param.split("=", 1)
        cmd += ["-P", f"{clave}={valor}"]
    return cmd


# ---------------------------------------------------------------------------
# Versiones candidatas por pregunta (hiperparámetros hardcodeados).
# Cada tupla: (nombre_etiqueta, nombre_modelo, lista_params_mlflow_run, entry_point)
# ---------------------------------------------------------------------------
VERSIONES_P1 = [
    ("baseline", "baseline", [], "train_p1"),
    ("poisson", "poisson", ["lr=0.05", "alpha=0.0001", "max_iter=300"], "train_p1"),
    ("lgbm_poisson", "lgbm_poisson", [], "train_p1"),
    ("lgbm_poisson_search", "lgbm_p_search", [], "train_p1"),
    ("lgbm_tweedie", "lgbm_tweedie", [], "train_p1"),
    ("lgbm_tweedie_search", "lgbm_t_search", [], "train_p1"),
]

VERSIONES_P2 = [
    ("baseline", "baseline", [], "train_p2"),
    ("logistic_balanced", "logistic", [f"umbral={P2_UMBRAL}", "class_weight=balanced", "max_iter=1000"], "train_p2"),
    ("logistic_sin_peso", "logistic", [f"umbral={P2_UMBRAL}", "class_weight=None", "max_iter=1000"], "train_p2"),
    ("logistic_umbral_sensible", "logistic", ["umbral=0.3", "class_weight=balanced", "max_iter=1000"], "train_p2"),
    ("logistic_search", "logistic_search", [], "train_p2"),
    ("gbm", "gbm", [], "train_p2"),
    ("gbm_search", "gbm_search", [], "train_p2"),
]

VERSIONES_P3 = [
    ("baseline", "baseline", [], "train_p3"),
    ("kmeans_search", "kmeans_search", ["n_init=10"], "train_p3"),
    ("kmeans_k3", "kmeans", ["n_clusters=3", "n_init=10"], "train_p3"),
    ("kmeans_k4", "kmeans", ["n_clusters=4", "n_init=10"], "train_p3"),
    ("kmeans_k5", "kmeans", ["n_clusters=5", "n_init=10"], "train_p3"),
    ("kmeans_k6", "kmeans", ["n_clusters=6", "n_init=10"], "train_p3"),
]


def _filtrar(versiones: list, lista: list | None) -> list:
    if lista is None or "todos" in lista:
        return versiones
    return [v for v in versiones if v[0] in lista]


@app.command()
def train(
    pregunta: str = typer.Option("p1", help="p1, p2 o p3"),
    versiones: str = typer.Option(
        "todos",
        "--versiones",
        help="Conjunto de versiones a entrenar: 'todos' o una lista separada por comas (p. ej. 'baseline,logistic').",
    ),
):
    """
    Entrena todas las versiones candidatas de la pregunta indicada mediante mlflow run.

    Cada versión se ejecuta como un proyecto MLflow independiente
    (``mlflow run . -e train_pX -P ...``), registrando su propio run en MLflow.
    """
    p = pregunta.lower()
    if p not in ("p1", "p2", "p3"):
        typer.secho(f"Pregunta invalida: {pregunta}. Usa p1, p2 o p3.", fg=typer.colors.RED)
        raise typer.Exit(1)

    lista = [v.strip() for v in versiones.split(",")] if versiones != "todos" else None

    if p == "p1":
        versiones_list = _filtrar(VERSIONES_P1, lista)
        entry_point = "train_p1"
    elif p == "p2":
        versiones_list = _filtrar(VERSIONES_P2, lista)
        entry_point = "train_p2"
    else:
        versiones_list = _filtrar(VERSIONES_P3, lista)
        entry_point = "train_p3"

    for nombre, modelo, extra, _ep in versiones_list:
        cmd = _mlflow_run_cmd(entry_point, ["modelo=" + modelo] + extra)
        _ejecutar(cmd, f"{p.upper()} - {nombre} ({modelo})")


@app.command(name="exec-single", help="Ejecuta una sola versión (usado por MLproject train).")
def exec_single(
    pregunta: str = typer.Option(..., "--pregunta", help="p1, p2 o p3"),
    modelo: str = typer.Option("baseline", help="Nombre del modelo a entrenar."),
    lr: float = typer.Option(P1_LR_DEFAULT, help="Tasa de aprendizaje (P1)."),
    alpha: float = typer.Option(P1_ALPHA_DEFAULT, help="Regularización (P1 Poisson)."),
    max_iter: int = typer.Option(P1_MAX_ITER_DEFAULT, help="Iteraciones máximas (P1 Poisson / P2 Logistic)."),
    umbral: float = typer.Option(P2_UMBRAL, help="Umbral de decisión de riesgo (P2)."),
    class_weight: str = typer.Option(P2_CLASS_WEIGHT_DEFAULT, help="Peso de clases para Regresión Logística (P2)."),
    n_clusters: int = typer.Option(P3_K_ELEGIDO, help="Numero de clusters (P3)."),
    n_init: int = typer.Option(P3_N_INIT_DEFAULT, help="n_init de KMeans (P3)."),
):
    """
    Ejecuta una sola versión directamente (sin mlflow run).

    Es el entry point usado por el entry point ``train`` de MLproject para evitar
    recursión infinita. Recibe la pregunta, el nombre del modelo y sus
    hiperparámetros, y despacha al módulo de entrenamiento correspondiente.
    """
    p = pregunta.lower()
    if p not in ("p1", "p2", "p3"):
        typer.secho(f"Pregunta invalida: {pregunta}. Usa p1, p2 o p3.", fg=typer.colors.RED)
        raise typer.Exit(1)

    if p == "p1":
        modulo = "proyecto_mibici.modeling.train_p1"
        modelo_cli = "--model"
        extra = []
        if modelo == "poisson":
            extra = ["--lr", str(lr), "--alpha", str(alpha), "--max_iter", str(max_iter)]
    elif p == "p2":
        modulo = "proyecto_mibici.modeling.train_p2"
        modelo_cli = "--modelo"
        cw = None if class_weight == "None" else class_weight
        extra = ["--umbral", str(umbral), "--class-weight", str(cw), "--max-iter", str(max_iter)]
    else:
        modulo = "proyecto_mibici.modeling.train_p3"
        modelo_cli = "--modelo"
        extra = ["--n-clusters", str(n_clusters), "--n-init", str(n_init)]

    cmd = [sys.executable, "-m", modulo, modelo_cli, modelo] + extra
    _ejecutar(cmd, f"{p.upper()} - single ({modelo})")


if __name__ == "__main__":
    app()
