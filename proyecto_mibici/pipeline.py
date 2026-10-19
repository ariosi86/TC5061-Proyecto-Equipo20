#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Orquestador maestro End-to-End del proyecto MiBici.
Ejecuta:
  1. Ingesta y creación de Parquets + manifest.json (dataset.py)
  2. Entrenamiento, evaluación y tracking en MLflow (train_p1, train_p2, train_p3)
  3. Ejecución de inferencia CLI sobre muestras de prueba (predict_p1, predict_p2, predict_p3)
  4. Suite de pruebas automatizadas (pytest)

Ejecución: python -m proyecto_mibici.pipeline
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from proyecto_mibici.config import (
    P3_DURACION_MAX_MIN,
    P3_MIN_VIAJES_ESTACION,
    P3_TEST_FIN,
    P3_TEST_INICIO,
    PROJ_ROOT,
    REPORTS_DIR,
    STATIONS_PROCESSED_PATH,
    TRIPS_PROCESSED_PATH,
)
from proyecto_mibici.dataset import main as run_dataset
from proyecto_mibici.features import (
    construir_dataset_p1,
    construir_dataset_p2,
    construir_perfil_estaciones,
    limpiar_viajes_p3,
)
from proyecto_mibici.modeling.predict_p1 import main as run_predict_p1
from proyecto_mibici.modeling.predict_p2 import main as run_predict_p2
from proyecto_mibici.modeling.predict_p3 import main as run_predict_p3
from proyecto_mibici.modeling.train_p1 import main as run_train_p1
from proyecto_mibici.modeling.train_p2 import main as run_train_p2
from proyecto_mibici.modeling.train_p3 import main as run_train_p3

app = typer.Typer(help="Orquestador End-to-End (Datos + Train + Predict + Testing) para MiBici.")


def ejecutar_pruebas_inferencia():
    """Genera muestras de validación y ejecuta los 3 scripts de inferencia (predict_p1, p2, p3)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    sample_p1_out = REPORTS_DIR / "predicciones_p1_test.parquet"
    sample_p2_out = REPORTS_DIR / "predicciones_p2_test.parquet"
    sample_p3_out = REPORTS_DIR / "predicciones_p3_test.csv"

    # --- FASE 1: GENERACIÓN DE INFERENCIA EN CARPETA TEMPORAL ---
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. P1: Demanda Horaria
        logger.info("Generando muestra y ejecutando inferencia P1 (predict_p1.py)...")
        df_p1, _ = construir_dataset_p1()
        sample_p1_in = tmp_path / "sample_p1_input.parquet"
        df_p1.tail(200).to_parquet(sample_p1_in, index=False)
        run_predict_p1(input_path=sample_p1_in, output_path=sample_p1_out)

        # 2. P2: Riesgo de Desbalance
        logger.info("Generando muestra y ejecutando inferencia P2 (predict_p2.py)...")
        df_p2 = construir_dataset_p2()
        sample_p2_in = tmp_path / "sample_p2_input.parquet"
        df_p2.sample(n=min(200, len(df_p2)), random_state=42).to_parquet(
            sample_p2_in, index=False
        )
        run_predict_p2(input_path=sample_p2_in, output_path=sample_p2_out, umbral=0.5)

        # 3. P3: Clustering de Estaciones
        logger.info("Generando muestra y ejecutando inferencia P3 (predict_p3.py)...")
        cols = ["viaje_id", "inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"]
        df_trips = pd.read_parquet(TRIPS_PROCESSED_PATH, columns=cols)
        df_stations = pd.read_parquet(STATIONS_PROCESSED_PATH)
        df_clean = limpiar_viajes_p3(df_trips, df_stations, duracion_max_min=P3_DURACION_MAX_MIN)
        df_test_p3 = df_clean[
            (df_clean["inicio_del_viaje"] >= P3_TEST_INICIO)
            & (df_clean["inicio_del_viaje"] < P3_TEST_FIN)
        ]
        perfil_test = construir_perfil_estaciones(df_test_p3, min_viajes=P3_MIN_VIAJES_ESTACION)
        sample_p3_in = tmp_path / "sample_p3_input.parquet"
        perfil_test.reset_index().to_parquet(sample_p3_in, index=False)
        run_predict_p3(input_path=sample_p3_in, output_path=sample_p3_out)

    # --- FASE 2: TABLERO EJECUTIVO UNIFICADO EN UN SOLO LOG ---
    W = 76
    lineas = [
        "",
        "┌" + "─" * (W - 2) + "┐",
        f"│{'TABLERO EJECUTIVO DE INFERENCIA EN PRODUCCIÓN':^{W - 2}}│",
        "├" + "─" * (W - 2) + "┤",
    ]

    # P1: Demanda Horaria
    if sample_p1_out.exists():
        df_p1_res = pd.read_parquet(sample_p1_out)
        media_p1 = df_p1_res["demanda_predicha"].mean()
        max_p1 = df_p1_res["demanda_predicha"].max()
        lineas.extend([
            f"│  [P1] DEMANDA HORARIA (Regresión Lineal)                              │",
            f"│       • Registros: {len(df_p1_res):<6} | Media: {media_p1:.2f} viajes/h | Pico máx: {max_p1:.2f} viajes/h │",
            f"│       • Vista previa:                                                │",
        ])
        cols_p1 = [c for c in ["origen_id", "target_hour", "demanda_predicha"] if c in df_p1_res.columns]
        if not cols_p1:
            cols_p1 = ["demanda_predicha"]
        for row in df_p1_res[cols_p1].head(3).to_string(index=False).split("\n"):
            lineas.append(f"│         {row:<63}│")
        lineas.append("├" + "─" * (W - 2) + "┤")

    # P2: Riesgo de Desbalance
    if sample_p2_out.exists():
        df_p2_res = pd.read_parquet(sample_p2_out)
        n_riesgo = int(df_p2_res["pred_riesgo"].sum())
        total_p2 = len(df_p2_res)
        pct = (n_riesgo / total_p2 * 100) if total_p2 > 0 else 0.0
        lineas.extend([
            f"│  [P2] RIESGO DE DESBALANCE (Logistic Regression)                      │",
            f"│       • Estaciones en riesgo: {n_riesgo}/{total_p2} ({pct:.1f}%) [Umbral: 0.50]           │",
            f"│       • Vista previa:                                                │",
        ])
        cols_p2 = [c for c in ["estacion_id", "hora", "prob_riesgo", "pred_riesgo"] if c in df_p2_res.columns]
        if cols_p2:
            for row in df_p2_res[cols_p2].head(3).to_string(index=False).split("\n"):
                lineas.append(f"│         {row:<63}│")
        lineas.append("├" + "─" * (W - 2) + "┤")

    # P3: Clustering de Estaciones
    if sample_p3_out.exists():
        df_p3_res = pd.read_csv(sample_p3_out)
        dist_p3 = str(df_p3_res["grupo"].value_counts().to_dict())
        lineas.extend([
            f"│  [P3] CLUSTERING FUNCIONAL (K-Means K=4)                             │",
            f"│       • Asignadas: {len(df_p3_res)} estaciones                                        │",
            f"│       • Distribución: {dist_p3:<49}│",
            f"│       • Vista previa:                                                │",
        ])
        cols_p3 = [c for c in ["origen_id", "grupo", "viajes_dia_log"] if c in df_p3_res.columns]
        if not cols_p3:
            cols_p3 = df_p3_res.columns[:3]
        for row in df_p3_res[cols_p3].head(3).to_string(index=False).split("\n"):
            lineas.append(f"│         {row:<63}│")
        lineas.append("└" + "─" * (W - 2) + "┘")

    logger.info("\n" + "\n".join(lineas))


def ejecutar_suite_pytest():
    """Ejecuta pytest directamente sobre el directorio interno proyecto_mibici/tests/."""
    tests_dir = PROJ_ROOT / "proyecto_mibici" / "tests"
    if tests_dir.exists():
        logger.info(f"Ejecutando suite de pruebas automatizadas en {tests_dir}...")
        resultado = subprocess.run(
            [sys.executable, "-m", "pytest", str(tests_dir), "-v"],
            cwd=PROJ_ROOT,
        )
        if resultado.returncode != 0:
            raise RuntimeError("La suite de pruebas de pytest reportó fallos.")
    else:
        logger.warning(f"No se encontró el directorio '{tests_dir}'. Omitiendo pytest.")


@app.command()
def main(
    skip_data: bool = typer.Option(
        False,
        "--skip-data",
        help="Omite la descarga de datos si los Parquets ya existen.",
    ),
    force_data: bool = typer.Option(
        False,
        "--force-data",
        "--force",
        help="Forzar la descarga y procesamiento de datos desde cero.",
    ),
    skip_tests: bool = typer.Option(
        False,
        "--skip-tests",
        help="Omite la ejecución de inferencia de prueba y pytest.",
    ),
):
    inicio_total = time.time()

    logger.info("=" * 76)
    logger.info("       INICIANDO PIPELINE INTEGRAL MIBICI (DATOS + TRAIN + TESTS)    ")
    logger.info("=" * 76)

    # ETAPA 1: DATOS
    archivos_existen = TRIPS_PROCESSED_PATH.exists() and STATIONS_PROCESSED_PATH.exists()

    if force_data:
        logger.info("[Etapa 1/6] Parámetro --force activado. Re-descargando datos...")
        run_dataset()
    elif skip_data and archivos_existen:
        logger.info("[Etapa 1/6] Omitiendo descarga (--skip-data activado y parquets detectados).")
    elif not archivos_existen:
        logger.info("[Etapa 1/6] Parquets no encontrados. Ejecutando dataset.py...")
        run_dataset()
    else:
        logger.info("[Etapa 1/6] Parquets detectados. Usando datos existentes.")

    # ETAPA 2: ENTRENAMIENTO P1
    logger.info("[Etapa 2/6] Entrenando y evaluando P1 (Demanda Horaria - 6 Modelos)...")
    run_train_p1()

    # ETAPA 3: ENTRENAMIENTO P2
    logger.info("[Etapa 3/6] Entrenando y evaluando P2 (Riesgo de Desbalance)...")
    run_train_p2()

    # ETAPA 4: ENTRENAMIENTO P3
    logger.info("[Etapa 4/6] Entrenando y evaluando P3 (Clustering K-Means K=4)...")
    run_train_p3()

    if not skip_tests:
        # ETAPA 5: INFERENCIA CLI
        logger.info("[Etapa 5/6] Ejecutando pruebas de inferencia (predict_p1, p2, p3)...")
        ejecutar_pruebas_inferencia()

        # ETAPA 6: PYTEST
        logger.info("[Etapa 6/6] Ejecutando validaciones y pruebas unitarias (pytest)...")
        ejecutar_suite_pytest()

    duracion = (time.time() - inicio_total) / 60
    logger.success("=" * 76)
    logger.success(f" PIPELINE Y TESTING COMPLETADOS CON ÉXITO EN {duracion:.2f} MINUTOS ")
    logger.success("=" * 76)


if __name__ == "__main__":
    app()