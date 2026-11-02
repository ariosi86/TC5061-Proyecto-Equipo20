#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Suite de pruebas de integración y validación de artefactos para MiBici.
Ejecución independiente: pytest tests/ -v

Estas pruebas revisan los ARTEFACTOS que genera el flujo completo (datos procesados,
modelos .pkl, figuras y reportes). Esos archivos no se suben al repositorio
(data/, models/ y reports/ están en el .gitignore), así que en un clon recién hecho
no existen. En ese caso las pruebas se OMITEN (SKIPPED) con un mensaje que indica
qué ejecutar, en lugar de fallar.

Para generarlos, desde la raíz del proyecto:
    python -m proyecto_mibici.pipeline     # ingesta + entrenamiento + inferencia de prueba

Si un artefacto SÍ existe pero no corresponde al código actual (por ejemplo, un modelo
entrenado con una versión anterior), la prueba FALLA e indica que hay que reentrenar.
"""

import json

import numpy as np
import pandas as pd
import pytest

from proyecto_mibici.config import (
    FIGURES_DIR,
    MANIFEST_PATH,
    MODELS_DIR,
    P1_FEATURES_V2,
    P2_VARIABLES,
    P3_K_ELEGIDO,
    P3_VARIABLES,
    PROJ_ROOT,
    REPORTS_DIR,
    STATIONS_PROCESSED_PATH,
    TRIPS_PROCESSED_PATH,
    TRIPS_TRAIN_PATH,
    TRIPS_TEST_PATH,
)
from proyecto_mibici.modeling.model_p1 import DemandaLinealP1
from proyecto_mibici.modeling.model_p2 import RiesgoGBMP2
from proyecto_mibici.modeling.model_p3 import ClusterKMeansP3

# Modelos que revisa la suite. En P2 se usa el GBM, que es el modelo campeón.
RUTA_P1 = MODELS_DIR / "p1_linear_model.pkl"
RUTA_P2 = MODELS_DIR / "p2_gbm_model.pkl"
RUTA_P3 = MODELS_DIR / "p3_kmeans_model.pkl"

COMO_GENERAR_DATOS = "python -m proyecto_mibici.dataset"
COMO_GENERAR_TODO = "python -m proyecto_mibici.pipeline"


def omitir_si_faltan(rutas, como_generarlos):
    """Omite la prueba si falta alguno de los archivos, indicando cómo generarlos."""
    faltan = [r.relative_to(PROJ_ROOT).as_posix() for r in rutas if not r.exists()]
    if faltan:
        pytest.skip(f"Faltan {', '.join(faltan)}. Para generarlos: {como_generarlos}")


def test_archivos_datos_y_manifiesto():
    """Verifica que los Parquets procesados (maestro, train y test), el manifest y las estaciones existan y sean válidos."""
    omitir_si_faltan(
        [TRIPS_PROCESSED_PATH, STATIONS_PROCESSED_PATH, TRIPS_TRAIN_PATH, TRIPS_TEST_PATH, MANIFEST_PATH],
        COMO_GENERAR_DATOS,
    )

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "trips" in manifest and "sha256" in manifest["trips"]  # nosec B101
    assert manifest["trips"]["total_rows"] > 0  # nosec B101

    # Validación adicional de rangos en los parquets particionados
    df_train = pd.read_parquet(TRIPS_TRAIN_PATH, columns=["inicio_del_viaje"])
    df_test = pd.read_parquet(TRIPS_TEST_PATH, columns=["inicio_del_viaje"])

    assert len(df_train) > 0, "El dataset de train 2025 está vacío"  # nosec B101
    assert len(df_test) > 0, "El dataset de test 2026 está vacío"  # nosec B101

    assert df_train["inicio_del_viaje"].min().year == 2025, "Train contiene datos fuera del año 2025"  # nosec B101
    assert df_test["inicio_del_viaje"].min().month >= 4, "Test de 2026 inicia antes de abril"  # nosec B101
    assert df_test["inicio_del_viaje"].max().month <= 9, "Test de 2026 supera septiembre"  # nosec B101


@pytest.fixture(scope="module")
def modelos():
    """Carga los 3 modelos guardados (o omite las pruebas si aún no se han entrenado)."""
    omitir_si_faltan([RUTA_P1, RUTA_P2, RUTA_P3], COMO_GENERAR_TODO)
    return {
        "p1": DemandaLinealP1.load(RUTA_P1),
        "p2": RiesgoGBMP2.load(RUTA_P2),
        "p3": ClusterKMeansP3.load(RUTA_P3),
    }


def verificar_que_esten_vigentes(modelos):
    """Falla con un mensaje claro si algún modelo se entrenó con una versión anterior del código."""
    reentrenar = f"modelo desactualizado respecto al código actual; reentrenar con {COMO_GENERAR_TODO}"
    assert modelos["p3"].pipeline is not None, f"P3: {reentrenar}"  # nosec B101
    assert list(modelos["p2"].features_) == list(P2_VARIABLES), f"P2: {reentrenar}"  # nosec B101


def test_modelos_serializados_y_pipelines(modelos):
    """Verifica que los 3 modelos .pkl carguen y correspondan al código y configuración actuales."""
    verificar_que_esten_vigentes(modelos)

    # P1 y P3 escalan dentro de su pipeline; P2 (GBM) no necesita escalar
    assert "scaler" in modelos["p1"].pipeline.named_steps  # nosec B101
    assert "scaler" in modelos["p3"].pipeline.named_steps  # nosec B101


def test_inferencia_sintetica_modelos(modelos):
    """Verifica que los modelos cargados respondan con formatos válidos ante datos sintéticos."""
    verificar_que_esten_vigentes(modelos)
    rng = np.random.default_rng(42)

    # P1: Predicciones >= 0
    X_p1 = pd.DataFrame(rng.uniform(0, 10, size=(10, len(P1_FEATURES_V2))), columns=P1_FEATURES_V2)
    preds_p1 = modelos["p1"].predict(X_p1)
    assert len(preds_p1) == 10  # nosec B101
    assert (preds_p1 >= 0).all(), "P1 produjo demandas negativas"  # nosec B101

    # P2: Probabilidades en [0, 1] y una predicción binaria por fila
    X_p2 = pd.DataFrame(rng.uniform(-1, 1, size=(10, len(P2_VARIABLES))), columns=P2_VARIABLES)
    probs_p2 = modelos["p2"].predict_proba(X_p2)
    preds_p2 = modelos["p2"].predict(X_p2)
    assert ((probs_p2 >= 0.0) & (probs_p2 <= 1.0)).all(), "Probabilidades fuera de [0, 1]"  # nosec B101
    assert len(preds_p2) == 10  # nosec B101
    assert set(np.unique(preds_p2)) <= {0, 1, False, True}  # nosec B101

    # P3: Clusters en [0, K-1]
    X_p3 = pd.DataFrame(rng.uniform(0, 1, size=(10, len(P3_VARIABLES))), columns=P3_VARIABLES)
    clusters_p3 = modelos["p3"].predict(X_p3)
    assert set(clusters_p3).issubset(set(range(P3_K_ELEGIDO)))  # nosec B101


def test_figuras_y_reportes_generados():
    """Verifica que las gráficas en reports/figures/ y los reportes de inferencia se hayan creado."""
    reportes = [
        REPORTS_DIR / "p3_comparacion.parquet",          # lo genera train_p3.py
        REPORTS_DIR / "predicciones_p1_test.parquet",    # los tres siguientes, la etapa de
        REPORTS_DIR / "predicciones_p2_test.parquet",    # inferencia de pipeline.py
        REPORTS_DIR / "predicciones_p3_test.csv",
    ]
    omitir_si_faltan(reportes, COMO_GENERAR_TODO)

    pngs = list(FIGURES_DIR.glob("*.png"))
    assert len(pngs) >= 6, f"Se esperaban al menos 6 gráficas en reports/figures/, se encontraron {len(pngs)}"  # nosec B101
