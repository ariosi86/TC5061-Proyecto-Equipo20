"""Validación de datos de P2 con DeepChecks: integridad del dataset y drift entre conjuntos.

A diferencia de las pruebas unitarias, estas usan el dataset REAL de P2, construido
con construir_dataset_p2() a partir de los Parquet de data/processed/. Si esos
archivos no existen (por ejemplo, en un clon recién hecho), las pruebas se omiten:
primero hay que correr `python -m proyecto_mibici.dataset`.

Los reportes HTML quedan en reports/deepchecks/ como evidencia.
"""

import warnings

import numpy as np

# DeepChecks 0.19.1 todavía usa nombres que numpy 2 eliminó (np.Inf, np.NaN).
# Se restauran como alias ANTES de importarlo; no cambia nada del proyecto.
for _nombre, _valor in (("Inf", np.inf), ("NaN", np.nan)):
    if not hasattr(np, _nombre):
        setattr(np, _nombre, _valor)

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from deepchecks.tabular import Dataset, Suite  # noqa: E402
from deepchecks.tabular.checks import (  # noqa: E402
    ClassImbalance,
    DataDuplicates,
    DateTrainTestLeakageOverlap,
    FeatureDrift,
    FeatureLabelCorrelation,
    IsSingleValue,
    LabelDrift,
    MixedDataTypes,
    MixedNulls,
    MultivariateDrift,
    NewLabelTrainTest,
    PercentOfNulls,
)

from proyecto_mibici.config import (  # noqa: E402
    P2_OBJETIVO,
    P2_VARIABLES,
    PROJ_ROOT,
    RANDOM_SEED,
    STATIONS_PROCESSED_PATH,
    TRIPS_PROCESSED_PATH,
)
from proyecto_mibici.features_p2 import construir_dataset_p2  # noqa: E402

warnings.filterwarnings("ignore")

VARIABLES_CATEGORICAS = ["hora_del_dia", "dia_semana", "fin_de_semana", "riesgo_vacia", "riesgo_llena"]
LLAVE = ["estacion_id", "hora"]          # una fila por estación y hora
TAMANO_MUESTRA = 50_000                  # filas por conjunto para los chequeos de drift
CARPETA_REPORTES = PROJ_ROOT / "reports" / "deepchecks"

pytestmark = pytest.mark.skipif(
    not (TRIPS_PROCESSED_PATH.exists() and STATIONS_PROCESSED_PATH.exists()),
    reason="Faltan los Parquet procesados: correr `python -m proyecto_mibici.dataset`",
)


@pytest.fixture(scope="module")
def serie_p2():
    """Dataset real de P2 (se construye una sola vez para todas las pruebas del archivo)."""
    return construir_dataset_p2()


def como_dataset(df, columnas_extra=()):
    categoricas = [c for c in VARIABLES_CATEGORICAS if c in P2_VARIABLES]
    return Dataset(
        df[list(columnas_extra) + P2_VARIABLES + [P2_OBJETIVO]],
        label=P2_OBJETIVO,
        features=P2_VARIABLES,
        cat_features=categoricas,
        datetime_name="hora" if "hora" in columnas_extra else None,
    )


def guardar_y_revisar(resultado, nombre_archivo):
    """Guarda el reporte HTML y devuelve la lista de condiciones que no se cumplieron."""
    CARPETA_REPORTES.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_REPORTES / nombre_archivo
    # DeepChecks no sobrescribe: si el archivo existe crea "nombre (1).html". Se borra antes
    # para que siempre quede un solo reporte, el de la última corrida.
    ruta.unlink(missing_ok=True)
    # as_widget=False: HTML estático, sin depender de los widgets de Jupyter
    resultado.save_as_html(str(ruta), as_widget=False)
    fallas = []
    for chequeo in resultado.results:
        if not hasattr(chequeo, "conditions_results"):          # el chequeo no pudo correr
            fallas.append(f"{chequeo.get_header()}: no se ejecutó ({chequeo.exception})")
            continue
        for condicion in chequeo.conditions_results:
            if not condicion.is_pass():
                fallas.append(f"{chequeo.get_header()}: {condicion.name} -> {condicion.details}")
    return fallas


# ---------------------------------------------------------------------------
# 1. Integridad: tipos, faltantes, duplicados y calidad del objetivo
# ---------------------------------------------------------------------------
def test_integridad_del_dataset_p2(serie_p2):
    # Corre sobre el dataset completo (los tres conjuntos)
    suite = Suite(
        "Integridad del dataset de P2",
        # n_samples alto: revisar las 4.8 M de filas (por defecto solo toma una muestra de 100 mil)
        PercentOfNulls(n_samples=10_000_000).add_condition_percent_of_nulls_not_greater_than(0),
        MixedNulls().add_condition_different_nulls_less_equal_to(1),
        MixedDataTypes().add_condition_rare_type_ratio_not_in_range(),
        IsSingleValue().add_condition_not_single_value(),
        # Un duplicado real es la misma estación en la misma hora dos veces.
        # (Filas con los mismos valores en distintas horas son normales: p. ej., madrugadas en 0.)
        DataDuplicates(columns=LLAVE).add_condition_ratio_less_or_equal(0),
        # Informativo (sin condición): proporción de horas en riesgo, la clase minoritaria
        ClassImbalance(),
        FeatureLabelCorrelation(random_state=RANDOM_SEED).add_condition_feature_pps_less_than(0.8),
    )

    resultado = suite.run(como_dataset(serie_p2, columnas_extra=LLAVE))
    fallas = guardar_y_revisar(resultado, "p2_integridad.html")

    assert not fallas, "\n".join(fallas)


def test_reglas_de_dominio_p2(serie_p2):
    """Reglas propias de P2 que DeepChecks no conoce."""
    assert (serie_p2["capacidad"] > 0).all()
    assert (serie_p2["salidas"] >= 0).all() and (serie_p2["llegadas"] >= 0).all()
    assert serie_p2["hora_del_dia"].between(0, 23).all()
    assert serie_p2["dia_semana"].between(0, 6).all()
    assert set(serie_p2[P2_OBJETIVO].unique()) <= {False, True}
    assert set(serie_p2["conjunto"].unique()) == {"entrenamiento", "validacion", "prueba"}


# ---------------------------------------------------------------------------
# 2. Drift y fuga temporal: entrenamiento contra validación y contra prueba
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("conjunto", ["validacion", "prueba"])
def test_sin_drift_ni_fuga_temporal(serie_p2, conjunto):
    def muestra(nombre):
        filas = serie_p2[serie_p2["conjunto"] == nombre]
        return filas.sample(min(TAMANO_MUESTRA, len(filas)), random_state=RANDOM_SEED)

    entrenamiento = como_dataset(muestra("entrenamiento"), columnas_extra=["hora"])
    comparado = como_dataset(muestra(conjunto), columnas_extra=["hora"])

    suite = Suite(
        f"Drift de P2: entrenamiento vs {conjunto}",
        # Todas las fechas del conjunto comparado deben ser posteriores a las de entrenamiento
        DateTrainTestLeakageOverlap().add_condition_leakage_ratio_less_or_equal(0),
        NewLabelTrainTest().add_condition_new_labels_number_less_or_equal(0),
        FeatureDrift().add_condition_drift_score_less_than(
            max_allowed_categorical_score=0.2, max_allowed_numeric_score=0.2
        ),
        LabelDrift().add_condition_drift_score_less_than(0.15),
        MultivariateDrift().add_condition_overall_drift_value_less_than(0.25),
    )

    resultado = suite.run(train_dataset=entrenamiento, test_dataset=comparado)
    fallas = guardar_y_revisar(resultado, f"p2_drift_{conjunto}.html")

    assert not fallas, "\n".join(fallas)
