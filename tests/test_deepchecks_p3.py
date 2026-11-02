"""Validación de datos de P3 con DeepChecks.

Se construye el perfil funcional de cada estación con el mismo código de train_p3 a partir 
de los Parquet de data/processed/. Si esos archivos no existen las pruebas se omiten. 
Primero hay que correr `python -m proyecto_mibici.dataset`.

P3 no tiene variable objetivo, así que los chequeos revisan solo las variables de entrada
de KMeans. Los reportes HTML quedan en reports/deepchecks/ como evidencia.
"""

import warnings

import numpy as np

# DeepChecks 0.19.1 todavía usa nombres que numpy 2 eliminó (np.Inf, np.NaN)
for _nombre, _valor in (("Inf", np.inf), ("NaN", np.nan)):
    if not hasattr(np, _nombre):
        setattr(np, _nombre, _valor)

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from deepchecks.tabular import Dataset, Suite  # noqa: E402
from deepchecks.tabular.checks import (  # noqa: E402
    DataDuplicates,
    FeatureDrift,
    IsSingleValue,
    MixedDataTypes,
    MixedNulls,
    MultivariateDrift,
    PercentOfNulls,
)

from proyecto_mibici.config import (  # noqa: E402
    P3_DURACION_MAX_MIN,
    P3_MIN_VIAJES_ESTACION,
    P3_TEST_FIN,
    P3_TEST_INICIO,
    P3_TRAIN_FIN,
    P3_TRAIN_INICIO,
    P3_VARIABLES,
    PROJ_ROOT,
    STATIONS_PROCESSED_PATH,
    TRIPS_PROCESSED_PATH,
)
from proyecto_mibici.features_p3 import (  # noqa: E402
    construir_perfil_estaciones,
    limpiar_viajes_p3,
)

warnings.filterwarnings("ignore")

COLUMNAS_VIAJES = ["viaje_id", "inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"]
CARPETA_REPORTES = PROJ_ROOT / "reports" / "deepchecks"

pytestmark = pytest.mark.skipif(
    not (TRIPS_PROCESSED_PATH.exists() and STATIONS_PROCESSED_PATH.exists()),
    reason="Faltan los Parquet procesados: correr `python -m proyecto_mibici.dataset`",
)


@pytest.fixture(scope="module")
def perfiles_p3():
    """Perfil real de estaciones en entrenamiento (2025) y prueba (may-sep 2026).

    Se construye una sola vez para todas las pruebas del archivo.
    """
    viajes = pd.read_parquet(TRIPS_PROCESSED_PATH, columns=COLUMNAS_VIAJES)
    estaciones = pd.read_parquet(STATIONS_PROCESSED_PATH)
    limpios = limpiar_viajes_p3(viajes, estaciones, duracion_max_min=P3_DURACION_MAX_MIN)

    def perfil_entre(inicio, fin):
        filas = limpios[(limpios["inicio_del_viaje"] >= inicio) & (limpios["inicio_del_viaje"] < fin)]
        return construir_perfil_estaciones(filas, min_viajes=P3_MIN_VIAJES_ESTACION)

    return {
        "entrenamiento": perfil_entre(P3_TRAIN_INICIO, P3_TRAIN_FIN),
        "prueba": perfil_entre(P3_TEST_INICIO, P3_TEST_FIN),
    }


def como_dataset(perfil):
    """Dataset de DeepChecks con las variables de KMeans (todas numéricas, sin objetivo)."""
    return Dataset(perfil[P3_VARIABLES], cat_features=[])


def guardar_y_revisar(resultado, nombre_archivo):
    """Guarda el reporte HTML y devuelve la lista de condiciones que no se cumplieron."""
    CARPETA_REPORTES.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_REPORTES / nombre_archivo
    #Si el archivo existe crea "nombre (1).html".
    ruta.unlink(missing_ok=True)
    resultado.save_as_html(str(ruta), as_widget=False)
    fallas = []
    for chequeo in resultado.results:
        if not hasattr(chequeo, "conditions_results"):
            fallas.append(f"{chequeo.get_header()}: no se ejecutó ({chequeo.exception})")
            continue
        for condicion in chequeo.conditions_results:
            if not condicion.is_pass():
                fallas.append(f"{chequeo.get_header()}: {condicion.name} -> {condicion.details}")
    return fallas


# ---------------------------------------------------------------------------
# 1. Integridad del perfil de estaciones
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("conjunto", ["entrenamiento", "prueba"])
def test_integridad_del_perfil_p3(perfiles_p3, conjunto):
    suite = Suite(
        f"Integridad del perfil de estaciones de P3 ({conjunto})",
        PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0),
        MixedNulls().add_condition_different_nulls_less_equal_to(1),
        MixedDataTypes().add_condition_rare_type_ratio_not_in_range(),
        IsSingleValue().add_condition_not_single_value(),
        # Dos estaciones con el mismo perfil exacto indicarían un error al agrupar
        DataDuplicates().add_condition_ratio_less_or_equal(0),
    )

    resultado = suite.run(como_dataset(perfiles_p3[conjunto]))
    fallas = guardar_y_revisar(resultado, f"p3_integridad_{conjunto}.html")

    assert not fallas, "\n".join(fallas)


def test_reglas_de_dominio_p3(perfiles_p3):
    """Reglas propias del perfil de P3 que DeepChecks no conoce."""
    for perfil in perfiles_p3.values():
        pct_horarios = perfil[["pct_madrugada", "pct_manana", "pct_mediodia", "pct_tarde", "pct_noche"]]

        assert perfil.index.is_unique  # una fila por estación
        assert np.allclose(pct_horarios.sum(axis=1), 1.0)
        assert perfil["pct_fin_de_semana"].between(0, 1).all()
        assert perfil["balance_neto"].between(-1, 1).all()
        assert (perfil["duracion_media"] > 0).all()
        assert (perfil["duracion_media"] <= P3_DURACION_MAX_MIN).all()


# ---------------------------------------------------------------------------
# 2. Drift entre entrenamiento (2025) y prueba (may-sep 2026)
# ---------------------------------------------------------------------------
def test_estaciones_de_2025_siguen_operando_en_2026(perfiles_p3):
    # Los grupos se aprendieron con las estaciones de 2025. 
    # Si alguna de esas estaciones desapareciera, el modelo no podría asignarle un grupo en 2026.
    # Se exige que al menos 95 % siga activa.
    entrenamiento = perfiles_p3["entrenamiento"].index
    prueba = perfiles_p3["prueba"].index

    assert entrenamiento.isin(prueba).mean() >= 0.95


def test_sin_drift_en_las_estaciones_comunes(perfiles_p3):
    entrenamiento = perfiles_p3["entrenamiento"]
    prueba = perfiles_p3["prueba"]
    comunes = entrenamiento.index.intersection(prueba.index)

    # Sistema completo sin condición
    # incluye las estaciones nuevas de 2026
    completo = Suite("Drift del perfil de P3: todas las estaciones", FeatureDrift(), MultivariateDrift())
    guardar_y_revisar(
        completo.run(train_dataset=como_dataset(entrenamiento), test_dataset=como_dataset(prueba)),
        "p3_drift_todas_las_estaciones.html",
    )

    # Sistema condicionado
    # las mismas estaciones en ambos periodos para medir cambios de uso 
    # sin confundirlos con estaciones nuevas o desaparecidas
    suite = Suite(
        "Drift del perfil de P3: estaciones presentes en ambos periodos",
        FeatureDrift().add_condition_drift_score_less_than(max_allowed_numeric_score=0.2),
        MultivariateDrift().add_condition_overall_drift_value_less_than(0.25),
    )
    resultado = suite.run(
        train_dataset=como_dataset(entrenamiento.loc[comunes]),
        test_dataset=como_dataset(prueba.loc[comunes]),
    )
    fallas = guardar_y_revisar(resultado, "p3_drift_estaciones_comunes.html")

    assert not fallas, "\n".join(fallas)