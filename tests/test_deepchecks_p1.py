"""
Calidad de datos de P1 con DeepChecks + pytest.

Dos grupos:
  A) Sobre los DATOS REALES (el parquet de la tabla de modelado). Se saltan solos si no hay datos.
     Dónde buscar los datos: variable de entorno P1_DATASET_PARQUET=ruta/al/parquet
  B) Sobre datos SINTÉTICOS con problemas inyectados: demuestran que los chequeos sí detectan fallas.
"""
import os

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

try:
    # IMPORTANTE: primero el módulo del script. Al importarse aplica los parches de compatibilidad
    # (NumPy 2, scikit-learn >= 1.6, Python 3.14) que DeepChecks necesita para poder importarse.
    from validar_datos_p1 import (
        COL_ESTACION,
        armar_dataset,
        cargar_df_p1,
        columna_tiempo,
        dividir_train_test,
    )
    from deepchecks.tabular.checks import (
        DataDuplicates,
        DateTrainTestLeakageOverlap,
        FeatureDrift,
        FeatureLabelCorrelation,
        IsSingleValue,
        LabelDrift,
    )
except Exception as e:  # no instalado o incompatible con las versiones del entorno
    pytest.skip(f"DeepChecks no disponible en este entorno: {e}", allow_module_level=True)

from proyecto_mibici.config import P1_DATASET_PARQUET, P1_FEATURES_V2, P1_TARGET

# Umbrales: son decisiones nuestras, ajustarlas con los datos reales y justificarlas en el README.
MAX_DRIFT_FEATURES = 0.2
MAX_DRIFT_TARGET = 0.2
MAX_PPS = 0.8   # poder predictivo máximo de una sola feature sobre el target (valor por defecto de DeepChecks)
FEATURES_DRIFT = ["lag_1h", "lag_24h", "lag_168h", "rolling_3h", "rolling_168h"]


# ======================= A) DATOS REALES =======================

RUTA_PARQUET_P1 = Path("data/interim/p1_modelado.parquet")


@pytest.fixture(scope="module")
def df_real():
    ruta = Path(os.environ.get("P1_DATASET_PARQUET", RUTA_PARQUET_P1))
    if ruta.exists():
        return cargar_df_p1(str(ruta))

    # No hay parquet: se reconstruye con el pipeline (tarda) y se guarda para la próxima vez.
    try:
        df = cargar_df_p1()          # sin ruta, llama a construir_dataset_p1()
    except Exception as e:
        pytest.skip(f"No existe {ruta} y no se pudo construir la tabla con construir_dataset_p1(): {e}")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta)
    return df


@pytest.fixture(scope="module")
def splits_reales(df_real):
    train, test = dividir_train_test(df_real)
    if train.empty or test.empty:
        pytest.skip("Train o test vacío con las fechas de config.py.")
    return train, test


def test_real_llave_estacion_hora_unica(df_real):
    """Integridad: no debe haber dos filas para la misma estación y la misma hora."""
    col = columna_tiempo(df_real)
    ds = armar_dataset(df_real.dropna(subset=[P1_TARGET]))
    r = DataDuplicates(columns=[COL_ESTACION, col]).add_condition_ratio_less_or_equal(0).run(ds)
    assert r.passed_conditions()


def test_real_sin_nulos_en_features_ni_target(df_real):
    columnas = list(P1_FEATURES_V2) + [P1_TARGET]
    nulos = df_real[columnas].isna().sum()
    assert nulos.sum() == 0, f"Nulos por columna:\n{nulos[nulos > 0]}"


def test_real_tipos_numericos(df_real):
    columnas = list(P1_FEATURES_V2) + [P1_TARGET]
    no_numericas = [c for c in columnas if not pd.api.types.is_numeric_dtype(df_real[c])]
    assert not no_numericas, f"Columnas no numéricas: {no_numericas}"


def test_real_rangos_validos(df_real):
    assert (df_real[P1_TARGET] >= 0).all(), "Hay demanda objetivo negativa"
    for c in ["lag_1h", "lag_24h", "lag_168h"]:
        assert (df_real[c] >= 0).all(), f"{c} tiene valores negativos"
    assert df_real["target_hour_sin"].between(-1, 1).all()
    assert df_real["target_hour_cos"].between(-1, 1).all()


def test_real_target_no_constante(df_real):
    r = IsSingleValue().add_condition_not_single_value().run(armar_dataset(df_real.dropna(subset=[P1_TARGET])))
    assert r.passed_conditions()


def test_real_drift_de_features_entre_train_y_test(splits_reales):
    train, test = splits_reales
    r = FeatureDrift(columns=FEATURES_DRIFT).add_condition_drift_score_less_than(
        max_allowed_numeric_score=MAX_DRIFT_FEATURES).run(armar_dataset(train), armar_dataset(test))
    assert r.passed_conditions(), "Hay drift en alguna feature: revisar reports/deepchecks (posible estacionalidad)"


def test_real_drift_del_target_entre_train_y_test(splits_reales):
    train, test = splits_reales
    r = LabelDrift().add_condition_drift_score_less_than(
        max_allowed_drift_score=MAX_DRIFT_TARGET).run(armar_dataset(train), armar_dataset(test))
    assert r.passed_conditions()


def test_real_sin_fuga_temporal_entre_train_y_test(splits_reales):
    """Ninguna fecha de test debe caer dentro del periodo de train."""
    train, test = splits_reales
    r = DateTrainTestLeakageOverlap().add_condition_leakage_ratio_less_or_equal(0).run(
        armar_dataset(train), armar_dataset(test))
    assert r.passed_conditions()


def test_real_ninguna_feature_predice_el_target_por_si_sola(df_real):
    """PPS (poder predictivo): una feature que casi predice sola el target sugiere fuga de información."""
    ds = armar_dataset(df_real.dropna(subset=[P1_TARGET]))
    r = FeatureLabelCorrelation(random_state=42).add_condition_feature_pps_less_than(MAX_PPS).run(ds)
    assert r.passed_conditions()


# ======================= B) SINTÉTICOS: ¿los chequeos detectan fallas? =======================

def _sintetico(semilla=0, desplazamiento=0.0, estaciones=10, horas=100):
    rng = np.random.default_rng(semilla)
    n = estaciones * horas
    df = pd.DataFrame({
        COL_ESTACION: np.repeat(np.arange(1, estaciones + 1), horas),
        "hour_window": np.tile(pd.date_range("2023-01-01", periods=horas, freq="h"), estaciones),
        "lag_1h": rng.poisson(3 + desplazamiento, n), "lag_24h": rng.poisson(3 + desplazamiento, n),
        "lag_168h": rng.poisson(3 + desplazamiento, n),
        "rolling_3h": rng.uniform(0, 6, n) + desplazamiento, "rolling_168h": rng.uniform(0, 6, n) + desplazamiento,
        "target_hour_sin": rng.uniform(-1, 1, n), "target_hour_cos": rng.uniform(-1, 1, n),
        "target_is_weekend": rng.integers(0, 2, n),
    })
    df[P1_TARGET] = rng.poisson(df["rolling_3h"].clip(0.1) + 1)
    return df


def _duplicados_por_llave(df):
    ds = armar_dataset(df)
    return DataDuplicates(columns=[COL_ESTACION, "hour_window"]).add_condition_ratio_less_or_equal(0).run(ds)


def test_sintetico_sin_duplicados_en_datos_sanos():
    assert _duplicados_por_llave(_sintetico()).passed_conditions()


def test_sintetico_detecta_duplicados_inyectados():
    df = _sintetico()
    df = pd.concat([df, df.iloc[:50]], ignore_index=True)
    assert not _duplicados_por_llave(df).passed_conditions()


def test_sintetico_filas_con_mismas_features_no_son_duplicados_si_la_llave_difiere():
    """Caso típico de demanda baja: features idénticas en horas distintas NO es un duplicado real."""
    df = _sintetico()
    df[FEATURES_DRIFT[:3]] = 0                      # muchas filas con los mismos rezagos
    assert _duplicados_por_llave(df).passed_conditions()


def test_sintetico_sin_drift_con_misma_distribucion():
    r = FeatureDrift(columns=FEATURES_DRIFT).add_condition_drift_score_less_than(
        max_allowed_numeric_score=MAX_DRIFT_FEATURES).run(
        armar_dataset(_sintetico(semilla=1)), armar_dataset(_sintetico(semilla=2)))
    assert r.passed_conditions()


def test_sintetico_detecta_drift_inyectado():
    r = FeatureDrift(columns=FEATURES_DRIFT).add_condition_drift_score_less_than(
        max_allowed_numeric_score=MAX_DRIFT_FEATURES).run(
        armar_dataset(_sintetico(semilla=1)), armar_dataset(_sintetico(semilla=2, desplazamiento=3.0)))
    assert not r.passed_conditions()


def _fuga_temporal_pasa(train, test):
    r = DateTrainTestLeakageOverlap().add_condition_leakage_ratio_less_or_equal(0).run(
        armar_dataset(train), armar_dataset(test))
    return r.passed_conditions()


def test_sintetico_sin_fuga_temporal_si_test_es_posterior():
    train = _sintetico(semilla=1)
    test = _sintetico(semilla=2).assign(hour_window=lambda d: d["hour_window"] + pd.Timedelta(hours=1000))
    assert _fuga_temporal_pasa(train, test)


def test_sintetico_detecta_fuga_temporal_inyectada():
    """Train y test con las mismas fechas: el test queda dentro del periodo de train."""
    assert not _fuga_temporal_pasa(_sintetico(semilla=1), _sintetico(semilla=2))


def _pps_pasa(df):
    r = FeatureLabelCorrelation(random_state=42).add_condition_feature_pps_less_than(MAX_PPS).run(armar_dataset(df))
    return r.passed_conditions()


def test_sintetico_sin_feature_que_prediga_el_target_en_datos_sanos():
    assert _pps_pasa(_sintetico())


def test_sintetico_detecta_feature_que_es_copia_del_target():
    df = _sintetico()
    df["lag_1h"] = df[P1_TARGET]          # fuga de información inyectada
    assert not _pps_pasa(df)