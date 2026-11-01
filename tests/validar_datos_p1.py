"""
Validación de datos de P1 (demanda horaria) con DeepChecks.

Revisa la TABLA DE MODELADO (la salida de construir_dataset_p1), partida igual que train_p1:
  train = antes de P1_VAL_INICIO   |   test = [P1_TEST_INICIO, P1_TEST_FIN)

Uso (desde la raíz del repo):
    python tests/validar_datos_p1.py
    python tests/validar_datos_p1.py --parquet data/interim/p1_modelado.parquet

Genera en reports/deepchecks/: p1_data_integrity.html, p1_train_test_validation.html
y p1_model_evaluation.html (este último solo si existe models/p1_champion.pkl).
Los HTML pesan decenas de MB: no se suben al repo, solo las capturas.
"""
import os
import sys
import types
from pathlib import Path
from typing import Dict, Optional, Tuple
import packaging.version
import pandas as pd
import numpy as np

import joblib

from loguru import logger

from proyecto_mibici.config import MODELS_DIR, P1_FEATURES_V2, P1_TARGET

from proyecto_mibici.config import (
    P1_FEATURES_V2,
    P1_TARGET,
    P1_TEST_FIN,
    P1_TEST_INICIO,
    P1_VAL_INICIO,
)

SALIDA = Path("reports/deepchecks")
COL_ESTACION = "origen_id"
CAT_FEATURES = [COL_ESTACION, "target_is_weekend"]

# Parche para simular pkg_resources en Python 3.14 para DeepChecks
if "pkg_resources" not in sys.modules:
    dummy_pkg = types.ModuleType("pkg_resources")
    dummy_pkg.parse_version = packaging.version.parse
    sys.modules["pkg_resources"] = dummy_pkg

# Parche para max_error en sklearn.metrics._scorer para DeepChecks
import sklearn.metrics._scorer
if "max_error" not in sklearn.metrics._scorer._SCORERS:
    from sklearn.metrics import max_error, make_scorer
    sklearn.metrics._scorer._SCORERS["max_error"] = make_scorer(max_error, score_func_has_args=False)

# Parche para np.Inf y np.NaN, eliminados en NumPy 2.0, que DeepChecks todavía usa
for _nombre, _valor in (("Inf", np.inf), ("NaN", np.nan)):
    if not hasattr(np, _nombre):
        setattr(np, _nombre, _valor)

# Parche de compatibilidad para __sklearn_tags__ en Scikit-Learn >= 1.6 y DeepChecks
import sklearn.base

if not hasattr(sklearn.base.BaseEstimator, "__sklearn_tags__"):
    def __sklearn_tags__(self):
        from sklearn.utils._tags import Tags
        return Tags()
    sklearn.base.BaseEstimator.__sklearn_tags__ = __sklearn_tags__

from deepchecks.tabular import Dataset

def cargar_df_p1(ruta_parquet: Optional[str] = None) -> pd.DataFrame:
    """Lee la tabla de modelado de un parquet; si no se da, la reconstruye con el pipeline."""
    ruta_parquet = ruta_parquet or os.environ.get("P1_DATASET_PARQUET")
    if ruta_parquet:
        return pd.read_parquet(ruta_parquet)
    from proyecto_mibici.features_p1 import construir_dataset_p1
    return construir_dataset_p1()[0]

def columna_tiempo(df: pd.DataFrame) -> str:
    """Misma regla que train_p1: se particiona por target_hour si existe."""
    return "target_hour" if "target_hour" in df.columns else "hour_window"


def dividir_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    col = columna_tiempo(df)
    df = df.dropna(subset=[P1_TARGET])
    train = df[df[col] < pd.Timestamp(P1_VAL_INICIO)].copy()
    test = df[(df[col] >= pd.Timestamp(P1_TEST_INICIO)) & (df[col] < pd.Timestamp(P1_TEST_FIN))].copy()
    return train, test


def armar_dataset(df: pd.DataFrame):
    """Dataset de DeepChecks con la llave (estación + hora), las features y el target."""
    from importlib import import_module

    try:
        Dataset = import_module("deepchecks.tabular").Dataset
    except ModuleNotFoundError as exc:
        if exc.name == "deepchecks" or (exc.name and exc.name.startswith("deepchecks.")):
            raise ImportError("Instala DeepChecks con `pip install deepchecks`. ") from exc
        raise
    
    col = columna_tiempo(df)
    cols = [COL_ESTACION, col] + list(P1_FEATURES_V2) + [P1_TARGET]
    cat_cols_validas = [c for c in CAT_FEATURES if c in P1_FEATURES_V2]
    
    return Dataset(
        df[cols], 
        label=P1_TARGET, 
        features=list(P1_FEATURES_V2),
        cat_features=cat_cols_validas if cat_cols_validas else None,
        datetime_name=col
    )

def imprimir_resumen_suite(res, titulo: str = ""):
    """Imprime en consola el desglose detallado de fallos y no ejecutados para una suite."""
    if titulo:
        print(f"\n==================================================")
        print(f" SUITE: {titulo.upper()}")
        print(f"==================================================")
    
    pasaron = len(res.get_passed_checks())
    no_pasaron = len(res.get_not_passed_checks())
    no_corrieron = len(res.get_not_ran_checks())
    
    print(f"Pasaron: {pasaron} | No pasaron: {no_pasaron} | No corrieron: {no_corrieron}\n")
    
    if no_pasaron > 0:
        print("--- Chequeos con Fallos/Advertencias ---")
        for r in res.get_not_passed_checks():
            for c in r.conditions_results:
                if c.category.name != "PASS":
                    print(f"  [{c.category.name}] {r.get_header()} -> {c.name}: {c.details}")
                    
    if no_corrieron > 0:
        print("\n--- Chequeos No Ejecutados ---")
        for r in res.get_not_ran_checks():
            print(f"  [NO CORRIÓ] {r.get_header()}")

    

def correr_chequeos(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, object]:
    """Ejecuta las suites de DeepChecks y retorna un diccionario con los resultados."""
    from importlib import import_module

    try:
        suites = import_module("deepchecks.tabular.suites")
    except ModuleNotFoundError as exc:
        if exc.name == "deepchecks" or (exc.name and exc.name.startswith("deepchecks.")):
            raise ImportError("Instala DeepChecks con `pip install deepchecks`.") from exc
        raise

    SALIDA.mkdir(parents=True, exist_ok=True)
    ds_train, ds_test = armar_dataset(df_train), armar_dataset(df_test)
    resultados = {}

    # 1. Integridad de datos
    logger.info("Ejecutando suite: Integridad de datos...")
    r_integridad = suites.data_integrity().run(ds_train)
    r_integridad.save_as_html(str(SALIDA / "p1_data_integrity.html"), as_widget=False)
    resultados["Integridad de Datos"] = r_integridad

    # 2. Validación Train / Test
    logger.info("Ejecutando suite: Train/Test Drift & Validation...")
    r_drift = suites.train_test_validation().run(ds_train, ds_test)
    r_drift.save_as_html(str(SALIDA / "p1_train_test_validation.html"), as_widget=False)
    resultados["Train/Test Validation"] = r_drift

    # 3. Evaluación del modelo Champion (si existe el artefacto .pkl)
    ruta_modelo = MODELS_DIR / "p1_champion.pkl"
    try:
        if ruta_modelo.exists():
            logger.info(f"Cargando modelo champion desde {ruta_modelo}...")
            modelo_champion = joblib.load(ruta_modelo)

            r_modelo = suites.model_evaluation().run(
                train_dataset=ds_train,
                test_dataset=ds_test,
                model=modelo_champion
            )
            r_modelo.save_as_html(str(SALIDA / "p1_model_evaluation.html"), as_widget=False)
            resultados["Evaluación del Modelo"] = r_modelo
        else:
            logger.warning("No se encontró p1_champion.pkl. Se omite la suite model_evaluation.")
    except Exception as e:
        logger.error(f"Error durante la evaluación del modelo en DeepChecks: {e}")

    return resultados

def main(parquet: Optional[str] = None):
    df = cargar_df_p1(parquet)
    train, test = dividir_train_test(df)
    print(f"Filas -> train: {len(train):,} | test: {len(test):,}")
    
    if train.empty or test.empty:
        raise SystemExit("Train o test quedó vacío: revisa P1_VAL_INICIO / P1_TEST_INICIO / P1_TEST_FIN.")

    pct_target_zero = (test["target_next_hour"] == 0).mean() * 100
    pct_lags_zero = (test[["lag_1h", "lag_24h", "lag_168h"]] == 0).all(axis=1).mean() * 100

    print(f"\n--- DIAGNÓSTICO DE TEST ---")
    print(f" Demanda en 0 (% target_next_hour): {pct_target_zero:.2f}%")
    print(f" Todos los rezagos en 0 (lag_1h, 24h, 168h): {pct_lags_zero:.2f}%")
    print("---------------------------\n")
    
    # Correr suites y obtener diccionario
    resultados_dict = correr_chequeos(train, test)

    # Imprimir resumen de cada suite
    for nombre_suite, res in resultados_dict.items():
        imprimir_resumen_suite(res, titulo=nombre_suite)

    print(f"\nReportes HTML guardados exitosamente en `{SALIDA}/`\n")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", help="Parquet con la tabla de modelado de P1")
    main(ap.parse_args().parquet)
