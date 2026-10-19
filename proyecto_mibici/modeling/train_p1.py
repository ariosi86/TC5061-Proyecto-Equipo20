#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de entrenamiento, evaluación y registro en MLflow para P1 (Demanda Horaria).
Integra, evalúa e imprime en consola los 6 modelos:
  - 4 baselines ingenuos (persistencia, lag 24h, lag 168h, promedio 24h/168h)
  - v1 Regresión Lineal (lags + componentes temporales)
  - v2 Regresión Lineal (v1 + medias móviles rolling_3h y rolling_168h)
Exporta métricas en reports/p1_comparacion.csv y predicciones en reports/p1_predicciones_modelos.parquet.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    P1_FEATURES_V1,
    P1_FEATURES_V2,
    P1_TARGET,
    P1_TEST_FIN,
    P1_TEST_INICIO,
    P1_VAL_INICIO,
    REPORTS_DIR,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.features import construir_dataset_p1
from proyecto_mibici.modeling.model_p1 import BaselineIngenuoP1, DemandaLinealP1
from proyecto_mibici.modeling.tracking import MLFlowTracker
from proyecto_mibici.plots import figuras_p1

app = typer.Typer()


def evaluar_metricas_regresion(y_real, y_pred, conjunto_nombre: str) -> dict:
    """Calcula MAE, RMSE, WAPE y Sesgo para series de tiempo."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    suma_real = y_real.sum()
    wape = float(np.abs(y_real - y_pred).sum() / suma_real) if suma_real > 0 else 0.0

    return {
        "conjunto": conjunto_nombre,
        "MAE": float(mean_absolute_error(y_real, y_pred)),
        "RMSE": float(root_mean_squared_error(y_real, y_pred)),
        "WAPE": wape,
        "Sesgo": float((y_pred - y_real).mean()),
    }


def imprimir_metricas_modelo(nombre: str, m_train: dict, m_test: dict):
    """Muestra en terminal las métricas del modelo evaluado."""
    logger.info(
        f"[{nombre}] "
        f"Train -> MAE: {m_train['MAE']:.3f} | WAPE: {m_train['WAPE']:.3f} || "
        f"Test -> MAE: {m_test['MAE']:.3f} | RMSE: {m_test['RMSE']:.3f} | WAPE: {m_test['WAPE']:.3f} | Sesgo: {m_test['Sesgo']:.3f}"
    )


@app.command()
def main():
    tracker = MLFlowTracker(
        tracking_uri=URI_MLFLOW_TRACKING,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )

    logger.info("Cargando y preparando dataset P1...")
    res = construir_dataset_p1()

    # 1. Recuperar el DataFrame principal
    if isinstance(res, tuple):
        df_p1 = res[0].copy()
    else:
        df_p1 = res.copy()

    # 2. Identificar la columna objetivo real de forma dinámica y defensiva
    posibles_targets = [P1_TARGET, "target_next_hour", "demand_count", "demanda", "viajes"]
    target_col = None
    for col in posibles_targets:
        if col in df_p1.columns:
            target_col = col
            break

    if target_col is None:
        raise KeyError(
            f"No se encontró la columna objetivo en el dataset. Columnas disponibles: {list(df_p1.columns)}"
        )
    logger.info(f"Columna objetivo detectada: '{target_col}'")

    # 3. Identificar columna temporal de ordenamiento
    col_tiempo = "target_hour" if "target_hour" in df_p1.columns else "hour_window"
    logger.info(f"Columna temporal usada para partición: '{col_tiempo}'")

    # 4. Partición cronológica directamente sobre df_p1
    df_con_target = df_p1.dropna(subset=[target_col]).copy()

    train_mask = df_con_target[col_tiempo] < pd.Timestamp(P1_VAL_INICIO)
    val_mask = (
        (df_con_target[col_tiempo] >= pd.Timestamp(P1_VAL_INICIO))
        & (df_con_target[col_tiempo] < pd.Timestamp(P1_TEST_INICIO))
    )
    test_mask = (
        (df_con_target[col_tiempo] >= pd.Timestamp(P1_TEST_INICIO))
        & (df_con_target[col_tiempo] <= pd.Timestamp(P1_TEST_FIN))
    )

    df_train = df_con_target.loc[train_mask].copy()
    df_val = df_con_target.loc[val_mask].copy()
    df_test = df_con_target.loc[test_mask].copy()

    if len(df_test) == 0:
        logger.warning("Filtro de fechas de test no arrojó registros. Usando el 20% final de la serie para Test.")
        corte = int(len(df_con_target) * 0.8)
        df_train = df_con_target.iloc[:corte].copy()
        df_test = df_con_target.iloc[corte:].copy()
        df_val = df_train.tail(int(len(df_train) * 0.1)).copy()

    # Asegurar que df_test cuente con 'target_hour' para el generador de figuras
    if "target_hour" not in df_test.columns:
        df_test["target_hour"] = df_test[col_tiempo]

    splits = {
        "Train": df_train,
        "Validación": df_val,
        "Test": df_test,
    }

    y_train = df_train[target_col]
    y_test = df_test[target_col]

    logger.info(
        f"Registros listos -> Train: {len(df_train):,} | Validación: {len(df_val):,} | Test: {len(df_test):,}"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    params_base, tags_base = tracker.armar_params_tags(
        "P1",
        splits,
        extra_params={"variable_objetivo": target_col, "frecuencia": "horaria"},
        col_tiempo=col_tiempo,
    )

    filas_comparacion = []

    # Consolidado de predicciones sobre el conjunto Test
    cols_identificadores = [c for c in ["origen_id", col_tiempo] if c in df_test.columns]
    df_predicciones = df_test[cols_identificadores].copy()
    df_predicciones["real"] = y_test.values

    # -------------------------------------------------------------
    # 1. EVALUACIÓN Y PREDICCIÓN DE 4 MODELOS INGENUOS (BASELINES)
    # -------------------------------------------------------------
    logger.info("--- Evaluando Modelos Ingenuos (Baselines) ---")
    estrategias_ingenuas = [
        ("naive_persistencia", "persistencia"),
        ("naive_lag_24h", "lag_24h"),
        ("naive_lag_168h", "lag_168h"),
        ("naive_promedio_24h_168h", "promedio_24h_168h"),
    ]

    for nombre_run, estrategia in estrategias_ingenuas:
        modelo_ingenuo = BaselineIngenuoP1(estrategia=estrategia)
        pred_train = modelo_ingenuo.predict(df_train)
        pred_test = modelo_ingenuo.predict(df_test)

        df_predicciones[nombre_run] = np.round(pred_test, 2)

        m_train = evaluar_metricas_regresion(y_train, pred_train, "Train")
        m_test = evaluar_metricas_regresion(y_test, pred_test, "Test")

        imprimir_metricas_modelo(nombre_run, m_train, m_test)

        tabla_metricas = pd.DataFrame([m_train, m_test])
        tracker.registrar_run(
            run_name=f"P1_{nombre_run}",
            params={**params_base, "modelo": nombre_run, "estrategia": estrategia},
            tags={**tags_base, "version": nombre_run},
            tabla=tabla_metricas,
            tabla_est=pd.DataFrame(),
        )

        filas_comparacion.append({
            "modelo": nombre_run,
            "train_mae": m_train["MAE"],
            "train_wape": m_train["WAPE"],
            "test_mae": m_test["MAE"],
            "test_rmse": m_test["RMSE"],
            "test_wape": m_test["WAPE"],
            "test_sesgo": m_test["Sesgo"],
        })

    # -------------------------------------------------------------
    # 2. MODELO V1: REGRESIÓN LINEAL (SIN MEDIAS MÓVILES)
    # -------------------------------------------------------------
    logger.info("--- Entrenando Modelo v1: Regresión Lineal Base ---")
    modelo_v1 = DemandaLinealP1()
    modelo_v1.fit(df_train[P1_FEATURES_V1], y_train)

    pred_v1_train = modelo_v1.predict(df_train)
    pred_v1_test = modelo_v1.predict(df_test)
    df_predicciones["v1_lineal"] = np.round(pred_v1_test, 2)

    m_v1_train = evaluar_metricas_regresion(y_train, pred_v1_train, "Train")
    m_v1_test = evaluar_metricas_regresion(y_test, pred_v1_test, "Test")

    imprimir_metricas_modelo("v1_lineal", m_v1_train, m_v1_test)

    tracker.registrar_run(
        run_name="P1_v1_linear",
        params={**params_base, "modelo": "LinearRegression_v1", "variables": ", ".join(P1_FEATURES_V1)},
        tags={**tags_base, "version": "v1_linear"},
        tabla=pd.DataFrame([m_v1_train, m_v1_test]),
        tabla_est=pd.DataFrame(),
        modelo=modelo_v1.pipeline,
        X_ejemplo=df_train[P1_FEATURES_V1].head(5),
    )

    filas_comparacion.append({
        "modelo": "v1_lineal",
        "train_mae": m_v1_train["MAE"],
        "train_wape": m_v1_train["WAPE"],
        "test_mae": m_v1_test["MAE"],
        "test_rmse": m_v1_test["RMSE"],
        "test_wape": m_v1_test["WAPE"],
        "test_sesgo": m_v1_test["Sesgo"],
    })

    # -------------------------------------------------------------
    # 3. MODELO V2: REGRESIÓN LINEAL CON ROLLING (CAMPEÓN)
    # -------------------------------------------------------------
    logger.info("--- Entrenando Modelo v2: Regresión Lineal + Medias Móviles ---")
    modelo_v2 = DemandaLinealP1()
    modelo_v2.fit(df_train[P1_FEATURES_V2], y_train)

    ruta_modelo = MODELS_DIR / "p1_linear_model.pkl"
    modelo_v2.save(ruta_modelo)
    logger.info(f"Pipeline campeón P1 persistido en {ruta_modelo}")

    pred_v2_train = modelo_v2.predict(df_train)
    pred_v2_test = modelo_v2.predict(df_test)
    df_predicciones["v2_lineal_rolling"] = np.round(pred_v2_test, 2)

    m_v2_train = evaluar_metricas_regresion(y_train, pred_v2_train, "Train")
    m_v2_test = evaluar_metricas_regresion(y_test, pred_v2_test, "Test")

    imprimir_metricas_modelo("v2_lineal_rolling", m_v2_train, m_v2_test)

    # Llamada corregida con argumentos directos: df_test, pred, target (str), etiqueta (str)
    figuras = figuras_p1(df_test, pred_v2_test, target_col, "v2_lineal_rolling")

    tracker.registrar_run(
        run_name="P1_v2_lineal_rolling",
        params={**params_base, "modelo": "LinearRegression_v2", "variables": ", ".join(P1_FEATURES_V2)},
        tags={**tags_base, "version": "v2_lineal_rolling", "baseline": "true"},
        tabla=pd.DataFrame([m_v2_train, m_v2_test]),
        tabla_est=pd.DataFrame(),
        modelo=modelo_v2.pipeline,
        X_ejemplo=df_train[P1_FEATURES_V2].head(5),
        figuras=figuras,
    )

    filas_comparacion.append({
        "modelo": "v2_lineal_rolling",
        "train_mae": m_v2_train["MAE"],
        "train_wape": m_v2_train["WAPE"],
        "test_mae": m_v2_test["MAE"],
        "test_rmse": m_v2_test["RMSE"],
        "test_wape": m_v2_test["WAPE"],
        "test_sesgo": m_v2_test["Sesgo"],
    })

    # -------------------------------------------------------------
    # 4. GUARDAR Y MOSTRAR COMPARATIVA DE MÉTRICAS
    # -------------------------------------------------------------
    df_p1_comp = pd.DataFrame(filas_comparacion).set_index("modelo")
    p1_comp_path = REPORTS_DIR / "p1_comparacion.csv"
    df_p1_comp.round(3).to_csv(p1_comp_path)

    logger.info(f"Reporte de entrenamiento P1 guardado en {p1_comp_path}:")
    logger.info("\n" + df_p1_comp.round(3).to_string())

    # -------------------------------------------------------------
    # 5. GUARDAR Y MOSTRAR PREDICCIONES CONSOLIDADAS
    # -------------------------------------------------------------
    preds_out_path = REPORTS_DIR / "p1_predicciones_modelos.parquet"
    df_predicciones.to_parquet(preds_out_path, index=False)
    logger.success(f"Predicciones de los 6 modelos guardadas en {preds_out_path}")

    cols_mostrar = [
        c for c in [
            "origen_id", col_tiempo, "real",
            "naive_persistencia", "naive_promedio_24h_168h",
            "v1_lineal", "v2_lineal_rolling"
        ] if c in df_predicciones.columns
    ]
    logger.info(
        "Comparativa de predicciones (muestra de 10 registros en Test):\n" +
        df_predicciones[cols_mostrar].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    app()