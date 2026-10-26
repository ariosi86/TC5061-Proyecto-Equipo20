#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de entrenamiento, evaluación y registro en MLflow para P1 (Demanda Horaria).
Integra baselines, Regresión Lineal (v1, v2) y el candidato LightGBM Poisson.

Exporta predicciones en reports/p1_predicciones_modelos.parquet.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import typer
import mlflow
import itertools
import time
import lightgbm as lgb
from loguru import logger
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    P1_ALPHA_DEFAULT,
    P1_FEATURES_V1,
    P1_FEATURES_V2,
    P1_LR_DEFAULT,
    P1_MAX_ITER_DEFAULT,
    P1_TARGET,
    P1_TEST_FIN,
    P1_TEST_INICIO,
    P1_VAL_INICIO,
    REPORTS_DIR,
    TRIPS_TRAIN_PATH,
    URI_MLFLOW_TRACKING,
    RANDOM_SEED,
    P1_LGBM_GRID_POISSON,
    P1_LGBM_GRID_TWEEDIE,
)
from proyecto_mibici.features import construir_dataset_p1
from proyecto_mibici.modeling.model_p1 import BaselineIngenuoP1, DemandaLinealP1, DemandaPoissonP1
from proyecto_mibici.modeling.tracking import MLFlowTracker
from proyecto_mibici.plots import figuras_p1


app = typer.Typer(help="CLI modular para entrenar y registrar modelos de P1.")


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


def calcular_metricas_por_estacion(
    df: pd.DataFrame, pred: np.ndarray, target: str, df_stations: pd.DataFrame = None
) -> pd.DataFrame:
    """Calcula métricas desglosadas por estación para el artefacto en MLflow."""
    res = df[["origen_id", target]].copy()
    res["pred"] = np.asarray(pred)
    filas = []
    for est, g in res.groupby("origen_id"):
        y_r = g[target].to_numpy()
        y_p = g["pred"].to_numpy()
        suma_r = y_r.sum()
        wape = float(np.abs(y_r - y_p).sum() / suma_r) if suma_r > 0 else 0.0
        filas.append({
            "origen_id": est,
            "horas_evaluadas": len(g),
            "viajes_reales_total": float(suma_r),
            "viajes_predichos_total": float(y_p.sum()),
            "MAE": float(mean_absolute_error(y_r, y_p)),
            "RMSE": float(root_mean_squared_error(y_r, y_p)),
            "WAPE": wape,
            "Sesgo": float((y_p - y_r).mean()),
        })
    out = pd.DataFrame(filas)
    if df_stations is not None and "id" in df_stations.columns and "name" in df_stations.columns:
        out = (
            out.merge(df_stations[["id", "name"]], left_on="origen_id", right_on="id", how="left")
            .drop(columns="id")
            .rename(columns={"name": "nombre_estacion"})
        )
    out["error_abs_total"] = out["MAE"] * out["horas_evaluadas"]
    return out


def imprimir_metricas_modelo(nombre: str, m_train: dict, m_val: dict, m_test: dict):
    """Muestra en terminal las métricas del modelo evaluado en los 3 splits."""
    logger.info(
        f"[{nombre}] "
        f"Train -> MAE: {m_train['MAE']:.3f} | WAPE: {m_train['WAPE']:.3f} || "
        f"Val -> MAE: {m_val['MAE']:.3f} | WAPE: {m_val['WAPE']:.3f} || "
        f"Test -> MAE: {m_test['MAE']:.3f} | RMSE: {m_test['RMSE']:.3f} | WAPE: {m_test['WAPE']:.3f} | Sesgo: {m_test['Sesgo']:.3f}"
    )

#------------------------------------
# SIMPLFIFIED FUNCTIONS FOR MODELS
#------------------------------------
def registrar_fila_comparacion(filas_comparacion, nombre_modelo, m_train, m_val, m_test):
    """Agrega las métricas de un modelo a la lista de comparación general."""
    filas_comparacion.append({
        "modelo": nombre_modelo,
        "train_mae": m_train["MAE"],
        "train_wape": m_train["WAPE"],
        "val_mae": m_val["MAE"],
        "val_wape": m_val["WAPE"],
        "test_mae": m_test["MAE"],
        "test_rmse": m_test["RMSE"],
        "test_wape": m_test["WAPE"],
        "test_sesgo": m_test["Sesgo"],
    })



@app.command()
def main(
    model: str = typer.Option("baseline", "--model", "-m", help="Modelo a entrenar: 'baseline', 'poisson', 'lgbm_poisson', 'lgbm_tweedie,'lgbm_p_search', 'lgbm_t_search'"),
    learning_rate: float = typer.Option(P1_LR_DEFAULT, "--lr", help="Tasa de aprendizaje para Poisson o LightGBM."),
    alpha: float = typer.Option(P1_ALPHA_DEFAULT, "--alpha", help='Tasa de penalización para Poisson'),
    max_iter: int = typer.Option(P1_MAX_ITER_DEFAULT, "--max_iter", help="Max iterations para Poisson"),
):

    tracker = MLFlowTracker(
        tracking_uri=URI_MLFLOW_TRACKING,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )

    logger.info("Cargando y preparando dataset P1...")
    res = construir_dataset_p1()

    if isinstance(res, tuple):
        df_p1 = res[0].copy()
        df_stations = res[1].copy() if len(res) > 1 else None
    else:
        df_p1 = res.copy()
        df_stations = None

    # Identificar columna objetivo
    posibles_targets = [P1_TARGET, "target_next_hour", "demand_count", "demanda", "viajes"]
    target_col = next((col for col in posibles_targets if col in df_p1.columns), None)
    if target_col is None:
        raise KeyError(
            f"No se encontró la columna objetivo en el dataset. Columnas disponibles: {list(df_p1.columns)}"
        )
    logger.info(f"Columna objetivo detectada: '{target_col}'")

    col_tiempo = "target_hour" if "target_hour" in df_p1.columns else "hour_window"
    logger.info(f"Columna temporal usada para partición: '{col_tiempo}'")

    #------------------------------------------------
    # TEST / VAL / TRAIN SPLIT
    #------------------------------------------------

    # Partición temporal idéntica al notebook (< P1_TEST_FIN)
    df_con_target = df_p1.dropna(subset=[target_col]).copy()

    train_mask = df_con_target[col_tiempo] < pd.Timestamp(P1_VAL_INICIO)
    val_mask = (
        (df_con_target[col_tiempo] >= pd.Timestamp(P1_VAL_INICIO))
        & (df_con_target[col_tiempo] < pd.Timestamp(P1_TEST_INICIO))
    )
    test_mask = (
        (df_con_target[col_tiempo] >= pd.Timestamp(P1_TEST_INICIO))
        & (df_con_target[col_tiempo] < pd.Timestamp(P1_TEST_FIN))
    )

    df_train = df_con_target.loc[train_mask].copy()
    df_val = df_con_target.loc[val_mask].copy()
    df_test = df_con_target.loc[test_mask].copy()

    if len(df_test) == 0:
        logger.warning("Filtro de fechas de test no arrojó registros. Usando el 20% final para Test.")
        corte = int(len(df_con_target) * 0.8)
        df_train = df_con_target.iloc[:corte].copy()
        df_test = df_con_target.iloc[corte:].copy()
        df_val = df_train.tail(int(len(df_train) * 0.1)).copy()

    if "target_hour" not in df_test.columns:
        df_test["target_hour"] = df_test[col_tiempo]

    splits = {
        "Train": df_train,
        "Validación": df_val,
        "Test": df_test,
    }

    y_train = df_train[target_col]
    y_val = df_val[target_col]
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
    cols_identificadores = [c for c in ["origen_id", col_tiempo] if c in df_test.columns]
    df_predicciones = df_test[cols_identificadores].copy()
    df_predicciones["real"] = y_test.values

    #--------------------------------
    # ENTRENAR LGBM
    #-------------------------------

    def entrenar_lgbm_candidato(hp: dict, nombre: str = "lgbm", extra_tags: dict = None, nested: bool = False):
        logger.info(f"--- Entrenando Candidato: LightGBM {hp['objective']} ({nombre}) {hp} ---")
        modelo = lgb.LGBMRegressor(random_state=RANDOM_SEED, **hp)
        modelo.fit(df_train[P1_FEATURES_V2], y_train)

        pred_train = np.clip(modelo.predict(df_train[P1_FEATURES_V2]), 0, None)
        pred_val = np.clip(modelo.predict(df_val[P1_FEATURES_V2]), 0, None)
        pred_test = np.clip(modelo.predict(df_test[P1_FEATURES_V2]), 0, None)
        df_predicciones[nombre] = np.round(pred_test, 2)

        m_train = evaluar_metricas_regresion(y_train, pred_train, "Train")
        m_val = evaluar_metricas_regresion(y_val, pred_val, "Validación")
        m_test = evaluar_metricas_regresion(y_test, pred_test, "Test")
        imprimir_metricas_modelo(nombre, m_train, m_val, m_test)

        tracker.registrar_run(
            run_name=f"P1_{nombre}",
            params={**params_base,
            "modelo": f"LightGBM_{hp['objective']}",
            "variables": ", ".join(P1_FEATURES_V2),
            **hp},
            tags={**tags_base, "version": nombre, "stage_of_work": "candidato",
                "model_family": "lightgbm", **(extra_tags or {})},
            tabla=pd.DataFrame([m_train, m_val, m_test]),
            tabla_est=calcular_metricas_por_estacion(df_test, pred_test, target_col, df_stations),
            modelo=modelo,
            X_ejemplo=df_train[P1_FEATURES_V2].head(5),
            figuras=figuras_p1(df_test, pred_test, target_col, nombre),
            serialization_format="cloudpickle",
            df_train=df_train,
            ruta_entrenamiento=str(TRIPS_TRAIN_PATH),
            nested=nested,
        )
        registrar_fila_comparacion(filas_comparacion, nombre, m_train, m_val, m_test)

    # ----------------------------------------------------------------
    # 1. MODELO BASELINE DEFINIDO (v2): REGRESIÓN LINEAL CON ROLLING
    # ----------------------------------------------------------------

    if model == "baseline":
        logger.info("--- Entrenando Modelo v2: Regresión Lineal + Medias Móviles ---")
        modelo_v2 = DemandaLinealP1(features=P1_FEATURES_V2)
        modelo_v2.fit(df_train[P1_FEATURES_V2], y_train)

        ruta_modelo_v2 = MODELS_DIR / "p1_v2_model.pkl"
        modelo_v2.save(ruta_modelo_v2)
        modelo_v2.save(MODELS_DIR / "p1_linear_model.pkl")
        logger.info(f"Modelo baseline guardado en {ruta_modelo_v2}")

        pred_v2_train = modelo_v2.predict(df_train[P1_FEATURES_V2])
        pred_v2_val = modelo_v2.predict(df_val[P1_FEATURES_V2])
        pred_v2_test = modelo_v2.predict(df_test[P1_FEATURES_V2])
        df_predicciones["Baseline_lineal_rolling"] = np.round(pred_v2_test, 2)

        m_v2_train = evaluar_metricas_regresion(y_train, pred_v2_train, "Train")
        m_v2_val = evaluar_metricas_regresion(y_val, pred_v2_val, "Validación")
        m_v2_test = evaluar_metricas_regresion(y_test, pred_v2_test, "Test")

        imprimir_metricas_modelo("Baseline_lineal_rolling", m_v2_train, m_v2_val, m_v2_test)

        tabla_est_v2 = calcular_metricas_por_estacion(df_test, pred_v2_test, target_col, df_stations)
        figuras_v2 = figuras_p1(df_test, pred_v2_test, target_col, "v2")

        tracker.registrar_run(
            run_name="P1_v2_rolling_mean",
            params={
                **params_base,
                "modelo": "LinearRegression",
                "variables_predictoras": ", ".join(P1_FEATURES_V2),
                "postprocesamiento": "np.clip(predicciones, 0, None)",
            },
            tags={**tags_base, "version": "Baseline", "stage_of_work": "baseline", "model_family": "linear_regression", "baseline": "true"},
            tabla=pd.DataFrame([m_v2_train, m_v2_val, m_v2_test]),
            tabla_est=tabla_est_v2,
            modelo=modelo_v2,
            X_ejemplo=df_train[P1_FEATURES_V2].head(5),
            figuras=figuras_v2,
            serialization_format="cloudpickle",
    )

        registrar_fila_comparacion(filas_comparacion, "baseline", m_v2_train, m_v2_val, m_v2_test)

    # -------------------------------------------------------------
    # POISSON 
    # -------------------------------------------------------------

    elif model == "poisson":

        logger.info(f"--- Entrenando Candidato: Regresión de Poisson (lr={learning_rate}, pen={alpha}, max_iter={max_iter}) ---",)
        poisson_model = DemandaPoissonP1(features=P1_FEATURES_V2, alpha=alpha, max_iter=max_iter)
 
        poisson_model.fit(df_train[P1_FEATURES_V2], y_train)
        
        ruta_modelo_poisson = MODELS_DIR / "p1_poisson_model.pkl"
    
        pred_poisson_train = poisson_model.predict(df_train[P1_FEATURES_V2])
        pred_poisson_val = poisson_model.predict(df_val[P1_FEATURES_V2])
        pred_poisson_test = poisson_model.predict(df_test[P1_FEATURES_V2])
        df_predicciones["poisson"] = np.round(pred_poisson_test, 2)

        m_poisson_train = evaluar_metricas_regresion(y_train, pred_poisson_train, "Train")
        m_poisson_val = evaluar_metricas_regresion(y_val, pred_poisson_val, "Validación")
        m_poisson_test = evaluar_metricas_regresion(y_test, pred_poisson_test, "Test")

        imprimir_metricas_modelo("poisson", m_poisson_train, m_poisson_val, m_poisson_test)

        tabla_est_poisson = calcular_metricas_por_estacion(df_test, pred_poisson_test, target_col, df_stations)
        figuras_poisson = figuras_p1(df_test, pred_poisson_test, target_col, "poisson")

        tracker.registrar_run(
            run_name="P1_poisson",
            params={
                **params_base,
                "modelo": "PoissonRegression",
                "variables_predictoras": ", ".join(P1_FEATURES_V2),
                "postprocesamiento": "np.clip(predicciones, 0, None)",
                "alpha": alpha,
                "max_iter":max_iter,
            },
            tags={**tags_base, "version": "poisson", "stage_of_work": "candidato", "model_family": "poisson_regression"},
            tabla=pd.DataFrame([m_poisson_train, m_poisson_val, m_poisson_test]),
            tabla_est=tabla_est_poisson,
            modelo=poisson_model,
            X_ejemplo=df_train[P1_FEATURES_V2].head(5),
            figuras=figuras_poisson,
            serialization_format="cloudpickle",
    )

        registrar_fila_comparacion(filas_comparacion, "poisson", m_poisson_train, m_poisson_val, m_poisson_test)

    # -------------------------------------------------------------
    # LIGHT GBM - POISSON (NESTED SEARCH)
    # -------------------------------------------------------------
    elif model == "lgbm_p_search":
        combinaciones = [dict(zip(P1_LGBM_GRID_POISSON, v)) for v in itertools.product(*P1_LGBM_GRID_POISSON.values())]
        mejor_hp, mejor_mae = None, float("inf")

        with mlflow.start_run(run_name="P1_lgbm_poisson_search") as padre:
            mlflow.set_tags({**tags_base,"stage_of_work": "search", "model_family": "lightgbm"})
            mlflow.log_params({"n_combinaciones": len(combinaciones), "criterio": "validation_mae", "objetivo": "poisson"})

            for i, hp in enumerate(combinaciones):
                source_tags = tracker._get_source_tags()
                with mlflow.start_run(run_name=f"search_{i:02d}", nested=True):
                    if source_tags:
                        mlflow.set_tags(source_tags)
                    mlflow.log_params(hp)
                    logger.info(f"search_{i:02d}:{hp}")
                    try:
                        t0 = time.time()
                        m = lgb.LGBMRegressor(random_state=RANDOM_SEED, **hp)
                        m.fit(df_train[P1_FEATURES_V2], y_train)
                        segundos = time.time() - t0

                        pred_val = np.clip(m.predict(df_val[P1_FEATURES_V2]), 0, None)
                        m_val = evaluar_metricas_regresion(y_val, pred_val, "Validación")

                        mlflow.set_tags({**tags_base, "stage_of_work": "search", "model_family": "lightgbm"})
                    
                        mlflow.log_metrics({**tracker.metricas_a_dict(pd.DataFrame([m_val])),
                                        "train_seconds": segundos})
                        if m_val["MAE"] < mejor_mae:
                            mejor_mae, mejor_hp = m_val["MAE"], hp
                    except Exception as e:
                        mlflow.set_tag("fallo", "true")
                        mlflow.set_tag("error", str(e)[:250])
                        logger.warning(f"search_{i:02d} falló y se omite: {e}")
                        continue


            mlflow.log_metric("best_validation_mae", mejor_mae)
            mlflow.log_params({f"best_{k}": v for k, v in mejor_hp.items()})
            id_padre = padre.info.run_id

        # Registrar el modelo final anidado bajo el padre de búsqueda
        entrenar_lgbm_candidato(mejor_hp, nombre="lgbm_poisson_tuned",
                                extra_tags={"search_run_id": id_padre}, nested=True)
    # -------------------------------------------------------------
    # POISSON CANDIDATO
    # -------------------------------------------------------------      
    elif model == "lgbm_poisson":
            entrenar_lgbm_candidato({"objective": "poisson","learning_rate": 0.1, "num_leaves": 127, "n_estimators": 600}, nombre="lgbm_poisson")


    # -------------------------------------------------------------
    # LIGHT GBM - TWEEDIE (NESTED SEARCH)
    # -------------------------------------------------------------
    elif model == "lgbm_t_search":
        combinaciones = [dict(zip(P1_LGBM_GRID_TWEEDIE, v)) for v in itertools.product(*P1_LGBM_GRID_TWEEDIE.values())]
        mejor_hp, mejor_mae = None, float("inf")

        with mlflow.start_run(run_name="P1_lgbm_tweedie_search") as padre:
            mlflow.set_tags({**tags_base, "stage_of_work": "search", "model_family": "lightgbm"})
            mlflow.log_params({"n_combinaciones": len(combinaciones), "criterio": "validation_mae", "objetivo": "tweedie"})

            for i, hp in enumerate(combinaciones):
                source_tags = tracker._get_source_tags()
                with mlflow.start_run(run_name=f"search_{i:02d}", nested=True):
                    if source_tags:
                        mlflow.set_tags(source_tags)
                    mlflow.log_params(hp)
                    logger.info(f"search_{i:02d}:{hp}")
                    try:
                        t0 = time.time()
                        m = lgb.LGBMRegressor(random_state=RANDOM_SEED, **hp)
                        m.fit(df_train[P1_FEATURES_V2], y_train)
                        segundos = time.time() - t0

                        pred_val = np.clip(m.predict(df_val[P1_FEATURES_V2]), 0, None)
                        m_val = evaluar_metricas_regresion(y_val, pred_val, "Validación")

                        mlflow.set_tags({**tags_base, "stage_of_work": "search", "model_family": "lightgbm"})
                    
                        mlflow.log_metrics({**tracker.metricas_a_dict(pd.DataFrame([m_val])),
                                        "train_seconds": segundos})
                        if m_val["MAE"] < mejor_mae:
                            mejor_mae, mejor_hp = m_val["MAE"], hp
                    except Exception as e:
                        mlflow.set_tag("fallo", "true")
                        mlflow.set_tag("error", str(e)[:250])
                        logger.warning(f"search_{i:02d} falló y se omite: {e}")
                        continue


            mlflow.log_metric("best_validation_mae", mejor_mae)
            mlflow.log_params({f"best_{k}": v for k, v in mejor_hp.items()})
            id_padre = padre.info.run_id

        # Registrar el modelo final anidado bajo el padre de búsqueda
        entrenar_lgbm_candidato(mejor_hp, nombre="lgbm_tweedie_tuned",
                                extra_tags={"search_run_id": id_padre}, nested=True)
    # -------------------------------------------------------------
    # TWEEDIE CANDIDATO
    # -------------------------------------------------------------
    elif model == "lgbm_tweedie":
            entrenar_lgbm_candidato({"objective": "tweedie",
                         "tweedie_variance_power": 1.8,
                         "learning_rate": 0.1, "num_leaves": 127, "n_estimators": 600},
                        nombre="lgbm_tweedie")
    else:
        raise ValueError(f"Modelo desconocido: '{model}'. Revisa el parámetro --model.")

   # -------------------------------------------------------------
    # 4. GUARDAR COMPARATIVA Y PREDICCIONES (SOLO PARQUET)
    # -------------------------------------------------------------
    df_p1_comp = pd.DataFrame(filas_comparacion).set_index("modelo")

    logger.info(f"Resumen de la corrida:")
    logger.info("\n" + df_p1_comp.round(3).to_string())

    preds_out_path = REPORTS_DIR / f"p1_predicciones_{model}.parquet"
    df_predicciones.to_parquet(preds_out_path, index=False)
    logger.success(f"Predicciones de los modelos guardadas en {preds_out_path}")

if __name__ == "__main__":
    app()