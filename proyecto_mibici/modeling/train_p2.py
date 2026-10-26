#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de entrenamiento, evaluación y registro en MLflow para P2 (Riesgo de Desbalance).
Integra y evalúa en Train, Validación y Test:
  - Baseline de Persistencia (inercia operativa)
  - Modelo Candidato: Regresión Logística
  - Modelo Candidato: HistGradientBoostingClassifier (GBM)
  - Búsquedas anidadas de hiperparámetros (logistic_search, gbm_search)
Exporta métricas a reports/p2_comparacion.parquet y predicciones a reports/p2_predicciones_modelos.parquet.
"""

from pathlib import Path
import itertools
import time
import os
import subprocess
import sys

import joblib

import mlflow
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import auc, classification_report, precision_recall_curve

from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    P2_CLASS_WEIGHT_DEFAULT,
    P2_GBM_GRID,
    P2_GBM_PARAMS,
    P2_GRID_LOGISTIC,
    P2_MAX_ITER_DEFAULT,
    P2_OBJETIVO,
    P2_UMBRAL,
    P2_VARIABLES,
    RANDOM_SEED,
    REPORTS_DIR,
    TRIPS_TRAIN_PATH,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.features import construir_dataset_p2
from proyecto_mibici.modeling.model_p2 import BaselinePersistenciaP2, RiesgoGBMP2, RiesgoLogisticoP2
from proyecto_mibici.modeling.tracking import MLFlowTracker
from proyecto_mibici.plots import figuras_p2

app = typer.Typer()


def extraer_metricas_clasificacion(y_real, y_pred, y_prob, conjunto_nombre: str) -> dict:
    """Calcula precision, recall, f1-score y PR-AUC para la clase de riesgo (positiva)."""
    y_real = np.asarray(y_real, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    rep = classification_report(y_real, y_pred, output_dict=True, zero_division=0)
    pos_key = True if True in rep else "True"

    precision_curve, recall_curve, _ = precision_recall_curve(y_real, y_prob)
    pr_auc_val = float(auc(recall_curve, precision_curve))

    return {
        "conjunto": conjunto_nombre,
        "precision": float(rep[pos_key]["precision"]),
        "recall": float(rep[pos_key]["recall"]),
        "f1_score": float(rep[pos_key]["f1-score"]),
        "pr_auc": pr_auc_val,
    }


def imprimir_metricas_modelo(nombre: str, m_train: dict, m_val: dict, m_test: dict):
    """Despliega en terminal las métricas del modelo evaluado en los 3 splits."""
    logger.info(
        f"[{nombre}] "
        f"Train -> F1: {m_train['f1_score']:.3f} | PR-AUC: {m_train['pr_auc']:.3f} || "
        f"Val -> F1: {m_val['f1_score']:.3f} | PR-AUC: {m_val['pr_auc']:.3f} || "
        f"Test -> Precision: {m_test['precision']:.3f} | Recall: {m_test['recall']:.3f} | F1: {m_test['f1_score']:.3f} | PR-AUC: {m_test['pr_auc']:.3f}"
    )


def _ejecutar(cmd: list[str], etiqueta: str) -> None:
    typer.secho(f"[Entrenamiento] {etiqueta}", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  $ {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("MLFLOW_TRACKING_URI", URI_MLFLOW_TRACKING)
    subprocess.run(cmd, check=True, env=env)


@app.command()
def main(
    modelo: str = typer.Option("ambos", "--modelo", help="Modelo a entrenar: 'baseline', 'logistic', 'gbm', 'logistic_search', 'gbm_search' o 'ambos'"),
    umbral: float = typer.Option(P2_UMBRAL, "--umbral", help="Umbral de decisión de riesgo (P2)"),
    class_weight: str = typer.Option(P2_CLASS_WEIGHT_DEFAULT, "--class-weight", help="Peso de clases para Regresión Logística"),
    max_iter: int = typer.Option(P2_MAX_ITER_DEFAULT, "--max-iter", help="Iteraciones máximas para Regresión Logística"),
):
    tracker = MLFlowTracker(
        tracking_uri=URI_MLFLOW_TRACKING,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )

    logger.info("Cargando y preparando dataset P2...")
    df_p2 = construir_dataset_p2()

    # Partición temporal excluyendo estaciones nuevas del test
    df_train = df_p2[df_p2["conjunto"] == "entrenamiento"].copy()
    df_val = df_p2[df_p2["conjunto"] == "validacion"].copy()
    df_test = df_p2[(df_p2["conjunto"] == "prueba") & (~df_p2["estacion_nueva"])].copy()

    splits = {"Train": df_train, "Validación": df_val, "Test": df_test}
    y_train = df_train[P2_OBJETIVO].astype(bool)
    y_val = df_val[P2_OBJETIVO].astype(bool)
    y_test = df_test[P2_OBJETIVO].astype(bool)

    logger.info(
        f"Registros P2 -> Train: {len(df_train):,} | Validación: {len(df_val):,} | Test: {len(df_test):,}"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    params_base, tags_base = tracker.armar_params_tags(
        "P2",
        splits,
        extra_params={"variable_objetivo": P2_OBJETIVO, "umbral_capacidad": umbral},
        col_tiempo="hora",
    )

    filas_comparacion = []
    cols_id = [c for c in ["estacion_id", "hora"] if c in df_test.columns]
    df_predicciones = df_test[cols_id].copy()
    df_predicciones["real"] = y_test.values.astype(int)

    # -------------------------------------------------------------
    # 1. EVALUACIÓN DE LÍNEA BASE (PERSISTENCIA INGENUA)
    # -------------------------------------------------------------
    if modelo.lower() in ("baseline", "naive", "ambos"):
        logger.info("--- Evaluando Línea Base Ingenua: Persistencia ---")
        modelo_naive = BaselinePersistenciaP2(col_persistencia="en_riesgo")

        ruta_modelo = MODELS_DIR / "p2_baseline_model.pkl"
        joblib.dump(modelo_naive, ruta_modelo)
        logger.info(f"Modelo P2 baseline guardado exitosamente en {ruta_modelo}")

        pred_train_naive = modelo_naive.predict(df_train)
        prob_train_naive = modelo_naive.predict_proba(df_train)[:, 1]
        m_naive_train = extraer_metricas_clasificacion(y_train, pred_train_naive, prob_train_naive, "Train")

        pred_val_naive = modelo_naive.predict(df_val)
        prob_val_naive = modelo_naive.predict_proba(df_val)[:, 1]
        m_naive_val = extraer_metricas_clasificacion(y_val, pred_val_naive, prob_val_naive, "Validación")

        pred_test_naive = modelo_naive.predict(df_test)
        prob_test_naive = modelo_naive.predict_proba(df_test)[:, 1]
        m_naive_test = extraer_metricas_clasificacion(y_test, pred_test_naive, prob_test_naive, "Test")

        df_predicciones["naive_persistencia"] = pred_test_naive.astype(int)
        imprimir_metricas_modelo("naive_persistencia", m_naive_train, m_naive_val, m_naive_test)

        tracker.registrar_run(
            run_name="P2_naive_persistencia",
            params={
                **params_base,
                "modelo": "BaselinePersistenciaP2",
                "regla": "en_riesgo_t",
            },
            tags={
                **tags_base,
                "version": "naive_persistencia",
                "question": "P2",
                "model_family": "naive",
                "stage_of_work": "baseline",
            },
            tabla=pd.DataFrame([m_naive_train, m_naive_val, m_naive_test]),
            tabla_est=pd.DataFrame(),
            df_train=df_train,
            ruta_entrenamiento=str(TRIPS_TRAIN_PATH),
        )

        filas_comparacion.append({
            "modelo": "naive_persistencia",
            "train_f1": m_naive_train["f1_score"],
            "val_f1": m_naive_val["f1_score"],
            "test_precision": m_naive_test["precision"],
            "test_recall": m_naive_test["recall"],
            "test_f1": m_naive_test["f1_score"],
            "test_pr_auc": m_naive_test["pr_auc"],
        })

    # -------------------------------------------------------------
    # 2. MODELO CANDIDATO: REGRESIÓN LOGÍSTICA
    # -------------------------------------------------------------
    if modelo.lower() in ("logistic", "logistica", "ambos"):
        logger.info("--- Entrenando Modelo Candidato: RiesgoLogisticoP2 ---")
        cw = None if class_weight == "None" else class_weight
        modelo_p2 = RiesgoLogisticoP2(
            features=P2_VARIABLES,
            class_weight=None if class_weight == "None" else class_weight,
            random_state=RANDOM_SEED,
            max_iter=max_iter,
            umbral=umbral,
        )
        modelo_p2.fit(df_train[P2_VARIABLES], y_train)

        ruta_modelo = MODELS_DIR / "p2_logistic_model.pkl"
        modelo_p2.save(ruta_modelo)
        logger.info(f"Modelo P2 guardado exitosamente en {ruta_modelo}")

        pred_train_log = modelo_p2.predict(df_train[P2_VARIABLES])
        prob_train_log = modelo_p2.predict_proba(df_train[P2_VARIABLES])[:, 1]
        m_log_train = extraer_metricas_clasificacion(y_train, pred_train_log, prob_train_log, "Train")

        pred_val_log = modelo_p2.predict(df_val[P2_VARIABLES])
        prob_val_log = modelo_p2.predict_proba(df_val[P2_VARIABLES])[:, 1]
        m_log_val = extraer_metricas_clasificacion(y_val, pred_val_log, prob_val_log, "Validación")

        pred_test_log = modelo_p2.predict(df_test[P2_VARIABLES])
        prob_test_log = modelo_p2.predict_proba(df_test[P2_VARIABLES])[:, 1]
        m_log_test = extraer_metricas_clasificacion(y_test, pred_test_log, prob_test_log, "Test")

        df_predicciones["prob_riesgo"] = np.round(prob_test_log, 4)
        df_predicciones["pred_riesgo"] = pred_test_log.astype(int)

        imprimir_metricas_modelo("logistic_regression", m_log_train, m_log_val, m_log_test)

        figuras = figuras_p2(y_test, pred_test_log, prob_test_log, "logistic_regression")
        estimador_log = getattr(modelo_p2, "estimator_", modelo_p2)

        tracker.registrar_run(
            run_name="p2-baseline-logistic-regression",
            params={
                **params_base,
                "modelo": "LogisticRegression",
                "variables_predictoras": ", ".join(P2_VARIABLES),
                "class_weight": class_weight,
                "max_iter": max_iter,
                "umbral": umbral,
            },
            tags={
                **tags_base,
                "version": "v1_logistic",
                "question": "P2",
                "model_family": "logistic_regression",
                "stage_of_work": "candidate",
            },
            tabla=pd.DataFrame([m_log_train, m_log_val, m_log_test]),
            tabla_est=pd.DataFrame(),
            modelo=estimador_log,
            X_ejemplo=df_train[P2_VARIABLES].head(5),
            figuras=figuras_p2(y_test, pred_test_log, prob_test_log, "logistic_regression"),
            df_train=df_train[P2_VARIABLES],
            ruta_entrenamiento=str(TRIPS_TRAIN_PATH),
        )

        filas_comparacion.append({
            "modelo": "logistic_regression",
            "train_f1": m_log_train["f1_score"],
            "val_f1": m_log_val["f1_score"],
            "test_precision": m_log_test["precision"],
            "test_recall": m_log_test["recall"],
            "test_f1": m_log_test["f1_score"],
            "test_pr_auc": m_log_test["pr_auc"],
        })

    # -------------------------------------------------------------
    # 3. MODELO CANDIDATO: GRADIENT BOOSTING (HistGradientBoostingClassifier)
    # -------------------------------------------------------------
    if modelo.lower() in ("gbm", "ambos"):
        logger.info("--- Entrenando Modelo Candidato: HistGradientBoostingClassifier (GBM) ---")
        modelo_gbm = RiesgoGBMP2(
            features=P2_VARIABLES,
            **P2_GBM_PARAMS,
        )
        modelo_gbm.fit(df_train[P2_VARIABLES], y_train)

        ruta_modelo = MODELS_DIR / "p2_gbm_model.pkl"
        modelo_gbm.save(ruta_modelo)
        logger.info(f"Modelo GBM P2 guardado exitosamente en {ruta_modelo}")

        pred_train_gbm = modelo_gbm.predict(df_train[P2_VARIABLES])
        prob_train_gbm = modelo_gbm.predict_proba(df_train[P2_VARIABLES])[:, 1]
        m_gbm_train = extraer_metricas_clasificacion(y_train, pred_train_gbm, prob_train_gbm, "Train")

        pred_val_gbm = modelo_gbm.predict(df_val[P2_VARIABLES])
        prob_val_gbm = modelo_gbm.predict_proba(df_val[P2_VARIABLES])[:, 1]
        m_gbm_val = extraer_metricas_clasificacion(y_val, pred_val_gbm, prob_val_gbm, "Validación")

        pred_test_gbm = modelo_gbm.predict(df_test[P2_VARIABLES])
        prob_test_gbm = modelo_gbm.predict_proba(df_test[P2_VARIABLES])[:, 1]
        m_gbm_test = extraer_metricas_clasificacion(y_test, pred_test_gbm, prob_test_gbm, "Test")

        df_predicciones["prob_riesgo_gbm"] = np.round(prob_test_gbm, 4)
        df_predicciones["pred_riesgo_gbm"] = pred_test_gbm.astype(int)

        imprimir_metricas_modelo("gbm", m_gbm_train, m_gbm_val, m_gbm_test)

        figuras = figuras_p2(y_test, pred_test_gbm, prob_test_gbm, "gbm")
        estimador_gbm = getattr(modelo_gbm, "estimator_", modelo_gbm)

        tracker.registrar_run(
            run_name="p2-gbm",
            params={
                **params_base,
                "modelo": "HistGradientBoostingClassifier",
                "variables_predictoras": ", ".join(P2_VARIABLES),
                **P2_GBM_PARAMS,
            },
            tags={
                **tags_base,
                "version": "v1_gbm",
                "question": "P2",
                "model_family": "hist_gradient_boosting",
                "stage_of_work": "candidate",
            },
            tabla=pd.DataFrame([m_gbm_train, m_gbm_val, m_gbm_test]),
            tabla_est=pd.DataFrame(),
            modelo=estimador_gbm,
            X_ejemplo=df_train[P2_VARIABLES].head(5),
            figuras=figuras_p2(y_test, pred_test_gbm, prob_test_gbm, "gbm"),
            df_train=df_train[P2_VARIABLES],
            ruta_entrenamiento=str(TRIPS_TRAIN_PATH),
        )

        filas_comparacion.append({
            "modelo": "gbm",
            "train_f1": m_gbm_train["f1_score"],
            "val_f1": m_gbm_val["f1_score"],
            "test_precision": m_gbm_test["precision"],
            "test_recall": m_gbm_test["recall"],
            "test_f1": m_gbm_test["f1_score"],
            "test_pr_auc": m_gbm_test["pr_auc"],
        })

    # -------------------------------------------------------------
    # 3.5. BÚSQUEDA ANIDADA DE HIPERPARÁMETROS (GRADIENT BOOSTING)
    # -------------------------------------------------------------
    if modelo.lower() == "gbm_search":
        combinaciones = [dict(zip(P2_GBM_GRID, v)) for v in itertools.product(*P2_GBM_GRID.values())]
        mejor_hp, mejor_pr_auc = None, float("-inf")

        with mlflow.start_run(run_name="P2_gbm_search") as padre:
            mlflow.set_tags({**tags_base, "stage_of_work": "search", "model_family": "hist_gradient_boosting"})
            mlflow.log_params({"n_combinaciones": len(combinaciones), "criterio": "validation_pr_auc", "objetivo": "gbm"})

            for i, hp in enumerate(combinaciones):
                source_tags = tracker._get_source_tags()
                with mlflow.start_run(run_name=f"search_{i:02d}", nested=True):
                    if source_tags:
                        mlflow.set_tags(source_tags)
                    mlflow.log_params(hp)
                    logger.info(f"search_{i:02d}:{hp}")
                    try:
                        t0 = time.time()
                        m = HistGradientBoostingClassifier(random_state=RANDOM_SEED, **hp)
                        m.fit(df_train[P2_VARIABLES], y_train)
                        segundos = time.time() - t0

                        pred_val = m.predict(df_val[P2_VARIABLES])
                        prob_val = m.predict_proba(df_val[P2_VARIABLES])[:, 1]
                        m_val = extraer_metricas_clasificacion(y_val, pred_val, prob_val, "Validación")

                        mlflow.set_tags({**tags_base, "stage_of_work": "search", "model_family": "hist_gradient_boosting"})
                        mlflow.log_metrics({**tracker.metricas_a_dict(pd.DataFrame([m_val])), "train_seconds": segundos})

                        if m_val["pr_auc"] > mejor_pr_auc:
                            mejor_pr_auc, mejor_hp = m_val["pr_auc"], hp
                    except Exception as e:
                        mlflow.set_tag("fallo", "true")
                        mlflow.set_tag("error", str(e)[:250])
                        logger.warning(f"search_{i:02d} falló y se omite: {e}")
                        continue

            mlflow.log_metric("best_validation_pr_auc", mejor_pr_auc)
            mlflow.log_params({f"best_{k}": v for k, v in mejor_hp.items()})
            id_padre = padre.info.run_id

            # Entrenar modelo final con mejores hiperparámetros (anidado bajo el padre)
            modelo_final = RiesgoGBMP2(
                features=P2_VARIABLES,
                **mejor_hp,
            )
            modelo_final.fit(df_train[P2_VARIABLES], y_train)

            ruta_modelo = MODELS_DIR / "p2_gbm_search_model.pkl"
            modelo_final.save(ruta_modelo)
            logger.info(f"Modelo P2 gbm_search guardado exitosamente en {ruta_modelo}")

            pred_train_gbm = modelo_final.predict(df_train[P2_VARIABLES])
            prob_train_gbm = modelo_final.predict_proba(df_train[P2_VARIABLES])[:, 1]
            m_gbm_train = extraer_metricas_clasificacion(y_train, pred_train_gbm, prob_train_gbm, "Train")

            pred_val_gbm = modelo_final.predict(df_val[P2_VARIABLES])
            prob_val_gbm = modelo_final.predict_proba(df_val[P2_VARIABLES])[:, 1]
            m_gbm_val = extraer_metricas_clasificacion(y_val, pred_val_gbm, prob_val_gbm, "Validación")

            pred_test_gbm = modelo_final.predict(df_test[P2_VARIABLES])
            prob_test_gbm = modelo_final.predict_proba(df_test[P2_VARIABLES])[:, 1]
            m_gbm_test = extraer_metricas_clasificacion(y_test, pred_test_gbm, prob_test_gbm, "Test")

            df_predicciones["prob_riesgo_gbm_search"] = np.round(prob_test_gbm, 4)
            df_predicciones["pred_riesgo_gbm_search"] = pred_test_gbm.astype(int)

            imprimir_metricas_modelo("gbm_search_tuned", m_gbm_train, m_gbm_val, m_gbm_test)

            figuras = figuras_p2(y_test, pred_test_gbm, prob_test_gbm, "gbm_tuned")
            estimador_gbm = getattr(modelo_final, "estimator_", modelo_final)

            tracker.registrar_run(
                run_name="p2-gbm-search-tuned",
                params={
                    **params_base,
                    "modelo": "HistGradientBoostingClassifier_Search",
                    "variables_predictoras": ", ".join(P2_VARIABLES),
                    **mejor_hp,
                },
                tags={
                    **tags_base,
                    "version": "gbm_search_tuned",
                    "question": "P2",
                    "model_family": "hist_gradient_boosting",
                    "stage_of_work": "candidate",
                    "search_run_id": id_padre,
                },
                tabla=pd.DataFrame([m_gbm_train, m_gbm_val, m_gbm_test]),
                tabla_est=pd.DataFrame(),
                modelo=estimador_gbm,
                X_ejemplo=df_train[P2_VARIABLES].head(5),
                figuras=figuras_p2(y_test, pred_test_gbm, prob_test_gbm, "gbm_tuned"),
                df_train=df_train[P2_VARIABLES],
                ruta_entrenamiento=str(TRIPS_TRAIN_PATH),
                nested=True,
            )

            filas_comparacion.append({
                "modelo": "gbm_search_tuned",
                "train_f1": m_gbm_train["f1_score"],
                "val_f1": m_gbm_val["f1_score"],
                "test_precision": m_gbm_test["precision"],
                "test_recall": m_gbm_test["recall"],
                "test_f1": m_gbm_test["f1_score"],
                "test_pr_auc": m_gbm_test["pr_auc"],
            })

    # -------------------------------------------------------------
    # 4. BÚSQUEDA ANIDADA DE HIPERPARÁMETROS (REGRESIÓN LOGÍSTICA)
    # -------------------------------------------------------------
    if modelo.lower() == "logistic_search":
        combinaciones = [dict(zip(P2_GRID_LOGISTIC, v)) for v in itertools.product(*P2_GRID_LOGISTIC.values())]
        mejor_hp, mejor_f1 = None, float("-inf")

        with mlflow.start_run(run_name="P2_logistic_search") as padre:
            mlflow.set_tags({**tags_base, "stage_of_work": "search", "model_family": "logistic_regression"})
            mlflow.log_params({"n_combinaciones": len(combinaciones), "criterio": "validation_f1", "objetivo": "logistic"})

            for i, hp in enumerate(combinaciones):
                source_tags = tracker._get_source_tags()
                with mlflow.start_run(run_name=f"search_{i:02d}", nested=True):
                    if source_tags:
                        mlflow.set_tags(source_tags)
                    mlflow.log_params(hp)
                    logger.info(f"search_{i:02d}:{hp}")
                    try:
                        m = RiesgoLogisticoP2(
                            features=P2_VARIABLES,
                            class_weight=hp["class_weight"],
                            random_state=RANDOM_SEED,
                            max_iter=hp["max_iter"],
                            umbral=umbral,
                            C=hp["C"],
                        )
                        m.fit(df_train[P2_VARIABLES], y_train)

                        pred_val = m.predict(df_val[P2_VARIABLES])
                        prob_val = m.predict_proba(df_val[P2_VARIABLES])[:, 1]
                        m_val = extraer_metricas_clasificacion(y_val, pred_val, prob_val, "Validación")

                        mlflow.set_tags({**tags_base, "stage_of_work": "search", "model_family": "logistic_regression"})
                        mlflow.log_metrics({**tracker.metricas_a_dict(pd.DataFrame([m_val]))})

                        if m_val["f1_score"] > mejor_f1:
                            mejor_f1, mejor_hp = m_val["f1_score"], hp
                    except Exception as e:
                        mlflow.set_tag("fallo", "true")
                        mlflow.set_tag("error", str(e)[:250])
                        logger.warning(f"search_{i:02d} falló y se omite: {e}")
                        continue

            mlflow.log_metric("best_validation_f1", mejor_f1)
            mlflow.log_params({f"best_{k}": v for k, v in mejor_hp.items()})
            id_padre = padre.info.run_id

            # Entrenar modelo final con mejores hiperparámetros (anidado bajo el padre)
            modelo_final = RiesgoLogisticoP2(
                features=P2_VARIABLES,
                class_weight=mejor_hp["class_weight"],
                random_state=RANDOM_SEED,
                max_iter=mejor_hp["max_iter"],
                umbral=umbral,
                C=mejor_hp["C"],
            )
            modelo_final.fit(df_train[P2_VARIABLES], y_train)

            ruta_modelo = MODELS_DIR / "p2_logistic_search_model.pkl"
            modelo_final.save(ruta_modelo)
            logger.info(f"Modelo P2 logistic_search guardado exitosamente en {ruta_modelo}")

            pred_train_log = modelo_final.predict(df_train[P2_VARIABLES])
            prob_train_log = modelo_final.predict_proba(df_train[P2_VARIABLES])[:, 1]
            m_log_train = extraer_metricas_clasificacion(y_train, pred_train_log, prob_train_log, "Train")

            pred_val_log = modelo_final.predict(df_val[P2_VARIABLES])
            prob_val_log = modelo_final.predict_proba(df_val[P2_VARIABLES])[:, 1]
            m_log_val = extraer_metricas_clasificacion(y_val, pred_val_log, prob_val_log, "Validación")

            pred_test_log = modelo_final.predict(df_test[P2_VARIABLES])
            prob_test_log = modelo_final.predict_proba(df_test[P2_VARIABLES])[:, 1]
            m_log_test = extraer_metricas_clasificacion(y_test, pred_test_log, prob_test_log, "Test")

            df_predicciones["prob_riesgo_log_search"] = np.round(prob_test_log, 4)
            df_predicciones["pred_riesgo_log_search"] = pred_test_log.astype(int)

            imprimir_metricas_modelo("logistic_search_tuned", m_log_train, m_log_val, m_log_test)

            figuras = figuras_p2(y_test, pred_test_log, prob_test_log, "logistic_tuned")
            estimador_log = getattr(modelo_final, "estimator_", modelo_final)

            tracker.registrar_run(
                run_name="p2-logistic-search-tuned",
                params={
                    **params_base,
                    "modelo": "LogisticRegression_Search",
                    "variables_predictoras": ", ".join(P2_VARIABLES),
                    **mejor_hp,
                },
                tags={
                    **tags_base,
                    "version": "logistic_search_tuned",
                    "question": "P2",
                    "model_family": "logistic_regression",
                    "stage_of_work": "candidate",
                    "search_run_id": id_padre,
                },
                tabla=pd.DataFrame([m_log_train, m_log_val, m_log_test]),
                tabla_est=pd.DataFrame(),
                modelo=estimador_log,
                X_ejemplo=df_train[P2_VARIABLES].head(5),
                figuras=figuras_p2(y_test, pred_test_log, prob_test_log, "logistic_tuned"),
                df_train=df_train[P2_VARIABLES],
                ruta_entrenamiento=str(TRIPS_TRAIN_PATH),
                nested=True,
            )

            filas_comparacion.append({
                "modelo": "logistic_search_tuned",
                "train_f1": m_log_train["f1_score"],
                "val_f1": m_log_val["f1_score"],
                "test_precision": m_log_test["precision"],
                "test_recall": m_log_test["recall"],
                "test_f1": m_log_test["f1_score"],
                "test_pr_auc": m_log_test["pr_auc"],
            })

    # -------------------------------------------------------------
    # 5. GUARDAR COMPARATIVA Y PREDICCIONES (SOLO PARQUET)
    # -------------------------------------------------------------
    df_p2_comp = pd.DataFrame(filas_comparacion).set_index("modelo")
    p2_comp_path = REPORTS_DIR / "p2_comparacion.parquet"
    df_p2_comp.round(3).reset_index().to_parquet(p2_comp_path, index=False)

    logger.info(f"Reporte de entrenamiento P2 guardado en Parquet: {p2_comp_path}")
    logger.info("\n" + df_p2_comp.round(3).to_string())

    preds_out_path = REPORTS_DIR / "p2_predicciones_modelos.parquet"
    df_predicciones.to_parquet(preds_out_path, index=False)
    logger.success(f"Predicciones de P2 guardadas en {preds_out_path}")


if __name__ == "__main__":
    app()
