#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de entrenamiento, evaluación y registro en MLflow para P2 (Riesgo de Desbalance).
Compara e imprime en consola:
  - Baseline de Persistencia (inercia operativa)
  - Modelo Candidato: Regresión Logística Balanceada
Exporta métricas a reports/p2_comparacion.csv y predicciones a reports/p2_predicciones_modelos.parquet.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import auc, classification_report, precision_recall_curve

from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    P2_OBJETIVO,
    P2_UMBRAL,
    P2_VARIABLES,
    REPORTS_DIR,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.features import construir_dataset_p2
from proyecto_mibici.modeling.model_p2 import BaselinePersistenciaP2, RiesgoLogisticoP2
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


def imprimir_metricas_modelo(nombre: str, m_train: dict, m_test: dict):
    """Despliega en terminal las métricas del modelo evaluado."""
    logger.info(
        f"[{nombre}] "
        f"Train -> Precision: {m_train['precision']:.3f} | Recall: {m_train['recall']:.3f} | F1: {m_train['f1_score']:.3f} || "
        f"Test -> Precision: {m_test['precision']:.3f} | Recall: {m_test['recall']:.3f} | F1: {m_test['f1_score']:.3f} | PR-AUC: {m_test['pr_auc']:.3f}"
    )


@app.command()
def main():
    tracker = MLFlowTracker(
        tracking_uri=URI_MLFLOW_TRACKING,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )

    logger.info("Cargando y preparando dataset P2...")
    df_p2 = construir_dataset_p2()
    
    df_train = df_p2[df_p2["conjunto"] == "entrenamiento"].copy()
    df_val = df_p2[df_p2["conjunto"] == "validacion"].copy()
    df_test = df_p2[(df_p2["conjunto"] == "prueba") & (~df_p2["estacion_nueva"])].copy()
    
    splits = {"Train": df_train, "Validación": df_val, "Test": df_test}
    y_train = df_train[P2_OBJETIVO]
    y_test = df_test[P2_OBJETIVO]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    params_base, tags_base = tracker.armar_params_tags(
        "P2",
        splits,
        extra_params={"variable_objetivo": P2_OBJETIVO, "umbral": P2_UMBRAL},
        col_tiempo="hora",
    )

    filas_comparacion = []

    # DataFrame para consolidar las predicciones de ambos modelos en Test
    cols_id = [c for c in ["estacion_id", "hora"] if c in df_test.columns]
    df_predicciones = df_test[cols_id].copy()
    df_predicciones["real"] = y_test.values

    # -------------------------------------------------------------
    # 1. EVALUACIÓN DE LÍNEA BASE (PERSISTENCIA INGENUA)
    # -------------------------------------------------------------
    logger.info("--- Evaluando Línea Base Ingenua: Persistencia ---")
    modelo_naive = BaselinePersistenciaP2(col_persistencia="en_riesgo")
    
    pred_train_naive = modelo_naive.predict(df_train)
    prob_train_naive = modelo_naive.predict_proba(df_train)[:, 1]
    m_naive_train = extraer_metricas_clasificacion(y_train, pred_train_naive, prob_train_naive, "Train")

    pred_test_naive = modelo_naive.predict(df_test)
    prob_test_naive = modelo_naive.predict_proba(df_test)[:, 1]
    m_naive_test = extraer_metricas_clasificacion(y_test, pred_test_naive, prob_test_naive, "Test")

    df_predicciones["naive_persistencia"] = pred_test_naive.astype(int)
    imprimir_metricas_modelo("naive_persistencia", m_naive_train, m_naive_test)

    tracker.registrar_run(
        run_name="P2_naive_persistencia",
        params={**params_base, "modelo": "BaselinePersistenciaP2", "regla": "en_riesgo_t"},
        tags={**tags_base, "version": "naive_persistencia"},
        tabla=pd.DataFrame([m_naive_train, m_naive_test]),
        tabla_est=pd.DataFrame(),
    )

    filas_comparacion.append({
        "modelo": "naive_persistencia",
        "train_precision": m_naive_train["precision"],
        "train_recall": m_naive_train["recall"],
        "train_f1": m_naive_train["f1_score"],
        "test_precision": m_naive_test["precision"],
        "test_recall": m_naive_test["recall"],
        "test_f1": m_naive_test["f1_score"],
        "test_pr_auc": m_naive_test["pr_auc"],
    })

    # -------------------------------------------------------------
    # 2. MODELO CANDIDATO: REGRESIÓN LOGÍSTICA BALANCEADA
    # -------------------------------------------------------------
    logger.info("--- Entrenando Modelo Candidato: RiesgoLogisticoP2 ---")
    modelo_p2 = RiesgoLogisticoP2()
    modelo_p2.fit(df_train[P2_VARIABLES], y_train)

    # Persistir modelo serializado oficial
    ruta_modelo = MODELS_DIR / "p2_logistic_model.pkl"
    modelo_p2.save(ruta_modelo)
    logger.info(f"Pipeline P2 guardado exitosamente en {ruta_modelo}")

    pred_train_log = modelo_p2.predict(df_train[P2_VARIABLES], umbral=P2_UMBRAL)
    prob_train_log = modelo_p2.predict_proba(df_train[P2_VARIABLES])[:, 1]
    m_log_train = extraer_metricas_clasificacion(y_train, pred_train_log, prob_train_log, "Train")

    pred_test_log = modelo_p2.predict(df_test[P2_VARIABLES], umbral=P2_UMBRAL)
    prob_test_log = modelo_p2.predict_proba(df_test[P2_VARIABLES])[:, 1]
    m_log_test = extraer_metricas_clasificacion(y_test, pred_test_log, prob_test_log, "Test")

    df_predicciones["prob_riesgo"] = np.round(prob_test_log, 4)
    df_predicciones["pred_riesgo"] = pred_test_log.astype(int)

    imprimir_metricas_modelo("logistic_regression", m_log_train, m_log_test)

    # Generación de figuras oficiales (Matriz de confusión y Precision-Recall)
    figuras = figuras_p2(y_test, pred_test_log, prob_test_log, "logistic_regression")

    tracker.registrar_run(
        run_name="P2_v1_logistic_regression",
        params={**params_base, "modelo": "LogisticRegression", "variables": ", ".join(P2_VARIABLES)},
        tags={**tags_base, "version": "v1_logistic"},
        tabla=pd.DataFrame([m_log_train, m_log_test]),
        tabla_est=pd.DataFrame(),
        modelo=modelo_p2.pipeline,
        X_ejemplo=df_train[P2_VARIABLES].head(5),
        figuras=figuras,
    )

    filas_comparacion.append({
        "modelo": "logistic_regression",
        "train_precision": m_log_train["precision"],
        "train_recall": m_log_train["recall"],
        "train_f1": m_log_train["f1_score"],
        "test_precision": m_log_test["precision"],
        "test_recall": m_log_test["recall"],
        "test_f1": m_log_test["f1_score"],
        "test_pr_auc": m_log_test["pr_auc"],
    })

    # -------------------------------------------------------------
    # 3. GUARDAR Y MOSTRAR REPORTE COMPARATIVO DE MÉTRICAS
    # -------------------------------------------------------------
    df_p2_comp = pd.DataFrame(filas_comparacion).set_index("modelo")
    p2_comp_path = REPORTS_DIR / "p2_comparacion.csv"
    df_p2_comp.round(3).to_csv(p2_comp_path)

    tabla_metricas_str = df_p2_comp.round(3).to_string()
    logger.info(
        f"\n{'='*75}\n"
        f"         RESUMEN COMPARATIVO DE MODELOS (P2 - RIESGO DE DESBALANCE)\n"
        f"{'='*75}\n"
        f"{tabla_metricas_str}\n"
        f"{'='*75}\n"
        f"Archivo guardado en: {p2_comp_path}"
    )

    # -------------------------------------------------------------
    # 4. GUARDAR Y MOSTRAR PREDICCIONES CONSOLIDADAS
    # -------------------------------------------------------------
    preds_out_path = REPORTS_DIR / "p2_predicciones_modelos.parquet"
    df_predicciones.to_parquet(preds_out_path, index=False)

    cols_mostrar = [
        c for c in [
            "estacion_id", "hora", "real",
            "naive_persistencia", "prob_riesgo", "pred_riesgo"
        ] if c in df_predicciones.columns
    ]
    tabla_preds_str = df_predicciones[cols_mostrar].head(10).to_string(index=False)
    logger.info(
        f"\n{'-'*75}\n"
        f"        MUESTRA COMPARATIVA DE PREDICCIONES (TEST - 10 REGISTROS)\n"
        f"{'-'*75}\n"
        f"{tabla_preds_str}\n"
        f"{'-'*75}\n"
        f"Predicciones guardadas en: {preds_out_path}\n"
    )


if __name__ == "__main__":
    app()