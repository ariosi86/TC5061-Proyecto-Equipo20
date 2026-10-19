#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo centralizado de visualizaciones y generación de figuras para P1, P2 y P3.
Utiliza el backend 'Agg' para evitar conflictos de interfaz gráfica en servidores o scripts.
"""

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc


# ==========================================
# P1: DEMANDA HORARIA (REGRESIÓN)
# ==========================================

def figuras_p1(df_test, pred, target, etiqueta, estacion_id=51):
    """Genera las figuras de P1: Penitenciaría (primera semana de test) y dispersión global."""
    res = df_test[["origen_id", "target_hour", target]].copy()
    res["pred"] = pred

    # 1. Penitenciaría: primera semana del test
    g = res[res["origen_id"] == estacion_id].sort_values("target_hour")
    g = g[g["target_hour"] < g["target_hour"].min() + pd.Timedelta(days=7)]
    fig1, ax = plt.subplots(figsize=(14, 5))
    ax.plot(g["target_hour"], g[target], label="Viajes reales")
    ax.plot(g["target_hour"], g["pred"], label="Viajes predichos")
    ax.set_title(f"Demanda real vs. predicha, López Cotilla / Penitenciaría ({etiqueta})")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Número de viajes")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig1.tight_layout()

    # 2. Dispersión global: real vs. predicho (test)
    fig2, ax = plt.subplots(figsize=(7, 7))
    y_real = df_test[target].to_numpy()
    ax.scatter(y_real, pred, alpha=0.15, s=8)
    lim = max(y_real.max(), pred.max())
    ax.plot([0, lim], [0, lim], linestyle="--", label="Predicción perfecta")
    ax.set_title(f"Viajes reales vs. predichos, Test ({etiqueta})")
    ax.set_xlabel("Viajes reales")
    ax.set_ylabel("Viajes predichos")
    ax.legend()
    fig2.tight_layout()

    return {
        f"penitenciaria_real_vs_pred_{etiqueta}": fig1,
        f"scatter_test_{etiqueta}": fig2,
    }


# ==========================================
# P2: RIESGO DE DESBALANCE (CLASIFICACIÓN)
# ==========================================

def figuras_p2(y_real, y_pred, y_prob, etiqueta):
    """Genera la matriz de confusión y la curva Precision-Recall para evaluar P2."""
    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    # 1. Matriz de Confusión
    cm = confusion_matrix(y_real, y_pred)
    fig1, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    fig1.colorbar(cax)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontweight="bold", fontsize=12)

    ax.set_xlabel("Predicción de Riesgo", fontweight="bold")
    ax.set_ylabel("Riesgo Real", fontweight="bold")
    ax.set_title(f"Matriz de Confusión - P2 ({etiqueta})", pad=15, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Sin Riesgo", "En Riesgo"])
    ax.set_yticklabels(["Sin Riesgo", "En Riesgo"])
    fig1.tight_layout()

    # 2. Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_real, y_prob)
    pr_auc = auc(recall, precision)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}", color="darkorange", lw=2)
    ax2.set_xlabel("Recall", fontweight="bold")
    ax2.set_ylabel("Precision", fontweight="bold")
    ax2.set_title(f"Curva Precision-Recall - P2 ({etiqueta})", pad=15, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    return {
        f"matriz_confusion_{etiqueta}": fig1,
        f"curva_pr_{etiqueta}": fig2,
    }


# ==========================================
# P3: CLUSTERING DE ESTACIONES (USO)
# ==========================================

def graficar_mapa_grupos(perfil, etiquetas, df_stations, titulo):
    """Dibuja las estaciones en el mapa de la ciudad coloreadas por grupo para P3."""
    coords = df_stations.set_index("id").loc[perfil.index, ["latitude", "longitude"]]

    fig, ax = plt.subplots(figsize=(7, 7))
    puntos = ax.scatter(coords["longitude"], coords["latitude"], c=etiquetas, cmap="tab10", s=18)
    ax.legend(*puntos.legend_elements(), title="Grupo")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(titulo)
    fig.tight_layout()
    
    return fig