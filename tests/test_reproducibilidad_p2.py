"""Prueba de reproducibilidad del entrenamiento del GBM de P2 (modelo champion).

El GBM usa early_stopping: aparta al azar una fracción de los datos para decidir
cuándo dejar de entrenar. La semilla (RANDOM_SEED de config.py) fija ese sorteo,
así que dos entrenamientos con la misma semilla deben dar exactamente lo mismo.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, f1_score

from proyecto_mibici.config import RANDOM_SEED
from proyecto_mibici.modeling.model_p2 import RiesgoGBMP2

VARIABLES = ["hora_del_dia", "dia_semana", "acumulado_relativo", "balance_ult_3h_relativo"]


@pytest.fixture
def particion():
    """Datos sintéticos con ruido, divididos en entrenamiento (primeras filas) y prueba."""
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({
        "hora_del_dia": rng.integers(0, 24, n),
        "dia_semana": rng.integers(0, 7, n),
        "acumulado_relativo": rng.uniform(-1, 1, n),
        "balance_ult_3h_relativo": rng.uniform(-0.5, 0.5, n),
    })
    ruido = rng.normal(0, 0.25, n)
    y = (X["acumulado_relativo"].abs() + X["balance_ult_3h_relativo"] + ruido) > 0.6
    corte = int(n * 0.8)   # partición ordenada, sin barajar, como la temporal del proyecto
    return X.iloc[:corte], y.iloc[:corte], X.iloc[corte:], y.iloc[corte:]


def entrenar_y_evaluar(semilla, X_tr, y_tr, X_te, y_te):
    """Entrena un GBM nuevo con la semilla dada y devuelve predicciones y métricas."""
    modelo = RiesgoGBMP2(features=VARIABLES, random_state=semilla).fit(X_tr, y_tr)
    proba = modelo.predict_proba(X_te)[:, 1]
    pred = modelo.predict(X_te)
    return {
        "pred": pred,
        "proba": proba,
        "f1": f1_score(y_te, pred),
        "pr_auc": average_precision_score(y_te, proba),
        "iteraciones": modelo.estimator_.n_iter_,
    }


def test_misma_semilla_produce_los_mismos_resultados(particion):
    r1 = entrenar_y_evaluar(RANDOM_SEED, *particion)
    r2 = entrenar_y_evaluar(RANDOM_SEED, *particion)

    assert np.array_equal(r1["pred"], r2["pred"])
    assert np.array_equal(r1["proba"], r2["proba"])   # igualdad exacta, no aproximada
    assert r1["f1"] == r2["f1"]
    assert r1["pr_auc"] == r2["pr_auc"]
    assert r1["iteraciones"] == r2["iteraciones"]


def test_la_semilla_si_influye_en_el_entrenamiento(particion):
    # Control: si cambiar la semilla no cambiara nada, la prueba anterior
    # pasaría aunque la semilla no se estuviera usando.
    r1 = entrenar_y_evaluar(RANDOM_SEED, *particion)
    r2 = entrenar_y_evaluar(RANDOM_SEED + 1, *particion)

    assert not np.array_equal(r1["proba"], r2["proba"])
