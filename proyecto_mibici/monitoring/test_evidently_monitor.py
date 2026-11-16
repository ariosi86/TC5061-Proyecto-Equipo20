#!/usr/bin/env python
"""
Pruebas unitarias de la clase EvidentlyMonitor usando datos sintéticos.

No requiere datos reales ni servidor MLflow. Valida:
- Instanciación de la clase.
- tabla_scores() con columnas numéricas y categóricas.
- comparar() con log_mlflow=False (genera HTML + CSV localmente).
- Casos borde: dataframes vacíos y columnas faltantes.

Uso:
    python proyecto_mibici/monitoring/test_evidently_monitor.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Asegura que se importe el paquete local.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proyecto_mibici.monitoring import EvidentlyMonitor  # noqa: E402

# Carpeta temporal de salida para las pruebas.
OUT_DIR = Path("/tmp/opencode/test_evidently_monitor")
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


def hacer_datos(n: int = 200, desplazamiento: float = 0.0) -> pd.DataFrame:
    """Genera un DataFrame sintético con columnas numéricas y categóricas."""
    return pd.DataFrame({
        "edad": np.random.normal(40 + desplazamiento, 10, n),
        "saldo": np.random.exponential(1000 + desplazamiento * 50, n),
        "puntaje": np.random.uniform(0, 100 + desplazamiento * 10, n),
        "sexo": np.random.choice(["M", "F"], n),
        "region": np.random.choice(["Norte", "Sur", "Este", "Oeste"], n),
    })


def test_instanciacion() -> None:
    print("\n[1] Probando instanciación de EvidentlyMonitor ...")
    m = EvidentlyMonitor(
        numerical_columns=["edad", "saldo", "puntaje"],
        categorical_columns=["sexo", "region"],
        num_threshold=0.15,
        cat_threshold=0.10,
        n_muestra=100,
        out_dir=OUT_DIR,
    )
    assert m.numerical_columns == ["edad", "saldo", "puntaje"]
    assert m.categorical_columns == ["sexo", "region"]
    assert m.num_threshold == 0.15
    assert m.cat_threshold == 0.10
    assert m.n_muestra == 100
    assert m.all_columns == ["edad", "saldo", "puntaje", "sexo", "region"]
    print("    OK: instanciación correcta.")


def test_tabla_scores() -> None:
    print("\n[2] Probando tabla_scores() ...")
    m = EvidentlyMonitor(
        numerical_columns=["edad", "saldo"],
        categorical_columns=["sexo"],
        num_threshold=0.1,
        out_dir=OUT_DIR,
    )
    ref = hacer_datos(300, desplazamiento=0.0)
    cur = hacer_datos(300, desplazamiento=0.0)
    scores = m.tabla_scores(ref, cur)

    assert list(scores.columns) == [
        "variable", "tipo", "media_ref", "media_actual",
        "std_ref", "std_actual", "wasserstein_norm", "drift",
    ]
    assert len(scores) == 3  # 3 columnas monitoreadas
    assert list(scores["tipo"]) == ["numerica", "numerica", "categorica"]
    assert all(scores.loc[scores["tipo"] == "numerica", "drift"].isin([True, False]))
    assert scores.loc[scores["tipo"] == "categorica", "drift"].isna().all()
    print("    OK: tabla_scores devuelve DataFrame con columnas esperadas.")
    print(scores.round(4).to_string(index=False))


def test_comparar_sin_mlflow() -> None:
    print("\n[3] Probando comparar() con log_mlflow=False ...")
    m = EvidentlyMonitor(
        numerical_columns=["edad", "saldo", "puntaje"],
        categorical_columns=["sexo", "region"],
        num_threshold=0.10,
        cat_threshold=0.10,
        n_muestra=150,
        out_dir=OUT_DIR,
    )
    # Referencia sin desplazamiento, actual con desplazamiento para forzar drift.
    ref = hacer_datos(400, desplazamiento=0.0)
    cur = hacer_datos(400, desplazamiento=15.0)  # distribución diferente

    resumen = m.comparar(ref, cur, nombre="test_sintetico", log_mlflow=False)

    assert resumen["nombre"] == "test_sintetico"
    assert resumen["n_ref"] == 150
    assert resumen["n_actual"] == 150
    assert isinstance(resumen["vars_con_drift"], int)
    assert Path(resumen["html"]).exists()
    assert Path(resumen["scores_csv"]).exists()
    assert resumen["html"].endswith("drift_test_sintetico.html")
    assert resumen["scores_csv"].endswith("drift_test_sintetico_scores.csv")

    print(f"    OK: resumen = {resumen}")
    print(f"    HTML generado: {resumen['html']}")
    print(f"    CSV generado:  {resumen['scores_csv']}")


def test_comparar_con_drift_evidente() -> None:
    print("\n[4] Probando detección de drift con desplazamiento grande ...")
    m = EvidentlyMonitor(
        numerical_columns=["edad"],
        categorical_columns=[],
        num_threshold=0.10,
        n_muestra=200,
        out_dir=OUT_DIR,
    )
    ref = hacer_datos(500, desplazamiento=0.0)
    cur = hacer_datos(500, desplazamiento=50.0)  # desplazamiento masivo

    resumen = m.comparar(ref, cur, nombre="drift_evidente", log_mlflow=False)
    assert resumen["vars_con_drift"] >= 1, "Se esperaba al menos una variable con drift."
    print(f"    OK: vars_con_drift = {resumen['vars_con_drift']} (se esperaba >= 1).")


def test_dataframe_vacio_lanza_error() -> None:
    print("\n[5] Probando dataframe vacío lanza ValueError ...")
    m = EvidentlyMonitor(
        numerical_columns=["edad"],
        categorical_columns=[],
        out_dir=OUT_DIR,
    )
    try:
        m.comparar(pd.DataFrame(), pd.DataFrame({"edad": [1, 2]}), nombre="vacio")
        raise AssertionError("Se esperaba ValueError para dataframe vacío.")
    except ValueError as exc:
        assert "vacías" in str(exc)
        print(f"    OK: ValueError lanzado correctamente: {exc}")


def test_columnas_faltantes_lanzan_error() -> None:
    print("\n[6] Probando columnas faltantes lanza ValueError ...")
    m = EvidentlyMonitor(
        numerical_columns=["edad", "saldo"],
        categorical_columns=[],
        out_dir=OUT_DIR,
    )
    ref = pd.DataFrame({"edad": [1, 2, 3]})  # falta "saldo"
    cur = pd.DataFrame({"edad": [1, 2, 3], "saldo": [4, 5, 6]})
    try:
        m.comparar(ref, cur, nombre="faltantes")
        raise AssertionError("Se esperaba ValueError por columnas faltantes.")
    except ValueError as exc:
        assert "saldo" in str(exc)
        print(f"    OK: ValueError lanzado correctamente: {exc}")


def main() -> int:
    print("=" * 60)
    print("PRUEBAS DE EvidentlyMonitor (datos sintéticos, sin MLflow)")
    print("=" * 60)

    fallos = 0
    pruebas = [
        test_instanciacion,
        test_tabla_scores,
        test_comparar_sin_mlflow,
        test_comparar_con_drift_evidente,
        test_dataframe_vacio_lanza_error,
        test_columnas_faltantes_lanzan_error,
    ]

    for prueba in pruebas:
        try:
            prueba()
        except Exception as exc:  # noqa: BLE001
            fallos += 1
            print(f"    FALLO: {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    if fallos == 0:
        print(f"TODAS LAS PRUEBAS PASARON ({len(pruebas)} pruebas).")
    else:
        print(f"{fallos} prueba(s) fallaron de {len(pruebas)}.")
    print("=" * 60)

    return 1 if fallos > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
