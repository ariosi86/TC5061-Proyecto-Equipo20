#!/usr/bin/env python
"""
Pruebas de simulación de registros de EvidentlyMonitor.

Simula el registro completo de:
1. Proyecto (nombre, descripción, columnas, umbrales)
2. Datasets (referencia y actual, con estadísticas)
3. Reportes (HTML, CSV, variables con drift)

Todos los registros se guardan como JSON bajo
reports/drift/registro/ (o una carpeta temporal para las pruebas).

No requiere datos reales ni servidor MLflow. Valida que la clase
registre correctamente todo el flujo y que los archivos JSON se creen
con la estructura esperada.

Uso:
    python proyecto_mibici/monitoring/test_evidently_registro.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Asegura que se importe el paquete local.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proyecto_mibici.monitoring import EvidentlyMonitor

# Carpeta temporal de salida para las pruebas de registro.
OUT_DIR = Path("/tmp/opencode/test_evidently_registro")
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


def hacer_datos(n: int = 200, desplazamiento: float = 0.0) -> pd.DataFrame:
    """Genera un DataFrame sintético con columnas numéricas y categóricas."""
    return pd.DataFrame(
        {
            "edad": np.random.normal(40 + desplazamiento, 10, n),
            "saldo": np.random.exponential(1000 + desplazamiento * 50, n),
            "puntaje": np.random.uniform(0, 100 + desplazamiento * 10, n),
            "sexo": np.random.choice(["M", "F"], n),
            "region": np.random.choice(["Norte", "Sur", "Este", "Oeste"], n),
        }
    )


def test_registro_completo() -> None:
    """Prueba el flujo completo: registrar proyecto, datasets y reporte."""
    print("\n[1] Probando registro completo (proyecto + datasets + reporte) ...")
    m = EvidentlyMonitor(
        numerical_columns=["edad", "saldo", "puntaje"],
        categorical_columns=["sexo", "region"],
        num_threshold=0.10,
        cat_threshold=0.10,
        n_muestra=150,
        out_dir=OUT_DIR,
        registrar_local=True,
    )

    ref = hacer_datos(400, desplazamiento=0.0)
    cur = hacer_datos(400, desplazamiento=15.0)

    resumen = m.comparar(
        ref,
        cur,
        nombre="test_registro_completo",
        log_mlflow=False,
        registrar_proyecto_nombre="proyecto-prueba-drift",
        registrar_proyecto_descripcion="Prueba de simulación de registros de EvidentlyMonitor",
    )

    # Verificar claves del resumen
    for clave in [
        "nombre",
        "n_ref",
        "n_actual",
        "vars_con_drift",
        "html",
        "scores_csv",
        "proyecto_id",
        "proyecto_nombre",
        "dataset_referencia_id",
        "dataset_actual_id",
        "reporte_id",
    ]:
        assert clave in resumen, f"Falta la clave '{clave}' en el resumen"

    assert resumen["nombre"] == "test_registro_completo"
    assert resumen["n_ref"] == 150
    assert resumen["n_actual"] == 150
    assert resumen["proyecto_nombre"] == "proyecto-prueba-drift"
    assert resumen["proyecto_id"] is not None
    assert resumen["dataset_referencia_id"] is not None
    assert resumen["dataset_actual_id"] is not None
    assert resumen["reporte_id"] is not None
    assert Path(resumen["html"]).exists()
    assert Path(resumen["scores_csv"]).exists()

    # Verificar que los archivos JSON de registro existen
    registro_dir = OUT_DIR / "registro"
    proyectos = list((registro_dir / "proyectos").glob("*.json"))
    datasets = list((registro_dir / "datasets").glob("*.json"))
    reportes = list((registro_dir / "reportes").glob("*.json"))

    assert len(proyectos) >= 1, f"Se esperaba al menos 1 proyecto, se encontraron {len(proyectos)}"
    assert len(datasets) >= 2, f"Se esperaban al menos 2 datasets, se encontraron {len(datasets)}"
    assert len(reportes) >= 1, f"Se esperaba al menos 1 reporte, se encontraron {len(reportes)}"

    # Verificar contenido del JSON del proyecto (buscar por nombre)
    archivo_proyecto = None
    for p in proyectos:
        datos = json.loads(p.read_text(encoding="utf-8"))
        if datos.get("nombre") == "proyecto-prueba-drift":
            archivo_proyecto = p
            datos_proyecto = datos
            break
    assert archivo_proyecto is not None, "No se encontró el proyecto 'proyecto-prueba-drift'"
    assert datos_proyecto["nombre"] == "proyecto-prueba-drift"
    assert datos_proyecto["descripcion"] == "Prueba de simulación de registros de EvidentlyMonitor"
    assert datos_proyecto["numerical_columns"] == ["edad", "saldo", "puntaje"]
    assert datos_proyecto["categorical_columns"] == ["sexo", "region"]
    assert datos_proyecto["num_threshold"] == 0.10
    assert datos_proyecto["evidently_api_url"] == "http://127.0.0.1:8000"
    assert datos_proyecto["registrado_en_evidently"] is True

    # Verificar contenido del JSON de un dataset
    datos_dataset = json.loads(datasets[0].read_text(encoding="utf-8"))
    assert datos_dataset["tipo"] in ("referencia", "actual")
    assert "estadisticas" in datos_dataset
    assert "filas" in datos_dataset["estadisticas"]
    assert "proyecto_id" in datos_dataset

    # Verificar contenido del JSON de un reporte (buscar por nombre)
    datos_reporte = None
    for f in reportes:
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("nombre") == "test_registro_completo":
            datos_reporte = d
            break
    assert datos_reporte is not None, "No se encontró el reporte 'test_registro_completo'"
    assert datos_reporte["nombre"] == "test_registro_completo"
    assert datos_reporte["vars_con_drift"] == resumen["vars_con_drift"]
    assert datos_reporte["n_ref"] == 150
    assert datos_reporte["n_actual"] == 150
    assert Path(datos_reporte["html_path"]).exists()
    assert Path(datos_reporte["scores_csv"]).exists()

    print(
        f"    OK: proyecto={len(proyectos)} archivo(s), datasets={len(datasets)} archivo(s), reportes={len(reportes)} archivo(s)"
    )
    print(f"    Resumen: {resumen}")


def test_registro_sin_proyecto() -> None:
    """Prueba comparar() sin registrar proyecto (solo datasets y reporte)."""
    print("\n[2] Probando registro sin proyecto (solo datasets + reporte) ...")
    m = EvidentlyMonitor(
        numerical_columns=["edad"],
        categorical_columns=[],
        num_threshold=0.10,
        n_muestra=100,
        out_dir=OUT_DIR,
        registrar_local=True,
    )

    ref = hacer_datos(300, desplazamiento=0.0)
    cur = hacer_datos(300, desplazamiento=5.0)

    resumen = m.comparar(
        ref, cur, nombre="test_sin_proyecto", log_mlflow=False,
        registrar_proyecto_nombre=None,
    )

    assert resumen["proyecto_id"] is None
    assert resumen["proyecto_nombre"] is None
    assert resumen["dataset_referencia_id"] is not None
    assert resumen["dataset_actual_id"] is not None
    assert resumen["reporte_id"] is not None

    # Verificar que no hay archivo de proyecto creado para este monitor
    # (pero sí pueden haber de pruebas anteriores)
    print(f"    OK: proyecto_id={resumen['proyecto_id']}, reporte_id={resumen['reporte_id']}")


def test_estructura_json_proyecto() -> None:
    """Valida la estructura exacta del JSON de proyecto."""
    print("\n[3] Probando estructura JSON de proyecto ...")
    m = EvidentlyMonitor(
        numerical_columns=["col1", "col2"],
        categorical_columns=["cat1"],
        num_threshold=0.15,
        cat_threshold=0.05,
        n_muestra=50,
        out_dir=OUT_DIR,
        registrar_local=True,
    )

    m.registrar_proyecto("mi-proyecto", descripcion="Descripción de prueba")

    archivo_proyecto = m.proyectos_dir / "mi-proyecto.json"
    assert archivo_proyecto.exists(), f"No se encontró {archivo_proyecto}"

    datos = json.loads(archivo_proyecto.read_text(encoding="utf-8"))
    claves_esperadas = [
        "id",
        "nombre",
        "descripcion",
        "numerical_columns",
        "categorical_columns",
        "num_method",
        "cat_method",
        "num_threshold",
        "cat_threshold",
        "n_muestra",
        "random_seed",
        "fecha_registro",
        "evidently_api_url",
        "registrado_en_evidently",
    ]
    for clave in claves_esperadas:
        assert clave in datos, f"Falta clave '{clave}' en el JSON del proyecto"

    assert datos["nombre"] == "mi-proyecto"
    assert datos["descripcion"] == "Descripción de prueba"
    assert datos["numerical_columns"] == ["col1", "col2"]
    assert datos["categorical_columns"] == ["cat1"]
    assert datos["num_threshold"] == 0.15
    assert datos["cat_threshold"] == 0.05
    assert datos["evidently_api_url"] == "http://127.0.0.1:8000"
    assert datos["registrado_en_evidently"] is True

    print(f"    OK: estructura JSON válida con {len(claves_esperadas)} claves")


def test_estructura_json_dataset() -> None:
    """Valida la estructura exacta del JSON de dataset."""
    print("\n[4] Probando estructura JSON de dataset ...")
    m = EvidentlyMonitor(
        numerical_columns=["edad"],
        categorical_columns=["sexo"],
        num_threshold=0.10,
        out_dir=OUT_DIR,
        registrar_local=True,
    )

    df = hacer_datos(100, desplazamiento=0.0)
    dataset_id = m.registrar_dataset(df, "datos-referencia", tipo="referencia")

    archivo_dataset = m.datasets_dir / f"{dataset_id}.json"
    assert archivo_dataset.exists()

    datos = json.loads(archivo_dataset.read_text(encoding="utf-8"))
    claves_esperadas = [
        "id",
        "nombre",
        "tipo",
        "proyecto_id",
        "proyecto_nombre",
        "columnas_monitoreadas",
        "estadisticas",
        "fecha_registro",
    ]
    for clave in claves_esperadas:
        assert clave in datos, f"Falta clave '{clave}' en el JSON del dataset"

    assert datos["nombre"] == "datos-referencia"
    assert datos["tipo"] == "referencia"
    assert datos["columnas_monitoreadas"] == ["edad", "sexo"]
    assert datos["estadisticas"]["filas"] == 100
    assert "media_edad" in datos["estadisticas"]
    assert "valores_unicos_sexo" in datos["estadisticas"]

    print(f"    OK: estructura JSON válida con {len(claves_esperadas)} claves")


def test_estructura_json_reporte() -> None:
    """Valida la estructura exacta del JSON de reporte."""
    print("\n[5] Probando estructura JSON de reporte ...")
    m = EvidentlyMonitor(
        numerical_columns=["edad", "saldo"],
        categorical_columns=["sexo"],
        num_threshold=0.10,
        out_dir=OUT_DIR,
        registrar_local=True,
    )

    m.registrar_proyecto("proy-reporte", descripcion="Para probar reporte")
    ref = hacer_datos(200, desplazamiento=0.0)
    cur = hacer_datos(200, desplazamiento=10.0)

    resumen = m.comparar(
        ref,
        cur,
        nombre="test_reporte_json",
        log_mlflow=False,
        registrar_proyecto_nombre="proy-reporte",
    )

    # Buscar el archivo de reporte por nombre y proyecto (el ID puede ser el de Evidently)
    archivo_reporte = None
    for f in m.reportes_dir.glob("*.json"):
        datos = json.loads(f.read_text(encoding="utf-8"))
        if datos.get("nombre") == "test_reporte_json" and datos.get("proyecto_nombre") == "proy-reporte":
            archivo_reporte = f
            break
    assert archivo_reporte is not None, "No se encontró el archivo de reporte 'test_reporte_json'"

    datos = json.loads(archivo_reporte.read_text(encoding="utf-8"))
    claves_esperadas = [
        "id",
        "nombre",
        "proyecto_id",
        "proyecto_nombre",
        "html_path",
        "scores_csv",
        "vars_con_drift",
        "n_ref",
        "n_actual",
        "num_threshold",
        "cat_threshold",
        "fecha_registro",
    ]
    for clave in claves_esperadas:
        assert clave in datos, f"Falta clave '{clave}' en el JSON del reporte"

    assert datos["nombre"] == "test_reporte_json"
    assert datos["proyecto_nombre"] == "proy-reporte"
    assert datos["vars_con_drift"] == resumen["vars_con_drift"]
    assert Path(datos["html_path"]).exists()
    assert Path(datos["scores_csv"]).exists()

    print(f"    OK: estructura JSON válida con {len(claves_esperadas)} claves")


def test_consola_evidently_no_disponible_registra_localmente() -> None:
    """Verifica que, si la API de Evidently no está disponible, el registro local funciona igual."""
    print("\n[6] Probando fallback: API de Evidently no disponible -> registro local ...")
    m = EvidentlyMonitor(
        numerical_columns=["edad"],
        categorical_columns=[],
        num_threshold=0.10,
        out_dir=OUT_DIR,
        registrar_local=True,
        usar_api_evidently=False,  # Forzar modo local sin API
    )

    assert m._evidently_disponible is False
    assert m._ws is None

    try:
        pid = m.registrar_proyecto("proyecto-sin-api", descripcion="Sin API de Evidently")
        assert pid is not None
        assert pid.startswith("local-"), f"Se esperaba ID local, se obtuvo {pid}"

        archivo = m.proyectos_dir / "proyecto-sin-api.json"
        assert archivo.exists()
        datos = json.loads(archivo.read_text(encoding="utf-8"))
        assert datos["registrado_en_evidently"] is False
    finally:
        pass

    print(f"    OK: registro local funcionó sin API de Evidently (id={pid})")


def main() -> int:
    print("=" * 60)
    print("PRUEBAS DE SIMULACIÓN DE REGISTROS (EvidentlyMonitor)")
    print("=" * 60)

    fallos = 0
    pruebas = [
        test_registro_completo,
        test_registro_sin_proyecto,
        test_estructura_json_proyecto,
        test_estructura_json_dataset,
        test_estructura_json_reporte,
        test_consola_evidently_no_disponible_registra_localmente,
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
