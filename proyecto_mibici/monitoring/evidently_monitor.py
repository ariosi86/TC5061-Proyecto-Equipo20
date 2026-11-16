#!/usr/bin/env python
"""
Clase reutilizable de monitoreo de drift con Evidently (Entregable 8).

Diseñada para que cualquier integrante del equipo la instancie con sus
propios conjuntos de referencia/actual y columnas, sin depender de P1.
No borra ni reemplaza nada de las entregas anteriores.

La clase incluye una capa de registro que usa la API real de Evidently Platform
para registrar:
- El proyecto de monitoreo (nombre, descripción, columnas, umbrales).
- Los datasets usados (referencia y actual, con estadísticas básicas).
- Los reportes generados (HTML, CSV, variables con drift, fecha).

Se conecta al servidor de Evidently UI en ``http://127.0.0.1:8000`` mediante
``RemoteWorkspace``. Si el servidor no está disponible, los registros se
guardan localmente como JSON bajo ``reports/drift/registro/`` para trazabilidad
y para el reporte PDF del entregable.

Ejemplo rápido (P1):
    from proyecto_mibici.monitoring import EvidentlyMonitor

    monitor = EvidentlyMonitor(
        numerical_columns=["lag_1h", "lag_24h", "lag_168h", "rolling_3h", "rolling_168h",
                           "target_hour_sin", "target_hour_cos"],
        categorical_columns=["target_is_weekend"],
        nombre_proyecto="p1-demanda-horaria",
    )
    resumen = monitor.comparar(df_referencia, df_actual, nombre="p1_ref2025_vs_oct2026")
    print(resumen)

Ejemplo genérico (P2 / P3 / cualquier otro):
    monitor = EvidentlyMonitor(
        numerical_columns=["edad", "saldo"],
        categorical_columns=["sexo", "region"],
        num_threshold=0.15,
        cat_threshold=0.10,
        nombre_proyecto="p2-riesgo-desequilibrio",
    )
    resumen = monitor.comparar(ref, cur, nombre="p2_ref_vs_actual", log_mlflow=False)
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import webbrowser

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from scipy.stats import chi2, wasserstein_distance

try:
    import matplotlib

    matplotlib.use("Agg")

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from sklearn.metrics import auc, f1_score, mean_tweedie_deviance, precision_recall_curve

from proyecto_mibici.config import (
    EVIDENTLY_PROJECT_NAME,
    MLFLOW_EXPERIMENT_MONITOREO,
    P1_ACTUAL_FIN,
    P1_FEATURES_V2,
    P1_TARGET,
    P1_TEST_FIN,
    P1_TEST_INICIO,
    P2_INICIO_VALIDACION,
    P2_OBJETIVO,
    P2_VARIABLES,
    P3_VARIABLES,
    RANDOM_SEED,
    REPORTS_DIR,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.monitoring.datos_p2 import (
    FIN_ACTUAL as P2_FIN_ACTUAL,
    INICIO_ACTUAL as P2_INICIO_ACTUAL,
    construir_serie as construir_serie_p2,
)

# URL del servidor de Evidently UI (API de la plataforma, no solo lectura del frontend).
EVIDENTLY_API_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------- #
# Funciones del A/B testing de P3 (clustering), usadas por ab_test_p3()
# ---------------------------------------------------------------------- #
def metricas_internas(X: np.ndarray, etiquetas: np.ndarray) -> dict:
    """Silhouette, Davies-Bouldin y Calinski-Harabasz de una asignación de grupos."""
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    if len(np.unique(etiquetas)) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return {
        "silhouette": float(silhouette_score(X, etiquetas)),
        "davies_bouldin": float(davies_bouldin_score(X, etiquetas)),
        "calinski_harabasz": float(calinski_harabasz_score(X, etiquetas)),
    }


def emparejar_grupos(centros_a: np.ndarray, centros_b: np.ndarray) -> dict:
    """Mapa grupo_b -> grupo_a que minimiza la distancia entre centroides (Hungarian)."""
    from scipy.optimize import linear_sum_assignment

    costo = ((centros_a[:, None, :] - centros_b[None, :, :]) ** 2).sum(axis=2)
    filas, columnas = linear_sum_assignment(costo)
    return {int(b): int(a) for a, b in zip(filas, columnas)}


def bootstrap_diferencias(X, etiquetas_a, etiquetas_b, n_boot: int, semilla: int) -> pd.DataFrame:
    """Remuestrea estaciones y calcula B - A en cada métrica sobre la misma muestra."""
    rng = np.random.default_rng(semilla)
    n = len(X)
    filas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m_a = metricas_internas(X[idx], etiquetas_a[idx])
        m_b = metricas_internas(X[idx], etiquetas_b[idx])
        filas.append({k: m_b[k] - m_a[k] for k in m_a})
    return pd.DataFrame(filas).dropna()


def proporcion_cambios(pipeline, perfil_entrenamiento: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """Estaciones comunes que el mismo modelo asigna a otro grupo en el periodo actual."""
    comunes = perfil_entrenamiento.index.intersection(actual.index)
    antes = pipeline.predict(perfil_entrenamiento.loc[comunes, P3_VARIABLES])
    despues = pipeline.predict(actual.loc[comunes, P3_VARIABLES])
    return {
        "estaciones_comunes": len(comunes),
        "cambian_de_grupo": int((antes != despues).sum()),
        "pct_cambian": float(100 * (antes != despues).mean()),
    }


def decidir_p3(ic_silhouette: tuple, delta_db: float, min_prop_b: float,
               min_proporcion_grupo: float = 0.05) -> tuple:
    """Regla de P3 fijada antes de ver resultados. Devuelve (veredicto, cumple)."""
    cumple = ic_silhouette[0] > 0 and delta_db <= 0 and min_prop_b >= min_proporcion_grupo
    if ic_silhouette[1] < 0:
        return "MANTENER CHAMPION (el challenger es peor)", False
    if cumple:
        return "PROMOVER CHALLENGER", True
    return "MANTENER CHAMPION (sin diferencia clara a favor del challenger)", False


def cargar_estimador(modelo_nombre: str, version) -> object:
    """Carga un modelo registrado como sklearn o, si se guardó con el envoltorio pyfunc
    del proyecto (tracking.py), el estimador de artifacts/modelo.pkl."""
    import joblib
    from mlflow.exceptions import MlflowException

    uri = f"models:/{modelo_nombre}/{version}"
    try:
        return mlflow.sklearn.load_model(uri)
    except MlflowException:
        ruta = mlflow.artifacts.download_artifacts(artifact_uri=uri)
        return joblib.load(os.path.join(ruta, "artifacts", "modelo.pkl"))


class EvidentlyMonitor:
    """Orquesta la comparación de drift de datos con Evidently y registra
    proyecto, datasets y reportes en la API de Evidently Platform.

    Se conecta al servidor de Evidently UI en ``http://127.0.0.1:8000`` mediante
    ``RemoteWorkspace``. Si el servidor no está disponible, los registros se
    guardan localmente como JSON bajo ``out_dir / "registro"``.

    Parametros
    ----------
    numerical_columns : iterable de str
        Columnas numéricas a comparar.
    categorical_columns : iterable de str, opcional
        Columnas categóricas/binarias a comparar.
    num_method : str
        Método de drift numérico de Evidently ("wasserstein", "ks", etc.).
    cat_method : str
        Método de drift categórico de Evidently ("jensenshannon", etc.).
    num_threshold : float
        Umbral de drift numérico (para Evidently y para la tabla de scores).
    cat_threshold : float
        Umbral de drift categórico (para Evidently).
    n_muestra : int
        Filas máximas por conjunto (Evidently es costoso con millones de filas).
    random_seed : int
        Semilla para el muestreo aleatorio.
    out_dir : str | Path | None
        Carpeta donde se guardan el HTML de Evidently y el CSV de scores.
        Por defecto: ``REPORTS_DIR / "drift"``.
    abrir_navegador : bool
        Si True, abre automáticamente el HTML generado en el navegador web
        después de crear el reporte. Por defecto: ``False``.
    registrar_local : bool
        Si True, guarda metadatos de proyecto, datasets y reportes como JSON
        bajo ``out_dir / "registro"`` (respaldo/trazabilidad). Por defecto: ``True``.
    usar_api_evidently : bool
        Si True, intenta registrar en la API de Evidently Platform en
        ``http://127.0.0.1:8000``. Si False o el servidor no está disponible,
        solo se usa registro local JSON. Por defecto: ``True``.
    """

    def __init__(
        self,
        numerical_columns: Iterable[str],
        categorical_columns: Iterable[str] | None = None,
        num_method: str = "wasserstein",
        cat_method: str = "jensenshannon",
        num_threshold: float = 0.1,
        cat_threshold: float = 0.1,
        n_muestra: int = 100_000,
        random_seed: int = RANDOM_SEED,
        out_dir: str | Path | None = None,
        abrir_navegador: bool = False,
        registrar_local: bool = True,
        usar_api_evidently: bool = True,
        nombre_proyecto: str | None = None,
        descripcion_proyecto: str | None = None,
    ) -> None:
        self.numerical_columns = list(numerical_columns)
        self.categorical_columns = list(categorical_columns or [])
        self.num_method = num_method
        self.cat_method = cat_method
        self.num_threshold = num_threshold
        self.cat_threshold = cat_threshold
        self.n_muestra = n_muestra
        self.random_seed = random_seed
        self.out_dir = Path(out_dir) if out_dir is not None else REPORTS_DIR / "drift"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.abrir_navegador = abrir_navegador
        self.registrar_local = registrar_local
        self.usar_api_evidently = usar_api_evidently
        self.nombre_proyecto = nombre_proyecto
        self.descripcion_proyecto = descripcion_proyecto

        # Columnas que Evidently va a usar: todas las que pidió el usuario.
        self.all_columns = self.numerical_columns + self.categorical_columns

        # Carpetas de registro local
        self.registro_dir = self.out_dir / "registro"
        self.proyectos_dir = self.registro_dir / "proyectos"
        self.datasets_dir = self.registro_dir / "datasets"
        self.reportes_dir = self.registro_dir / "reportes"
        if self.registrar_local:
            self.proyectos_dir.mkdir(parents=True, exist_ok=True)
            self.datasets_dir.mkdir(parents=True, exist_ok=True)
            self.reportes_dir.mkdir(parents=True, exist_ok=True)

        # ID del proyecto (se asigna al registrar o al consultar la API de Evidently)
        self.proyecto_id: str | None = None
        self.proyecto_nombre: str | None = None

        # Workspace de Evidently (se inicializa en _conectar_evidently)
        self._ws = None
        self._evidently_disponible = False

        if self.usar_api_evidently:
            self._conectar_evidently()

        logger.info(
            f"EvidentlyMonitor listo | num={len(self.numerical_columns)} | cat={len(self.categorical_columns)} | umbral_num={self.num_threshold} | "
            f"umbral_cat={self.cat_threshold} | muestra={self.n_muestra} | out={self.out_dir} | registro={self.registrar_local} | api={self._evidently_disponible}"
        )

    # ------------------------------------------------------------------ #
    # Conexión a la API de Evidently Platform
    # ------------------------------------------------------------------ #
    def _conectar_evidently(self) -> None:
        """Intenta conectar con el servidor de Evidently UI en el puerto 8000."""
        try:
            from evidently.ui.workspace import RemoteWorkspace

            self._ws = RemoteWorkspace(base_url=EVIDENTLY_API_URL, verify=True)
            self._evidently_disponible = True
            logger.info(f"Conectado a Evidently Platform en {EVIDENTLY_API_URL}")
        except ImportError:
            logger.warning(
                "No se pudo importar RemoteWorkspace de evidently.ui.workspace. "
                "Se usará solo registro local JSON."
            )
            self._evidently_disponible = False
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"No se pudo conectar a Evidently Platform en {EVIDENTLY_API_URL}: {exc}. "
                "Se usará solo registro local JSON."
            )
            self._evidently_disponible = False

    # ------------------------------------------------------------------ #
    # Utilidades internas
    # ------------------------------------------------------------------ #
    @staticmethod
    def _muestrear(df: pd.DataFrame, n: int, random_seed: int) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        return df.sample(n=min(n, len(df)), random_state=random_seed).reset_index(drop=True)

    def _definicion(self) -> DataDefinition:
        return DataDefinition(
            numerical_columns=self.numerical_columns,
            categorical_columns=self.categorical_columns,
        )

    def _dataset_evidently(self, df: pd.DataFrame) -> Dataset:
        """Crea un objeto Dataset de Evidently para subir a la plataforma."""
        definicion = self._definicion()
        return Dataset.from_pandas(df[self.all_columns], data_definition=definicion)

    def _estadisticas_dataset(self, df: pd.DataFrame) -> dict:
        """Estadísticas básicas de un dataset para el registro."""
        stats = {
            "filas": len(df),
            "columnas": len(df.columns),
            "columnas_monitoreadas": list(self.all_columns),
            "fecha_registro": datetime.now(timezone.utc).isoformat(),
        }
        for c in self.numerical_columns:
            if c in df.columns:
                stats[f"media_{c}"] = float(df[c].mean())
                stats[f"std_{c}"] = float(df[c].std())
                stats[f"min_{c}"] = float(df[c].min())
                stats[f"max_{c}"] = float(df[c].max())
        for c in self.categorical_columns:
            if c in df.columns:
                stats[f"valores_unicos_{c}"] = int(df[c].nunique())
                stats[f"top_valor_{c}"] = (
                    str(df[c].mode().iloc[0]) if not df[c].mode().empty else None
                )
        return stats

    def _guardar_json(self, ruta: Path, datos: dict) -> None:
        """Guarda un diccionario como JSON con formato legible."""
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Registro de proyecto (API de Evidently + JSON local)
    # ------------------------------------------------------------------ #
    def registrar_proyecto(self, nombre: str, descripcion: str | None = None) -> str:
        """Registra (o recupera) el proyecto de monitoreo en Evidently Platform.

        Intenta crear o encontrar el proyecto en el servidor de Evidently UI
        mediante la API de RemoteWorkspace. Además guarda metadatos locales
        como JSON para trazabilidad.

        Parametros
        ----------
        nombre : str
            Nombre del proyecto (ej. "p1-demanda-horaria").
        descripcion : str | None
            Descripción opcional del proyecto.

        Returns
        -------
        str
            ID del proyecto (de Evidently Platform o generado localmente).
        """
        self.proyecto_nombre = nombre

        # 1. Intentar registrar en la API de Evidently Platform
        if self._evidently_disponible and self._ws is not None:
            try:
                # Buscar si ya existe un proyecto con este nombre
                existentes = self._ws.search_project(nombre)
                if existentes:
                    self.proyecto_id = str(existentes[0].id)
                    logger.info(
                        f"Proyecto existente en Evidently: {nombre} (id={self.proyecto_id})"
                    )
                else:
                    project_model = self._ws.create_project(name=nombre, description=descripcion)
                    self.proyecto_id = str(project_model.id)
                    logger.success(
                        f"Proyecto creado en Evidently: {nombre} (id={self.proyecto_id})"
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"No se pudo registrar el proyecto en Evidently Platform: {exc}")
                self.proyecto_id = None

        # 2. Fallback: ID generado localmente si no se pudo en la API
        if self.proyecto_id is None:
            self.proyecto_id = (
                f"local-{nombre}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            )

        # 3. Guardar metadatos locales (respaldo para el reporte PDF)
        if self.registrar_local:
            datos_proyecto = {
                "id": self.proyecto_id,
                "nombre": nombre,
                "descripcion": descripcion,
                "numerical_columns": self.numerical_columns,
                "categorical_columns": self.categorical_columns,
                "num_method": self.num_method,
                "cat_method": self.cat_method,
                "num_threshold": self.num_threshold,
                "cat_threshold": self.cat_threshold,
                "n_muestra": self.n_muestra,
                "random_seed": self.random_seed,
                "fecha_registro": datetime.now(timezone.utc).isoformat(),
                "evidently_api_url": EVIDENTLY_API_URL,
                "registrado_en_evidently": self._evidently_disponible,
            }
            self._guardar_json(self.proyectos_dir / f"{nombre}.json", datos_proyecto)
            logger.success(f"Proyecto registrado localmente: {nombre} (id={self.proyecto_id})")

        return self.proyecto_id

    # ------------------------------------------------------------------ #
    # Registro de datasets (API de Evidently + JSON local)
    # ------------------------------------------------------------------ #
    def registrar_dataset(
        self,
        df: pd.DataFrame,
        nombre: str,
        tipo: str,  # "referencia" o "actual"
    ) -> str:
        """Registra un dataset (referencia o actual) en Evidently Platform y localmente.

        Sube el dataset al servidor de Evidently mediante ``ws.add_dataset()``
        y guarda metadatos locales como JSON para trazabilidad.

        Parametros
        ----------
        df : pd.DataFrame
            Dataset a registrar.
        nombre : str
            Nombre del dataset (ej. "ref2025", "actual_oct2026").
        tipo : str
            Tipo de dataset: "referencia" o "actual".

        Returns
        -------
        str
            ID del dataset (de Evidently Platform o generado localmente).
        """
        if tipo not in ("referencia", "actual"):
            raise ValueError(f'tipo debe ser "referencia" o "actual", se recibió "{tipo}"')

        # 1. Intentar subir a la API de Evidently Platform
        dataset_id_evidently: str | None = None
        if self._evidently_disponible and self._ws is not None and self.proyecto_id:
            try:
                ds_evidently = self._dataset_evidently(df)
                dataset_id_evidently = self._ws.add_dataset(
                    project_id=self.proyecto_id,
                    dataset=ds_evidently,
                    name=nombre,
                    description=f"Dataset {tipo} para monitoreo de drift",
                )
                dataset_id_evidently = str(dataset_id_evidently)
                logger.success(
                    f"Dataset subido a Evidently: {nombre} ({tipo}) | id={dataset_id_evidently}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"No se pudo subir el dataset a Evidently Platform: {exc}")

        # 2. ID local (siempre se genera para trazabilidad)
        dataset_id_local = f"{tipo}-{nombre}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        dataset_id = dataset_id_evidently or dataset_id_local

        # 3. Guardar metadatos locales (respaldo para el reporte PDF)
        if self.registrar_local:
            datos_dataset = {
                "id": dataset_id,
                "id_evidently": dataset_id_evidently,
                "nombre": nombre,
                "tipo": tipo,
                "proyecto_id": self.proyecto_id,
                "proyecto_nombre": self.proyecto_nombre,
                "columnas_monitoreadas": list(self.all_columns),
                "estadisticas": self._estadisticas_dataset(df),
                "fecha_registro": datetime.now(timezone.utc).isoformat(),
            }
            self._guardar_json(self.datasets_dir / f"{dataset_id_local}.json", datos_dataset)
            logger.success(
                f"Dataset registrado localmente: {nombre} ({tipo}) | {len(df):,} filas | id={dataset_id_local}"
            )

        return dataset_id

    # ------------------------------------------------------------------ #
    # Registro de reportes (API de Evidently + JSON local)
    # ------------------------------------------------------------------ #
    def registrar_reporte(
        self,
        nombre: str,
        html_path: Path,
        scores_csv: Path,
        vars_con_drift: int,
        n_ref: int,
        n_actual: int,
        snapshot: Snapshot | None = None,
    ) -> str:
        """Registra un reporte de drift en Evidently Platform y localmente.

        Sube el reporte (Snapshot de Evidently) al servidor mediante
        ``ws.add_run()`` y guarda metadatos locales como JSON.

        Parametros
        ----------
        nombre : str
            Nombre del reporte (ej. "p1_ref2025_vs_oct2026").
        html_path : Path
            Ruta al archivo HTML de Evidently.
        scores_csv : Path
            Ruta al CSV de scores complementario.
        vars_con_drift : int
            Número de variables numéricas con drift detectado.
        n_ref : int
            Filas del conjunto de referencia usadas.
        n_actual : int
            Filas del conjunto actual usadas.
        snapshot : Snapshot | None
            Objeto Snapshot de Evidently (resultado de report.run()). Si se
            proporciona y la API está disponible, se sube a Evidently Platform.

        Returns
        -------
        str
            ID del reporte (de Evidently Platform o generado localmente).
        """
        # 1. Intentar subir el reporte a la API de Evidently Platform
        reporte_id_evidently: str | None = None
        if (
            self._evidently_disponible
            and self._ws is not None
            and self.proyecto_id
            and snapshot is not None
        ):
            try:
                snapshot_ref = self._ws.add_run(
                    project_id=self.proyecto_id,
                    run=snapshot,
                    include_data=True,
                    name=nombre,
                )
                reporte_id_evidently = str(snapshot_ref.id)
                logger.success(
                    f"Reporte subido a Evidently: {nombre} | id={reporte_id_evidently} | url={snapshot_ref.url}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"No se pudo subir el reporte a Evidently Platform: {exc}")

        # 2. ID local (siempre se genera para trazabilidad)
        reporte_id_local = (
            f"reporte-{nombre}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        )
        reporte_id = reporte_id_evidently or reporte_id_local

        # 3. Guardar metadatos locales (respaldo para el reporte PDF)
        if self.registrar_local:
            datos_reporte = {
                "id": reporte_id,
                "id_evidently": reporte_id_evidently,
                "nombre": nombre,
                "proyecto_id": self.proyecto_id,
                "proyecto_nombre": self.proyecto_nombre,
                "html_path": str(html_path),
                "scores_csv": str(scores_csv),
                "vars_con_drift": int(vars_con_drift),
                "n_ref": int(n_ref),
                "n_actual": int(n_actual),
                "num_threshold": self.num_threshold,
                "cat_threshold": self.cat_threshold,
                "fecha_registro": datetime.now(timezone.utc).isoformat(),
            }
            self._guardar_json(self.reportes_dir / f"{reporte_id_local}.json", datos_reporte)
            logger.success(
                f"Reporte registrado localmente: {nombre} | vars_con_drift={vars_con_drift} | id={reporte_id_local}"
            )

        return reporte_id

    # ------------------------------------------------------------------ #
    # Utilidades
    # ------------------------------------------------------------------ #
    def abrir_en_navegador(self, html_path: Path | str) -> None:
        """Abre un archivo HTML de Evidently en el navegador web por defecto."""
        ruta = Path(html_path).resolve()
        url = ruta.as_uri()
        logger.info(f"Abriendo reporte en el navegador: {url}")
        webbrowser.open(url)

    def tabla_scores(self, referencia: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
        """Tabla complementaria (fuera de Evidently) útil para el reporte PDF.

        Calcula, por variable: medias, desviaciones estándares y una métrica de
        distancia normalizada (Wasserstein para numéricas). Marca ``drift=True``
        cuando excede ``num_threshold``. Para columnas categóricas deja las
        métricas numéricas en NaN (Evidently ya calcula Jensen-Shannon en el
        HTML).
        """
        filas = []
        for c in self.all_columns:
            es_numerica = c in self.numerical_columns
            fila = {
                "variable": c,
                "tipo": "numerica" if es_numerica else "categorica",
            }
            if es_numerica:
                sd = float(referencia[c].std())
                w_norm = (
                    float(wasserstein_distance(referencia[c], actual[c])) / sd
                    if sd > 0
                    else np.nan
                )
                fila.update(
                    {
                        "media_ref": float(referencia[c].mean()),
                        "media_actual": float(actual[c].mean()),
                        "std_ref": sd,
                        "std_actual": float(actual[c].std()),
                        "wasserstein_norm": w_norm,
                        "drift": bool(w_norm > self.num_threshold),
                    }
                )
            else:
                fila.update(
                    {
                        "media_ref": np.nan,
                        "media_actual": np.nan,
                        "std_ref": np.nan,
                        "std_actual": np.nan,
                        "wasserstein_norm": np.nan,
                        "drift": np.nan,
                    }
                )
            filas.append(fila)
        return pd.DataFrame(filas)

    # ------------------------------------------------------------------ #
    # API principal
    # ------------------------------------------------------------------ #
    def comparar(
        self,
        referencia: pd.DataFrame,
        actual: pd.DataFrame,
        nombre: str,
        *,
        abrir_navegador: bool | None = None,
        registrar_proyecto_nombre: str = EVIDENTLY_PROJECT_NAME,
        registrar_proyecto_descripcion: str | None = None,
        log_mlflow: bool = False,
        mlflow_run_name: str | None = None,
        experimento: str = MLFLOW_EXPERIMENT_MONITOREO,
    ) -> dict:
        """Ejecuta la comparación completa de drift con registro en Evidently.

        Pasos:
        1. Registra el proyecto en Evidently Platform (si se indica nombre).
        2. Registra los datasets de referencia y actual en Evidently Platform.
        3. Muestrea referencia y actual (si exceden ``n_muestra``).
        4. Genera el HTML de Evidently (data drift + data summary).
        5. (Opcional) Abre el HTML en el navegador web.
        6. Calcula la tabla de scores complementaria y la guarda como CSV.
        7. Registra el reporte (run) en Evidently Platform.
        8. Guarda metadatos locales JSON para trazabilidad.

        Parametros
        ----------
        referencia : pd.DataFrame
            Conjunto de referencia (periodo de entrenamiento).
        actual : pd.DataFrame
            Conjunto actual (periodo de producción / posterior).
        nombre : str
            Identificador del comparativo (se usa en nombres de archivo).
        abrir_navegador : bool | None
            Si True, abre el HTML en el navegador. Si None, usa el valor
            de ``self.abrir_navegador`` (por defecto False).
        registrar_proyecto_nombre : str
            Nombre del proyecto a registrar en Evidently.
            Por defecto usa ``EVIDENTLY_PROJECT_NAME`` ("proyecto_mibici_equipo20").
            Pasa ``None`` para no registrar proyecto.
        registrar_proyecto_descripcion : str | None
            Descripción del proyecto.

        Returns
        -------
        dict
            Resumen con claves: ``nombre``, ``n_ref``, ``n_actual``,
            ``vars_con_drift`` (solo numéricas), ``html`` (ruta del HTML),
            ``scores_csv`` (ruta del CSV), ``proyecto_id``,
            ``dataset_referencia_id``, ``dataset_actual_id``, ``reporte_id``.
        """
        if referencia.empty or actual.empty:
            raise ValueError("Referencia y actual no pueden estar vacías.")

        faltantes_ref = [c for c in self.all_columns if c not in referencia.columns]
        faltantes_cur = [c for c in self.all_columns if c not in actual.columns]
        if faltantes_ref or faltantes_cur:
            raise ValueError(
                f"Columnas faltantes: referencia={faltantes_ref} | actual={faltantes_cur}"
            )

        # 0. Registrar proyecto (si se pide)
        if registrar_proyecto_nombre:
            self.registrar_proyecto(
                registrar_proyecto_nombre, descripcion=registrar_proyecto_descripcion
            )

        # 1. Registrar datasets (referencia y actual) en Evidently Platform
        dataset_ref_id = self.registrar_dataset(referencia, f"ref-{nombre}", tipo="referencia")
        dataset_cur_id = self.registrar_dataset(actual, f"cur-{nombre}", tipo="actual")

        # 2. Muestrear para el reporte de Evidently
        ref_m = self._muestrear(referencia, self.n_muestra, self.random_seed)
        cur_m = self._muestrear(actual, self.n_muestra, self.random_seed)

        # 3. Generar reporte HTML de Evidently y obtener el Snapshot
        ds_ref = self._dataset_evidently(ref_m)
        ds_cur = self._dataset_evidently(cur_m)
        report = Report(
            [
                DataDriftPreset(
                    num_method=self.num_method,
                    num_threshold=self.num_threshold,
                    cat_method=self.cat_method,
                    cat_threshold=self.cat_threshold,
                ),
                DataSummaryPreset(),
            ]
        )
        snapshot = report.run(current_data=ds_cur, reference_data=ds_ref)
        salida_html = self.out_dir / f"drift_{nombre}.html"
        snapshot.save_html(str(salida_html))
        logger.success(f"Reporte HTML de Evidently -> {salida_html}")

        # 4. Abrir en navegador (si se solicita)
        abrir = self.abrir_navegador if abrir_navegador is None else abrir_navegador
        if abrir:
            self.abrir_en_navegador(salida_html)

        # 5. Tabla de scores complementaria
        scores = self.tabla_scores(ref_m, cur_m)
        scores_csv = self.out_dir / f"drift_{nombre}_scores.csv"
        scores.to_csv(scores_csv, index=False)
        logger.info("\n" + scores.round(4).to_string(index=False))

        vars_con_drift = int(
            scores.loc[scores["tipo"] == "numerica", "drift"].fillna(False).astype(bool).sum()
        )
        logger.info(
            f"Variables con drift (numéricas, umbral={self.num_threshold}): {vars_con_drift}"
        )

        # 6. Registrar reporte en Evidently Platform (subir el Snapshot)
        reporte_id = self.registrar_reporte(
            nombre=nombre,
            html_path=salida_html,
            scores_csv=scores_csv,
            vars_con_drift=vars_con_drift,
            n_ref=len(ref_m),
            n_actual=len(cur_m),
            snapshot=snapshot,
        )

        resumen = {
            "nombre": nombre,
            "n_ref": len(ref_m),
            "n_actual": len(cur_m),
            "vars_con_drift": vars_con_drift,
            "html": str(salida_html),
            "scores_csv": str(scores_csv),
            "proyecto_id": self.proyecto_id,
            "proyecto_nombre": self.proyecto_nombre,
            "dataset_referencia_id": dataset_ref_id,
            "dataset_actual_id": dataset_cur_id,
            "reporte_id": reporte_id,
        }

        if log_mlflow:
            mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
            mlflow.set_experiment(experimento)
            with mlflow.start_run(run_name=mlflow_run_name or f"drift_{nombre}"):
                mlflow.log_params({
                    "num_method": self.num_method,
                    "cat_method": self.cat_method,
                    "num_threshold": self.num_threshold,
                    "cat_threshold": self.cat_threshold,
                    "n_ref": len(ref_m),
                    "n_actual": len(cur_m),
                    "nombre": nombre,
                })
                mlflow.log_metric("vars_con_drift", float(vars_con_drift))
                mlflow.log_artifact(str(salida_html), artifact_path=f"drift_{nombre}")
                mlflow.log_artifact(str(scores_csv), artifact_path=f"drift_{nombre}")
            logger.success(f"Run de MLflow registrado en el experimento {experimento}.")

        return resumen

    # ------------------------------------------------------------------ #
    # Guardado de resultados del A/B testing (CSV + corrida de MLflow)
    # ------------------------------------------------------------------ #
    def guardar_ab(self, pregunta: str, resumen: dict, experimento: str) -> Path:
        """Guarda el resumen de un A/B testing en reports/ab_test/<pregunta>/ y lo registra en MLflow.

        Aplana el diccionario que devuelven ab_test_p1/p2/p3 (métricas, intervalos y
        criterios) en una tabla clave-valor, la guarda como CSV y abre una corrida en
        el experimento de monitoreo con las métricas numéricas, el veredicto como tag
        y el CSV como artefacto.
        """
        def aplanar(valor, prefijo=""):
            if isinstance(valor, dict):
                for k, v in valor.items():
                    yield from aplanar(v, f"{prefijo}{k}." if prefijo or k else "")
            elif isinstance(valor, (list, tuple)) and all(np.isscalar(x) for x in valor):
                for i, v in enumerate(valor):
                    yield f"{prefijo}{i}", v
            else:
                yield prefijo.rstrip("."), valor

        filas = [{"clave": k, "valor": v} for k, v in aplanar(resumen)]
        salida = REPORTS_DIR / "ab_test" / pregunta.lower()
        salida.mkdir(parents=True, exist_ok=True)
        csv = salida / "ab_resumen.csv"
        pd.DataFrame(filas).to_csv(csv, index=False)

        numericas = {}
        for f in filas:
            v = f["valor"]
            if isinstance(v, (bool, np.bool_)):
                v = float(v)
            if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
                clave = "".join(c if c.isalnum() or c in "._-/ " else "_" for c in f["clave"])
                numericas[clave] = float(v)

        mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
        mlflow.set_experiment(experimento)
        with mlflow.start_run(run_name=f"{pregunta.lower()}_ab_test"):
            mlflow.set_tags({"question": pregunta.upper(), "stage_of_work": "ab_test",
                             "veredicto": str(resumen.get("veredicto", ""))})
            if numericas:
                mlflow.log_metrics(numericas)
            mlflow.log_artifact(str(csv), artifact_path=f"ab_test_{pregunta.lower()}")
        logger.success(f"A/B de {pregunta.upper()} guardado en {csv} y en el experimento {experimento}.")
        return csv

    # ------------------------------------------------------------------ #
    # A/B testing P1 (regresión)
    # ------------------------------------------------------------------ #
    def ab_test_p1(
        self,
        champion_version: int | None = None,
        challenger_version: int | None = None,
        *,
        min_mejora_mae_pct: float = 1.0,
        promover: bool = False,
        experimento: str = "mibici-equipo20-p1-monitoreo",
        modelo_nombre: str = "mibici-equipo20-p1-demanda",
        tweedie_power: float = 1.8,
    ) -> dict:
        """A/B testing offline de P1 (regresión): champion vs challenger.

        Compara dos versiones del modelo registrado en MLflow Model Registry
        sobre el mismo conjunto de evaluación (octubre 2026 para P1).

        Parametros
        ----------
        champion_version : int | None
            Versión del champion. Si None, usa el alias 'champion'.
        challenger_version : int | None
            Versión del challenger. Si None, usa el alias 'challenger'.
        min_mejora_mae_pct : float
            Mejora mínima de MAE (%) para considerar promover.
        promover : bool
            Si True y el challenger cumple los criterios, mueve el alias
            'champion' al challenger.
        experimento : str
            Experimento de MLflow donde registrar el A/B test.
        modelo_nombre : str
            Nombre del modelo registrado en MLflow Model Registry.
        tweedie_power : float
            Parámetro de potencia para la deviance de Tweedie.

        Returns
        -------
        dict
            Resumen con métricas de ambas versiones, cambio relativo,
            veredicto y criterios cumplidos.
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
        client = MlflowClient()

        # Resolver versiones
        if champion_version is None:
            va = client.get_model_version_by_alias(modelo_nombre, "champion")
        else:
            va = client.get_model_version(modelo_nombre, str(champion_version))
        if challenger_version is None:
            vb = client.get_model_version_by_alias(modelo_nombre, "challenger")
        else:
            vb = client.get_model_version(modelo_nombre, str(challenger_version))

        if va.version == vb.version:
            raise ValueError("Champion y challenger son la misma versión.")

        etiqueta_a = f"v{va.version} (champion)"
        etiqueta_b = f"v{vb.version} (challenger)"
        logger.info(f"A/B P1: {etiqueta_a} vs {etiqueta_b}")

        # 1. Conjunto de evaluación: octubre 2026, el mismo para ambas versiones
        from proyecto_mibici.features import construir_dataset_p1

        df_p1, _ = construir_dataset_p1()
        t = df_p1["target_hour"]
        df = df_p1[(t >= pd.Timestamp(P1_TEST_FIN)) & (t < pd.Timestamp(P1_ACTUAL_FIN))]
        df = df.dropna(subset=P1_FEATURES_V2 + [P1_TARGET]).copy()
        if df.empty:
            raise ValueError("No hay filas de octubre en la tabla de modelado de P1.")

        X = df[P1_FEATURES_V2]
        y = df[P1_TARGET].to_numpy(dtype=float)

        # 2. Predicciones de ambas versiones
        modelo_a = mlflow.pyfunc.load_model(f"models:/{modelo_nombre}/{va.version}")
        modelo_b = mlflow.pyfunc.load_model(f"models:/{modelo_nombre}/{vb.version}")
        pred_a = np.clip(np.asarray(modelo_a.predict(X), dtype=float).ravel(), 0, None)
        pred_b = np.clip(np.asarray(modelo_b.predict(X), dtype=float).ravel(), 0, None)

        # 3. Métricas globales
        def metricas_regresion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
            wape = float(np.abs(y_true - y_pred).sum() / y_true.sum())
            sesgo = float((y_pred - y_true).mean())
            deviance = float(
                mean_tweedie_deviance(y_true, np.clip(y_pred, 1e-6, None), power=tweedie_power)
            )
            return {
                "MAE": mae,
                "RMSE": rmse,
                "WAPE": wape,
                "Sesgo": sesgo,
                "Deviance_Tweedie": deviance,
            }

        m_a = metricas_regresion(y, pred_a)
        m_b = metricas_regresion(y, pred_b)

        # 4. Cambio relativo
        cambio = {
            k: (m_b[k] - m_a[k]) / abs(m_a[k]) * 100 if m_a[k] != 0 else float("nan") for k in m_a
        }

        # 5. Día a día y bootstrap IC95
        diario = (
            pd.DataFrame(
                {
                    "dia": df["target_hour"].dt.normalize().to_numpy(),
                    "mae_champion": np.abs(y - pred_a),
                    "mae_challenger": np.abs(y - pred_b),
                }
            )
            .groupby("dia")
            .mean()
        )
        diario["mejora"] = diario["mae_champion"] - diario["mae_challenger"]

        rng = np.random.default_rng(RANDOM_SEED)
        n_boot = 5000
        medias_boot = rng.choice(
            diario["mejora"].to_numpy(), size=(n_boot, len(diario)), replace=True
        ).mean(axis=1)
        ic_inf, ic_sup = (
            float(np.percentile(medias_boot, 2.5)),
            float(np.percentile(medias_boot, 97.5)),
        )

        # 6. Criterios de decisión
        mejora_mae = (m_a["MAE"] - m_b["MAE"]) / m_a["MAE"] * 100
        criterios = {
            f"MAE mejora >= {min_mejora_mae_pct}%": mejora_mae >= min_mejora_mae_pct,
            "RMSE no empeora": m_b["RMSE"] <= m_a["RMSE"],
            "IC95% mejora diaria > 0": ic_inf > 0,
            "Deviance Tweedie no empeora": m_b["Deviance_Tweedie"] <= m_a["Deviance_Tweedie"],
        }
        cumple = all(criterios.values())

        # 8. Logs
        logger.info(
            "\n" + pd.DataFrame([m_a, m_b], index=[etiqueta_a, etiqueta_b]).round(4).to_string()
        )
        logger.info(f"Mejora MAE: {mejora_mae:.2f}% | IC95%: [{ic_inf:.4f}, {ic_sup:.4f}]")
        logger.success(
            f"VEREDICTO: {'reemplazar con ' + etiqueta_b if cumple else 'mantener ' + etiqueta_a}"
        )

        # 9. Promoción
        if promover and cumple:
            client.set_registered_model_alias(modelo_nombre, "previous_champion", va.version)
            client.set_registered_model_alias(modelo_nombre, "champion", vb.version)
            logger.success(f"Alias 'champion' -> v{vb.version} (anterior: v{va.version}).")

        return {
            "champion_version": va.version,
            "challenger_version": vb.version,
            "champion_run_id": va.run_id,
            "challenger_run_id": vb.run_id,
            "metricas_champion": m_a,
            "metricas_challenger": m_b,
            "cambio_relativo_pct": cambio,
            "mejora_mae_pct": float(mejora_mae),
            "ic95_inf": ic_inf,
            "ic95_sup": ic_sup,
            "criterios": criterios,
            "cumple": cumple,
            "veredicto": "reemplazar_champion" if cumple else "mantener_champion",
            "dias_gana_challenger": int((diario["mejora"] > 0).sum()),
            "total_dias": len(diario),
        }

    # ------------------------------------------------------------------ #
    # A/B testing P3 (clustering)
    # ------------------------------------------------------------------ #
    def ab_test_p3(
        self,
        champion_version: int | None = None,
        challenger_version: int | None = None,
        *,
        n_boot: int = 1000,
        promover: bool = False,
        experimento: str = "mibici-equipo20-p3-monitoreo",
        modelo_nombre: str = "mibici-equipo20-p3-grupos",
        min_proporcion_grupo: float = 0.05,
    ) -> dict:
        """A/B testing de P3 (clustering): champion vs challenger.

        Compara dos versiones de KMeans registradas en MLflow Model Registry
        sobre el mismo conjunto de estaciones (octubre 2026 para P3).
        Usa métricas internas de clustering (silhouette, Davies-Bouldin,
        Calinski-Harabasz) con bootstrap por estación.

        Parametros
        ----------
        champion_version : int | None
            Versión del champion. Si None, usa el alias 'champion'.
        challenger_version : int | None
            Versión del challenger. Si None, usa el alias 'challenger'.
        n_boot : int
            Repeticiones del bootstrap por estación.
        promover : bool
            Si True y el challenger cumple los criterios, mueve el alias
            'champion' al challenger.
        experimento : str
            Experimento de MLflow donde registrar el A/B test.
        modelo_nombre : str
            Nombre del modelo registrado en MLflow Model Registry.
        min_proporcion_grupo : float
            Proporción mínima de estaciones por grupo para considerar válido.

        Returns
        -------
        dict
            Resumen con métricas de ambas versiones, cambio relativo,
            bootstrap IC95, veredicto y criterios cumplidos.
        """
        import tempfile

        import joblib
        from sklearn.metrics import adjusted_rand_score
        from sklearn.preprocessing import StandardScaler

        from proyecto_mibici.monitoring.datos_p3 import (
            PERIODOS_P3,
            cargar_conjunto,
        )

        mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
        client = MlflowClient()

        # Resolver versiones
        if champion_version is None:
            va = client.get_model_version_by_alias(modelo_nombre, "champion")
        else:
            va = client.get_model_version(modelo_nombre, str(champion_version))
        if challenger_version is None:
            vb = client.get_model_version_by_alias(modelo_nombre, "challenger")
        else:
            vb = client.get_model_version(modelo_nombre, str(challenger_version))

        if va.version == vb.version:
            raise ValueError("Champion y challenger son la misma versión.")

        etiqueta_a = f"A: champion (v{va.version})"
        etiqueta_b = f"B: challenger (v{vb.version})"
        logger.info(f"A/B P3: {etiqueta_a} vs {etiqueta_b}")

        # 1. Cargar pipelines desde MLflow
        def cargar_pipeline(mv):
            destino = tempfile.mkdtemp(prefix=f"p3_v{mv.version}_")
            ruta = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{mv.run_id}/model", dst_path=destino
            )
            archivo = os.path.join(ruta, "artifacts", "modelo.pkl")
            if not os.path.isfile(archivo):
                raise FileNotFoundError(
                    f"No se encontró artifacts/modelo.pkl en la versión {mv.version}"
                )
            return joblib.load(archivo)

        modelo_a = cargar_pipeline(va)
        modelo_b = cargar_pipeline(vb)

        # 2. Conjunto de evaluación: octubre 2026
        actual = cargar_conjunto("actual")
        X_actual = actual[P3_VARIABLES]

        # 3. Ambas versiones asignan grupo a las mismas estaciones
        grupos_a = modelo_a.predict(X_actual)
        grupos_b_crudos = modelo_b.predict(X_actual)

        # KMeans numera grupos de forma arbitraria; mapear B a A por centroide más cercano
        espacio = StandardScaler().fit(X_actual)
        X = espacio.transform(X_actual)
        centros_a = espacio.transform(
            pd.DataFrame(
                modelo_a.named_steps["scaler"].inverse_transform(
                    modelo_a.named_steps["clusterer"].cluster_centers_
                ),
                columns=P3_VARIABLES,
            )
        )
        centros_b = espacio.transform(
            pd.DataFrame(
                modelo_b.named_steps["scaler"].inverse_transform(
                    modelo_b.named_steps["clusterer"].cluster_centers_
                ),
                columns=P3_VARIABLES,
            )
        )
        mapa = emparejar_grupos(centros_a, centros_b)
        grupos_b = np.vectorize(mapa.get)(grupos_b_crudos)

        # 4. Métricas internas en el mismo espacio
        m_a = metricas_internas(X, grupos_a)
        m_b = metricas_internas(X, grupos_b)

        # 5. Bootstrap por estación
        boot = bootstrap_diferencias(X, grupos_a, grupos_b, n_boot, RANDOM_SEED)
        ic = {
            k: (float(boot[k].quantile(0.025)), float(boot[k].quantile(0.975)))
            for k in boot.columns
        }
        gana_b_pct = float((boot["silhouette"] > 0).mean() * 100)

        # 6. Estabilidad: estaciones que cambian de grupo
        referencia = cargar_conjunto("referencia")
        reciente = cargar_conjunto("reciente")
        cambios_a = proporcion_cambios(modelo_a, referencia, actual)
        cambios_b = proporcion_cambios(modelo_b, reciente, actual)

        # 7. Acuerdo entre versiones (ARI)
        acuerdo = float(adjusted_rand_score(grupos_a, grupos_b))

        # 8. Tamaño de grupos
        tamanos_a = pd.Series(grupos_a).value_counts().sort_index()
        tamanos_b = pd.Series(grupos_b).value_counts().sort_index()
        min_prop_a = float(tamanos_a.min() / len(actual))
        min_prop_b = float(tamanos_b.min() / len(actual))

        # 9. Criterios de decisión
        delta_db = m_b["davies_bouldin"] - m_a["davies_bouldin"]
        veredicto, cumple = decidir_p3(ic["silhouette"], delta_db, min_prop_b, min_proporcion_grupo)

        # 10. Logs
        logger.info("\nMétricas internas (octubre 2026):")
        logger.info(pd.DataFrame([m_a, m_b], index=[etiqueta_a, etiqueta_b]).round(4).to_string())
        logger.info("Diferencia B - A (bootstrap por estación, IC 95%):")
        for k, (lo, hi) in ic.items():
            logger.info(f"  {k:<18} {m_b[k] - m_a[k]:+.4f}  IC [{lo:+.4f}, {hi:+.4f}]")
        logger.info(f"B mejor en silhouette en {gana_b_pct:.1f}% de las repeticiones")
        logger.info(f"Acuerdo entre versiones (ARI): {acuerdo:.3f}")
        logger.success(f"Decisión: {veredicto}")

        # 11. Promoción

        # 12. Promoción
        if promover and cumple:
            client.set_registered_model_alias(modelo_nombre, "previous_champion", va.version)
            client.set_registered_model_alias(modelo_nombre, "champion", vb.version)
            logger.success(f"Alias 'champion' -> v{vb.version} (anterior: v{va.version}).")
        elif promover:
            logger.info("No se mueve el alias: la regla de decisión no lo indica.")

        return {
            "champion_version": va.version,
            "challenger_version": vb.version,
            "champion_run_id": va.run_id,
            "challenger_run_id": vb.run_id,
            "metricas_champion": m_a,
            "metricas_challenger": m_b,
            "cambio_relativo": {k: float(m_b[k] - m_a[k]) for k in m_a},
            "ic95": ic,
            "gana_b_pct": gana_b_pct,
            "acuerdo_ari": acuerdo,
            "cambios_champion": cambios_a,
            "cambios_challenger": cambios_b,
            "min_prop_champion": min_prop_a,
            "min_prop_challenger": min_prop_b,
            "criterios": {
                "IC95_silhouette > 0": ic["silhouette"][0] > 0,
                "Davies-Bouldin no empeora": delta_db <= 0,
                "Min proporcion grupo >= umbral": min_prop_b >= min_proporcion_grupo,
            },
            "cumple": cumple,
            "veredicto": veredicto,
        }

    # ------------------------------------------------------------------ #
    # A/B testing P2 (clasificación de riesgo)
    # ------------------------------------------------------------------ #
    def ab_test_p2(
        self,
        champion_version: int | None = None,
        challenger_version: int | None = None,
        *,
        n_boot: int = 300,
        promover: bool = False,
        experimento: str = "mibici-equipo20-p2-monitoreo",
        modelo_nombre: str = "mibici-equipo20-p2-riesgo",
    ) -> dict:
        """A/B testing de P2 (clasificación): champion vs challenger.

        Compara dos versiones del modelo registrado en MLflow Model Registry
        sobre el mismo conjunto de evaluación (julio a septiembre de 2026 para P2).
        Diseño pareado: ambas versiones predicen exactamente las mismas filas.

        Métricas: precisión, recall, F1, PR-AUC.
        Incluye test de McNemar y bootstrap por estación para IC95 de la
        diferencia B - A en F1 y PR-AUC.

        Regla de decisión (fijada antes de ver resultados):
        - IC95 de F1 y PR-AUC por encima de 0 -> promover el challenger.
        - IC95 de F1 y PR-AUC por debajo de 0 -> mantener el champion.
        - Otro caso -> sin diferencia clara: mantener el champion.

        Parametros
        ----------
        champion_version : int | None
            Versión del champion. Si None, usa el alias 'champion'.
        challenger_version : int | None
            Versión del challenger. Si None, usa el alias 'challenger'.
        n_boot : int
            Repeticiones del bootstrap por estación.
        promover : bool
            Si True y el challenger cumple los criterios, mueve el alias
            'champion' al challenger.
        experimento : str
            Experimento de MLflow donde registrar el A/B test.
        modelo_nombre : str
            Nombre del modelo registrado en MLflow Model Registry.

        Returns
        -------
        dict
            Resumen con métricas de ambas versiones, cambio relativo,
            bootstrap IC95, resultado de McNemar, veredicto y criterios.
        """
        mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
        client = MlflowClient()

        # Resolver versiones
        if champion_version is None:
            va = client.get_model_version_by_alias(modelo_nombre, "champion")
        else:
            va = client.get_model_version(modelo_nombre, str(champion_version))
        if challenger_version is None:
            vb = client.get_model_version_by_alias(modelo_nombre, "challenger")
        else:
            vb = client.get_model_version(modelo_nombre, str(challenger_version))

        if va.version == vb.version:
            raise ValueError("Champion y challenger son la misma versión.")

        etiqueta_a = f"A: champion (v{va.version})"
        etiqueta_b = f"B: challenger (v{vb.version})"
        logger.info(f"A/B P2: {etiqueta_a} vs {etiqueta_b}")

        # 1. Cargar modelos desde MLflow
        modelo_a = cargar_estimador(modelo_nombre, va.version)
        modelo_b = cargar_estimador(modelo_nombre, vb.version)

        # 2. Cargar datos de evaluación (julio a septiembre de 2026)
        serie = construir_serie_p2(fin_periodo=P2_FIN_ACTUAL)
        inicio = pd.Timestamp(P2_INICIO_ACTUAL)
        fin = pd.Timestamp(P2_FIN_ACTUAL)
        en_actual = serie["hora_objetivo"].between(inicio, fin)
        df = serie[en_actual].reset_index(drop=True)

        # Excluir estaciones nuevas como en la evaluación original
        df_conocidas = df[~df["estacion_nueva"]].reset_index(drop=True)
        y = df_conocidas[P2_OBJETIVO].astype(bool).to_numpy()
        X = df_conocidas[P2_VARIABLES]

        # 3. Predicciones de ambas versiones
        prob_a = modelo_a.predict_proba(X)[:, 1]
        pred_a = np.asarray(modelo_a.predict(X)).astype(bool)
        prob_b = modelo_b.predict_proba(X)[:, 1]
        pred_b = np.asarray(modelo_b.predict(X)).astype(bool)

        # 4. Métricas globales
        def metricas_clasificacion(y_true, y_pred, y_prob, nombre: str) -> dict:
            return {
                "precision": float(
                    np.sum((y_pred & y_true)) / max(np.sum(y_pred), 1)
                ),
                "recall": float(
                    np.sum((y_pred & y_true)) / max(np.sum(y_true), 1)
                ),
                "f1_score": float(f1_score(y_true, y_pred)),
                "pr_auc": float(auc(*precision_recall_curve(y_true, y_prob)[1::-1])),
            }

        m_a = metricas_clasificacion(y, pred_a, prob_a, etiqueta_a)
        m_b = metricas_clasificacion(y, pred_b, prob_b, etiqueta_b)

        # 5. Test de McNemar
        ok_a, ok_b = pred_a == y, pred_b == y
        solo_a = int((ok_a & ~ok_b).sum())
        solo_b = int((~ok_a & ok_b).sum())
        total = solo_a + solo_b
        estadistico = (abs(solo_a - solo_b) - 1) ** 2 / total if total else 0.0
        p_mcnemar = float(chi2.sf(estadistico, df=1))

        # 6. Bootstrap por estación para IC95 de F1 y PR-AUC (B - A)
        rng = np.random.default_rng(RANDOM_SEED)
        grupos = list(df_conocidas.groupby("estacion_id").indices.values())
        difs = {"f1": [], "pr_auc": []}
        for _ in range(n_boot):
            elegidas = rng.integers(0, len(grupos), len(grupos))
            idx = np.concatenate([grupos[j] for j in elegidas])
            yy = y[idx]
            pa, pb = prob_a[idx], prob_b[idx]
            pa_pred, pb_pred = pred_a[idx], pred_b[idx]
            difs["f1"].append(float(f1_score(yy, pb_pred)) - float(f1_score(yy, pa_pred)))
            difs["pr_auc"].append(
                float(auc(*precision_recall_curve(yy, pb)[1::-1]))
                - float(auc(*precision_recall_curve(yy, pa)[1::-1]))
            )
        boot = pd.DataFrame(difs)
        ic = {
            k: (float(boot[k].quantile(0.025)), float(boot[k].quantile(0.975)))
            for k in boot.columns
        }
        gana_b_pct = float((boot["f1"] > 0).mean() * 100)

        # 7. Criterios de decisión
        inf_f1, sup_f1 = ic["f1"]
        inf_pr, sup_pr = ic["pr_auc"]
        cumple = (inf_f1 > 0) and (inf_pr > 0)
        if (sup_f1 < 0) and (sup_pr < 0):
            veredicto = "MANTENER CHAMPION (el challenger es peor)"
        elif cumple:
            veredicto = "PROMOVER CHALLENGER"
        else:
            veredicto = "MANTENER CHAMPION (sin diferencia clara a favor del challenger)"

        # 8. Logs
        logger.info("\nMétricas de clasificación (julio a septiembre de 2026):")
        logger.info(
            pd.DataFrame([m_a, m_b], index=[etiqueta_a, etiqueta_b]).round(4).to_string()
        )
        logger.info(
            f"McNemar: solo A acierta {solo_a:,} | solo B acierta {solo_b:,} | "
            f"chi2 {estadistico:.2f} | p {p_mcnemar:.3g}"
        )
        logger.info("Diferencia B - A (bootstrap por estación, IC 95%):")
        for k, (lo, hi) in ic.items():
            clave = "f1_score" if k == "f1" else k  # en las métricas globales F1 se llama f1_score
            logger.info(
                f"  {k:<18} {m_b[clave] - m_a[clave]:+.4f}  IC [{lo:+.4f}, {hi:+.4f}]"
            )
        logger.info(f"B mejor en F1 en {gana_b_pct:.1f}% de las repeticiones")
        logger.success(f"Decisión: {veredicto}")

        # 9. Promoción

        # 10. Promoción
        if promover and cumple:
            client.set_registered_model_alias(modelo_nombre, "previous_champion", va.version)
            client.set_registered_model_alias(modelo_nombre, "champion", vb.version)
            logger.success(
                f"Alias 'champion' -> v{vb.version} (anterior: v{va.version})."
            )
        elif promover:
            logger.info("No se mueve el alias: la regla de decisión no lo indica.")

        return {
            "champion_version": va.version,
            "challenger_version": vb.version,
            "champion_run_id": va.run_id,
            "challenger_run_id": vb.run_id,
            "metricas_champion": m_a,
            "metricas_challenger": m_b,
            "cambio_relativo": {k: float(m_b[k] - m_a[k]) for k in m_a},
            "ic95": ic,
            "gana_b_pct": gana_b_pct,
            "mcnemar_solo_a": solo_a,
            "mcnemar_solo_b": solo_b,
            "mcnemar_p": p_mcnemar,
            "mcnemar_chi2": estadistico,
            "criterios": {
                "IC95_F1 > 0": inf_f1 > 0,
                "IC95_PR_AUC > 0": inf_pr > 0,
            },
            "cumple": cumple,
            "veredicto": veredicto,
        }

    # ------------------------------------------------------------------ #
    # Umbral de A/B (WAPE mensual)
    # ------------------------------------------------------------------ #
    def calcular_umbral_ab(
        self,
        champion_version: int | None = None,
        *,
        experimento: str = "mibici-equipo20-p1-monitoreo",
        modelo_nombre: str = "mibici-equipo20-p1-demanda",
        inicio: str | None = None,
        fin: str | None = None,
    ) -> dict:
        """Calcula el umbral de A/B para P1 a partir de la variación mensual del WAPE.

        Calcula el WAPE de la versión vigente (v1) mes a mes, entre ``inicio`` y
        ``fin`` (meses que el campeón no usó para entrenar), y toma la variación
        relativa entre meses (desviación estándar / media) como umbral de mejora:
        una mejora menor que lo que el error de v1 varía normalmente de un mes a
        otro no se distingue de ese vaivén.

        Como el A/B se evalúa sobre el mismo conjunto, el cambio relativo del
        WAPE es igual al del MAE, así que el umbral se usa directo en
        ``ab_test_p1``.

        Parametros
        ----------
        champion_version : int | None
            Versión del champion. Si None, usa el alias 'champion'.
        experimento : str
            Experimento de MLflow donde registrar el cálculo del umbral.
        modelo_nombre : str
            Nombre del modelo registrado en MLflow Model Registry.
        inicio : str | None
            Fecha de inicio del periodo de evaluación (formato YYYY-MM-DD).
            Si None, usa ``P1_TEST_INICIO``.
        fin : str | None
            Fecha de fin del periodo de evaluación (formato YYYY-MM-DD).
            Si None, usa ``P1_ACTUAL_FIN``.

        Returns
        -------
        dict
            Tabla mensual de WAPE, media, desviación estándar y el umbral
            propuesto (redondeado hacia arriba a 0.1%).
        """
        import math

        mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
        client = MlflowClient()

        if champion_version is None:
            va = client.get_model_version_by_alias(modelo_nombre, "champion")
        else:
            va = client.get_model_version(modelo_nombre, str(champion_version))

        logger.info(f"Calculando umbral A/B para campeón v{va.version}")

        # 1. Conjunto de evaluación
        inicio = inicio or P1_TEST_INICIO
        fin = fin or P1_ACTUAL_FIN

        from proyecto_mibici.features import construir_dataset_p1

        df_p1, _ = construir_dataset_p1()
        t = df_p1["target_hour"]
        df = df_p1[(t >= pd.Timestamp(inicio)) & (t < pd.Timestamp(fin))]
        df = df.dropna(subset=P1_FEATURES_V2 + [P1_TARGET]).copy()
        if df.empty:
            raise ValueError("No hay filas en el periodo de evaluación para el umbral.")

        modelo = mlflow.pyfunc.load_model(f"models:/{modelo_nombre}/{va.version}")
        df["pred"] = np.clip(
            np.asarray(modelo.predict(df[P1_FEATURES_V2]), dtype=float).ravel(), 0, None
        )
        df["mes"] = df["target_hour"].dt.to_period("M")

        # 2. WAPE mensual
        filas = []
        for mes, g in df.groupby("mes"):
            y = g[P1_TARGET].to_numpy(dtype=float)
            p = g["pred"].to_numpy(dtype=float)
            filas.append(
                {
                    "mes": str(mes),
                    "filas": len(g),
                    "demanda_media": float(y.mean()),
                    "MAE": float(np.abs(y - p).mean()),
                    "WAPE": float(np.abs(y - p).sum() / y.sum()),
                }
            )
        tabla = pd.DataFrame(filas)

        # 3. Umbral: variación relativa mes a mes del WAPE
        media = float(tabla["WAPE"].mean())
        desv = float(tabla["WAPE"].std(ddof=1))
        variacion_pct = desv / media * 100
        umbral = math.ceil(variacion_pct * 10) / 10  # redondeo hacia arriba a 0.1%

        logger.info("\n" + tabla.round(4).to_string(index=False))
        logger.info(f"WAPE medio mensual: {media:.4f} | desviación estándar: {desv:.4f}")
        logger.success(
            f"Variación relativa mes a mes: {variacion_pct:.2f}% -> "
            f"UMBRAL propuesto: {umbral}% "
            f"(usar --min-mejora-mae {umbral} en ab_test_p1)"
        )


        # 5. Guardar tabla localmente
        salida = REPORTS_DIR / "ab_test"
        salida.mkdir(parents=True, exist_ok=True)
        tabla_csv = salida / "umbral_wape_mensual.csv"
        tabla.to_csv(tabla_csv, index=False)
        logger.info(f"Tabla guardada en {tabla_csv}")

        return {
            "campeon_version": va.version,
            "tabla_wape_mensual": tabla.to_dict("records"),
            "wape_medio": media,
            "desv_estandar": desv,
            "variacion_relativa_pct": variacion_pct,
            "umbral_propuesto": float(umbral),
            "tabla_csv": str(tabla_csv),
        }


# ---------------------------------------------------------------------- #
# Importación tardía de Snapshot para anotaciones de tipo
# ---------------------------------------------------------------------- #
from evidently.core.report import Snapshot

# ---------------------------------------------------------------------- #
# CLI opcional: permite ejecutar este módulo directamente como script
# ---------------------------------------------------------------------- #
import typer

app = typer.Typer(help="Monitoreo de drift con Evidently (clase reutilizable).")


@app.command()
def ejemplo_p1():
    """Ejemplo de uso con P1 (solo si existen los datos de modelado)."""
    from proyecto_mibici.config import (
        P1_ACTUAL_FIN,
        P1_FEATURES_V2,
        P1_TEST_FIN,
        P1_VAL_INICIO,
    )
    from proyecto_mibici.features import construir_dataset_p1

    df_p1, _ = construir_dataset_p1()
    t = df_p1["target_hour"]
    actual = df_p1[(t >= pd.Timestamp(P1_TEST_FIN)) & (t < pd.Timestamp(P1_ACTUAL_FIN))]
    ref = df_p1[t < pd.Timestamp(P1_VAL_INICIO)]

    monitor = EvidentlyMonitor(
        numerical_columns=[c for c in P1_FEATURES_V2 if c != "target_is_weekend"],
        categorical_columns=["target_is_weekend"],
        nombre_proyecto="p1-demanda-horaria",
        descripcion_proyecto="Monitoreo de drift P1: referencia 2025 vs octubre 2026",
    )
    resumen = monitor.comparar(ref, actual, nombre="ejemplo_p1")
    logger.info(f"Resumen P1: {resumen}")


if __name__ == "__main__":
    app()
