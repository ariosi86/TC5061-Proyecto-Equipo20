#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo centralizado de ingesta, procesamiento primario y versionado de datos para MiBici.
Descarga datos abiertos, procesa viajes y estaciones a Parquet y genera manifest.json.
Ejecución: python -m proyecto_mibici.dataset
"""

import hashlib
import json
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from loguru import logger
import pandas as pd
import requests
from tqdm import tqdm
import typer

from proyecto_mibici.config import (
    DATA_URLS,
    DEFAULT_HEADERS,
    MANIFEST_PATH,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RAW_DATA_DIR,
    STATIONS_PROCESSED_PATH,
    TRIPS_PROCESSED_PATH,
)


def compute_sha256(filepath: Path) -> str:
    """Calcula el hash SHA256 para garantizar reproducibilidad e inmutabilidad."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(url: str, destination_path: Path) -> Path:
    """Descarga el archivo si no existe previamente."""
    if destination_path.exists():
        logger.info(f"El archivo ya existe: {destination_path.name}")
        return destination_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Descargando {destination_path.name}...")
    response = requests.get(url, headers=DEFAULT_HEADERS, stream=True, timeout=90)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(destination_path, "wb") as f, tqdm(
        desc=destination_path.name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            bar.update(f.write(chunk))

    return destination_path


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza encabezados a minúsculas, sin acentos ni espacios."""
    replacements = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}
    new_cols = []
    for col in df.columns:
        c = str(col).strip().lower().replace(" ", "_")
        for orig, sub in replacements.items():
            c = c.replace(orig, sub)
        new_cols.append(c)
    df.columns = new_cols
    return df


def read_csv_flexible(file_path: Path) -> pd.DataFrame:
    """Lee CSV probando utf-8 y latin-1 para sortear bytes conflictivos (0xf1)."""
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin-1")


def process_stations(stations_file: Path) -> pd.DataFrame:
    """Procesa el catálogo de estaciones manteniendo los nombres originales del CSV."""
    logger.info("Procesando nomenclatura de estaciones...")
    df = read_csv_flexible(stations_file)
    df = clean_column_names(df)

    # 1. Limpieza y tipado del identificador original ('id')
    df = df.dropna(subset=["id"])
    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(-1).astype("int32")
    df = df[df["id"] != -1]

    # 2. Tipado de coordenadas originales
    if "latitude" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    if "longitude" in df.columns:
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # 3. Tipado de la capacidad original ('dpcapacity')
    if "dpcapacity" in df.columns:
        df["dpcapacity"] = (
            pd.to_numeric(df["dpcapacity"], errors="coerce")
            .fillna(0)
            .astype("int32")
        )

    # 4. Columnas atómicas auxiliares derivadas de 'name'
    if "name" in df.columns:
        df["station_code"] = df["name"].str.extract(r"\((.*?)\)", expand=False).fillna("")
        df["municipality_code"] = df["station_code"].str.split("-").str[0].fillna("")
        df["address"] = (
            df["name"]
            .str.replace(r"\(.*?\)", "", regex=True)
            .str.replace(r"\s*/\s*", " / ", regex=True)
            .str.strip()
        )

    # Exportar a Parquet
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(STATIONS_PROCESSED_PATH, index=False)
    logger.success(f"Estaciones guardadas: {len(df)} registros -> {STATIONS_PROCESSED_PATH.name}")
    return df


def process_trips(downloaded_paths: List[Path]) -> pd.DataFrame:
    """Concatena los CSVs mensuales y exporta a trips.parquet manteniendo nombres originales."""
    logger.info("Procesando archivos mensuales de viajes...")
    dfs = []
    for f in downloaded_paths:
        temp_df = read_csv_flexible(f)
        temp_df = clean_column_names(temp_df)
        temp_df["periodo"] = f.stem.replace("datos_abiertos_", "")
        dfs.append(temp_df)

    df = pd.concat(dfs, ignore_index=True)

    # Conversión de fechas con formato explícito
    date_format = "%Y-%m-%d %H:%M:%S"
    df["inicio_del_viaje"] = pd.to_datetime(
        df["inicio_del_viaje"], errors="coerce", format=date_format
    )
    df["fin_del_viaje"] = pd.to_datetime(
        df["fin_del_viaje"], errors="coerce", format=date_format
    )

    # Limpieza de registros sin fecha o estaciones clave
    df = df.dropna(subset=["inicio_del_viaje", "origen_id", "destino_id"])

    # Conversión de tipos numéricos
    df["origen_id"] = (
        pd.to_numeric(df["origen_id"], errors="coerce").fillna(-1).astype("int32")
    )
    df["destino_id"] = (
        pd.to_numeric(df["destino_id"], errors="coerce").fillna(-1).astype("int32")
    )

    if "viaje_id" in df.columns:
        df["viaje_id"] = pd.to_numeric(df["viaje_id"], errors="coerce").fillna(-1).astype("int64")

    # Guardar en Parquet
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TRIPS_PROCESSED_PATH, index=False)
    logger.success(f"Viajes procesados: {len(df):,} filas -> {TRIPS_PROCESSED_PATH.name}")


    # ==========================================
    # GENERACIÓN DE PARTICIONES DE MODELADO
    # ==========================================
    # 1. Train: Año 2025 completo (2025-01-01 a 2025-12-31)
    from proyecto_mibici.config import TRIPS_TRAIN_PATH, TRIPS_TEST_PATH
    
    df_train_2025 = df[
        (df["inicio_del_viaje"] >= "2025-01-01") & 
        (df["inicio_del_viaje"] < "2026-01-01")
    ]
    TRIPS_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_train_2025.to_parquet(TRIPS_TRAIN_PATH, index=False)
    logger.success(f"Train 2025 guardado: {len(df_train_2025):,} filas -> {TRIPS_TRAIN_PATH}")

    # 2. Test: Abril a Septiembre 2026 (2026-04-01 a 2026-09-30)
    df_test_2026 = df[
        (df["inicio_del_viaje"] >= "2026-04-01") & 
        (df["inicio_del_viaje"] <= "2026-09-30 23:59:59")
    ]
    TRIPS_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_test_2026.to_parquet(TRIPS_TEST_PATH, index=False)
    logger.success(f"Test (Abr-Sep 2026) guardado: {len(df_test_2026):,} filas -> {TRIPS_TEST_PATH}")


    return df


def generate_manifest(trips_df: pd.DataFrame, stations_df: pd.DataFrame) -> None:
    """Genera manifest.json con hashes y dimensiones oficiales."""
    manifest = {
        "dataset_name": "MiBici - TC5061",
        "trips": {
            "path": str(TRIPS_PROCESSED_PATH.relative_to(PROJ_ROOT)),
            "sha256": compute_sha256(TRIPS_PROCESSED_PATH),
            "total_rows": int(len(trips_df)),
            "date_range": [
                str(trips_df["inicio_del_viaje"].min()),
                str(trips_df["inicio_del_viaje"].max()),
            ],
        },
        "stations": {
            "path": str(STATIONS_PROCESSED_PATH.relative_to(PROJ_ROOT)),
            "sha256": compute_sha256(STATIONS_PROCESSED_PATH),
            "total_stations": int(len(stations_df)),
        },
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    logger.success(f"Manifiesto guardado en: {MANIFEST_PATH}")


app = typer.Typer()


@app.command()
def main():
    # 1. Estaciones
    stations_url = DATA_URLS["stations"]
    stations_filename = Path(urlparse(stations_url).path).name
    stations_local = RAW_DATA_DIR / stations_filename
    download_file(stations_url, stations_local)
    stations_df = process_stations(stations_local)

    # 2. Viajes
    trip_files = []
    for period, url in DATA_URLS["trips"].items():
        trip_filename = Path(urlparse(url).path).name
        dest = RAW_DATA_DIR / trip_filename
        try:
            download_file(url, dest)
            trip_files.append(dest)
        except Exception as e:
            logger.error(f"Error descargando {period}: {e}")

    # 3. Guardar Parquet y Manifiesto
    trips_df = process_trips(trip_files)
    generate_manifest(trips_df, stations_df)


if __name__ == "__main__":
    app()