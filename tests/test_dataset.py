"""Pruebas unitarias simples para funciones clave de dataset.py (ETL e integridad de datos).

Se prueban funciones puras que no requieren descarga de datos ni conexión a internet:
- clean_column_names: normalización de encabezados
- compute_sha256: hash criptográfico de archivos
"""
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from proyecto_mibici.dataset import clean_column_names, compute_sha256


def test_clean_column_names_minusculas_sin_espacios():
    """Los encabezados se convierten a minúsculas y los espacios se reemplazan por guiones bajos."""
    df = pd.DataFrame({"Fecha del Viaje": [1], "ID Estacion": [2], "Duracion (min)": [3]})
    resultado = clean_column_names(df)
    assert list(resultado.columns) == ["fecha_del_viaje", "id_estacion", "duracion_(min)"]  # nosec B101


def test_clean_column_names_sin_acentos():
    """Los caracteres con acento se reemplazan por su equivalente sin acento."""
    df = pd.DataFrame({"Estación": [1], "Municipio": [2], "Año": [3]})
    resultado = clean_column_names(df)
    assert list(resultado.columns) == ["estacion", "municipio", "ano"]  # nosec B101


def test_clean_column_names_no_modifica_valores():
    """La normalización de encabezados no altera los valores del DataFrame."""
    df = pd.DataFrame({"Nombre Columna": [1, 2, 3]})
    resultado = clean_column_names(df)
    assert list(resultado["nombre_columna"]) == [1, 2, 3]  # nosec B101


def test_compute_sha256_es_determinista(tmp_path):
        """El hash SHA-256 de un archivo siempre es el mismo para el mismo contenido."""
        archivo = tmp_path / "datos.txt"
        archivo.write_text("contenido de prueba para hash")
        hash_1 = compute_sha256(archivo)
        hash_2 = compute_sha256(archivo)
        assert hash_1 == hash_2  # nosec B101


def test_compute_sha256_cambia_si_el_contenido_cambia(tmp_path):
        """Si el contenido del archivo cambia, el hash también cambia."""
        archivo = tmp_path / "datos.txt"
        archivo.write_text("contenido A")
        hash_a = compute_sha256(archivo)
        archivo.write_text("contenido B")
        hash_b = compute_sha256(archivo)
        assert hash_a != hash_b  # nosec B101


def test_compute_sha256_coincide_con_hashlib(tmp_path):
        """El hash calculado coincide con el de la librería estándar hashlib."""
        archivo = tmp_path / "datos.txt"
        contenido = b"cualquier contenido para validar"
        archivo.write_bytes(contenido)
        hash_proyecto = compute_sha256(archivo)
        hash_estandar = hashlib.sha256(contenido).hexdigest()
        assert hash_proyecto == hash_estandar  # nosec B101
