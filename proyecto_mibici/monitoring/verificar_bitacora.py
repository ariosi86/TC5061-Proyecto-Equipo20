#!/usr/bin/env python
"""
Prueba de bitácora y conectividad contra la consola levantada en
http://127.0.0.1:8000.

Este script es independiente de las entregas anteriores: solo verifica
(1) que el servicio en el puerto 8000 responde y (2) que se escriben
registros de bitácora (logs) tanto en consola como en archivo.

Uso:
    python proyecto_mibici/monitoring/verificar_bitacora.py
"""

from __future__ import annotations

from pathlib import Path
import sys

from loguru import logger
import requests

# --------------------------------------------------------------------------- #
# Configuración de la bitácora (consola + archivo)
# --------------------------------------------------------------------------- #
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "verificar_bitacora.log"

# Limpiamos handlers previos de loguru y configuramos los nuestros.
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")
logger.add(
    LOG_FILE,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="DEBUG",
    rotation="1 MB",
    retention="3 days",
    enqueue=True,
)

CONSOLA_URL = "http://127.0.0.1:8000"
HEALTH_ENDPOINT = f"{CONSOLA_URL}/health"


def consultar_health(timeout: float = 5.0) -> dict:
    """Intenta consultar el endpoint /health de la consola en el puerto 8000.

    Devuelve un dict con el resultado: OK, código de estado, cuerpo y error.
    No lanza excepción si el servicio no está levantado; lo reporta como fallo.
    """
    logger.info(f"Consultando {HEALTH_ENDPOINT} (timeout={timeout}s) ...")
    resultado = {
        "url": HEALTH_ENDPOINT,
        "accesible": False,
        "status_code": None,
        "cuerpo": None,
        "error": None,
    }
    try:
        respuesta = requests.get(HEALTH_ENDPOINT, timeout=timeout)
        resultado["accesible"] = True
        resultado["status_code"] = respuesta.status_code
        try:
            resultado["cuerpo"] = respuesta.json()
        except Exception:
            resultado["cuerpo"] = respuesta.text
        logger.success(f"Respuesta HTTP {respuesta.status_code} desde {HEALTH_ENDPOINT}")
        logger.debug(f"Cuerpo: {resultado['cuerpo']}")
    except requests.exceptions.ConnectionError as exc:
        resultado["error"] = f"Conexión rechazada: {exc}"
        logger.warning(f"No se pudo conectar a {HEALTH_ENDPOINT}: {exc}")
    except requests.exceptions.Timeout as exc:
        resultado["error"] = f"Tiempo de espera agotado: {exc}"
        logger.warning(f"Timeout consultando {HEALTH_ENDPOINT}: {exc}")
    except Exception as exc:  # noqa: BLE001
        resultado["error"] = f"Error inesperado: {exc}"
        logger.error(f"Error consultando {HEALTH_ENDPOINT}: {exc}")
    return resultado


def verificar_bitacora_archivo() -> bool:
    """Verifica que el archivo de log se creó y contiene al menos una entrada."""
    logger.debug("Verificando archivo de bitácora ...")
    if not LOG_FILE.exists():
        logger.error(f"El archivo de bitácora no existe: {LOG_FILE}")
        return False
    texto = LOG_FILE.read_text(encoding="utf-8", errors="ignore")
    if not texto.strip():
        logger.warning(f"El archivo de bitácora existe pero está vacío: {LOG_FILE}")
        return False
    logger.success(f"Bitácora verificada en {LOG_FILE} ({len(texto)} bytes)")
    return True


def main() -> int:
    logger.info("=" * 60)
    logger.info("PRUEBA DE BITÁCORA Y CONECTIVIDAD (puerto 8000)")
    logger.info("=" * 60)

    # 1. Escribir algunas entradas de bitácora de prueba.
    logger.info("Entrada de prueba 1: inicio del script.")
    logger.debug("Entrada de prueba 2: nivel DEBUG (solo en archivo).")

    # 2. Consultar la consola en http://127.0.0.1:8000/health.
    resultado = consultar_health()

    # 3. Verificar que la bitácora se escriba en archivo.
    ok_bitacora = verificar_bitacora_archivo()

    # 4. Resumen final.
    logger.info("-" * 60)
    logger.info("RESUMEN")
    logger.info(
        f"  Consola {CONSOLA_URL}: {'ACCESIBLE' if resultado['accesible'] else 'NO ACCESIBLE'}"
    )
    if resultado["status_code"] is not None:
        logger.info(f"  Código HTTP: {resultado['status_code']}")
    if resultado["cuerpo"] is not None:
        logger.info(f"  Cuerpo: {resultado['cuerpo']}")
    if resultado["error"]:
        logger.info(f"  Error: {resultado['error']}")
    logger.info(f"  Bitácora (archivo): {'OK' if ok_bitacora else 'FALLO'}")

    if not resultado["accesible"]:
        logger.warning(
            "La consola no respondió en http://127.0.0.1:8000. "
            "Verifica que el servicio esté levantado en ese puerto."
        )
        return 1

    if not ok_bitacora:
        logger.error("La bitácora no se escribió correctamente en archivo.")
        return 1

    logger.success("Prueba superada: consola accesible y bitácora activa.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
