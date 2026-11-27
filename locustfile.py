"""Escenario de pruebas de carga con Locust para los microservicios MiBici (P1, P2, P3).

Uso local (contra puertos expuestos en localhost):
    locust -f locustfile.py --host http://localhost:8001

Uso headless generando CSV y reporte HTML en reports/locust/:
    locust -f locustfile.py --headless -u 20 -r 5 -t 1m \
        --csv reports/locust/mibici --html reports/locust/reporte_locust.html

Variables de entorno opcionales para apuntar a cada servicio:
    API_P1_URL (default: http://localhost:8001)
    API_P2_URL (default: http://localhost:8002)
    API_P3_URL (default: http://localhost:8003)
"""

import os
import random
from datetime import datetime, timedelta

from locust import HttpUser, between, task

API_P1_URL = os.getenv("API_P1_URL", "http://localhost:8001").rstrip("/")
API_P2_URL = os.getenv("API_P2_URL", "http://localhost:8002").rstrip("/")
API_P3_URL = os.getenv("API_P3_URL", "http://localhost:8003").rstrip("/")


def _payload_p1() -> dict:
    """Genera un payload válido para P1Input (demanda horaria)."""
    base = datetime(2026, 11, 4, 0, 0, 0)
    target = base + timedelta(hours=random.randint(0, 167))
    lag_1h = random.randint(0, 25)
    lag_24h = round(random.uniform(0.0, 25.0), 2)
    lag_168h = round(random.uniform(0.0, 25.0), 2)
    rolling_3h = round((lag_1h + random.uniform(0.0, 15.0)) / 2.0, 2)
    rolling_168h = round(random.uniform(1.0, 15.0), 2)
    return {
        "target_hour": target.strftime("%Y-%m-%dT%H:%M:%S"),
        "lag_1h": lag_1h,
        "lag_24h": lag_24h,
        "lag_168h": lag_168h,
        "rolling_3h": rolling_3h,
        "rolling_168h": rolling_168h,
    }


def _payload_p2() -> dict:
    """Genera un payload válido para P2Input (riesgo de desbalance)."""
    hora = random.randint(0, 23)
    dia = random.randint(0, 6)
    capacidad = random.choice([15.0, 19.0, 23.0, 27.0])
    salidas = float(random.randint(0, 18))
    llegadas = float(random.randint(0, 18))
    balance_dia = round(llegadas - salidas + random.uniform(-5.0, 5.0), 2)
    return {
        "hora_del_dia": hora,
        "dia_semana": dia,
        "fin_de_semana": 1 if dia in (5, 6) else 0,
        "capacidad": capacidad,
        "salidas": salidas,
        "llegadas": llegadas,
        "balance_acumulado_dia": balance_dia,
        "acumulado_relativo": round(balance_dia / capacidad, 4),
        "balance_ult_3h_relativo": round(random.uniform(-0.5, 0.5), 4),
        "riesgo_vacia": random.choice([0, 1]),
        "riesgo_llena": random.choice([0, 1]),
    }


def _payload_p3() -> dict:
    """Genera un payload válido para P3Input (clustering de estaciones)."""
    # Se generan 4 proporciones cuya suma sea <= 1.0 (la madrugada no entra al modelo)
    pesos = [random.uniform(0.05, 0.35) for _ in range(4)]
    total = sum(pesos)
    escala = random.uniform(0.85, 0.99) / total if total > 0.99 else 1.0
    pct_manana, pct_mediodia, pct_tarde, pct_noche = [
        round(p * escala, 4) for p in pesos
    ]
    return {
        "pct_manana": pct_manana,
        "pct_mediodia": pct_mediodia,
        "pct_tarde": pct_tarde,
        "pct_noche": pct_noche,
        "pct_fin_de_semana": round(random.uniform(0.05, 0.45), 4),
        "duracion_media": round(random.uniform(5.0, 35.0), 2),
        "viajes_dia_log": round(random.uniform(0.5, 5.5), 3),
        "balance_neto": round(random.uniform(-0.4, 0.4), 4),
    }


def _validar_respuesta(response, campo_esperado: str) -> None:
    """Marca fallo en Locust si el código HTTP != 200 o falta el campo esperado."""
    if response.status_code != 200:
        response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
        return
    try:
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        response.failure(f"JSON inválido: {exc}")
        return
    if campo_esperado not in data:
        response.failure(f"Falta '{campo_esperado}' en respuesta: {data}")
        return
    response.success()


class MiBiciP1User(HttpUser):
    """Usuario simulado que consulta predicciones de demanda horaria (API P1)."""

    host = API_P1_URL
    weight = 3
    wait_time = between(0.5, 2.0)

    @task(5)
    def predecir_demanda(self):
        with self.client.post(
            "/predict",
            json=_payload_p1(),
            name="P1 POST /predict",
            catch_response=True,
        ) as resp:
            _validar_respuesta(resp, "demanda_predicha")

    @task(1)
    def health(self):
        self.client.get("/health", name="P1 GET /health")


class MiBiciP2User(HttpUser):
    """Usuario simulado que consulta riesgo de desbalance de estaciones (API P2)."""

    host = API_P2_URL
    weight = 2
    wait_time = between(0.5, 2.0)

    @task(5)
    def predecir_riesgo(self):
        with self.client.post(
            "/predict",
            json=_payload_p2(),
            name="P2 POST /predict",
            catch_response=True,
        ) as resp:
            _validar_respuesta(resp, "prob_riesgo")

    @task(1)
    def health(self):
        self.client.get("/health", name="P2 GET /health")


class MiBiciP3User(HttpUser):
    """Usuario simulado que consulta asignación de cluster funcional (API P3)."""

    host = API_P3_URL
    weight = 1
    wait_time = between(1.0, 3.0)

    @task(5)
    def predecir_grupo(self):
        with self.client.post(
            "/predict",
            json=_payload_p3(),
            name="P3 POST /predict",
            catch_response=True,
        ) as resp:
            _validar_respuesta(resp, "grupo_kmeans")

    @task(1)
    def health(self):
        self.client.get("/health", name="P3 GET /health")
