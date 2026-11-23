"""Escenario de carga del sistema MiBici: usuarios concurrentes contra /predict de las tres APIs.

Cada pregunta tiene su propia clase de usuario con su host. Los payloads se generan al azar
respetando las reglas de validación de schemas.py (coherencia de P2 y franjas de P3), de modo
que una respuesta distinta de 200 es un error real del servicio. Un 5 % de las peticiones manda
un payload inválido a propósito y se cuenta como éxito solo si la API responde 422.

Prueba de estrés (sin tiempo de espera, para medir capacidad máxima):
    LOCUST_ESPERA_MIN=0 LOCUST_ESPERA_MAX=0 locust -f locust/locustfile.py --headless -u 30 -r 10 -t 2m ...

Ejecución sin interfaz (desde la raíz del repo, con el sistema levantado):
    locust -f locust/locustfile.py --headless -u 50 -r 5 -t 3m \
        --csv reports/locust/u50 --html reports/locust/u50.html
"""

from datetime import datetime, timedelta
import os
import random

from locust import HttpUser, between, task

HOST_P1 = os.environ.get("LOCUST_HOST_P1", "http://localhost:8001")
HOST_P2 = os.environ.get("LOCUST_HOST_P2", "http://localhost:8002")
HOST_P3 = os.environ.get("LOCUST_HOST_P3", "http://localhost:8003")
# Tiempo de espera entre peticiones de cada usuario (segundos). Con 0 y 0 se mide la capacidad máxima.
ESPERA_MIN = float(os.environ.get("LOCUST_ESPERA_MIN", "0.5"))
ESPERA_MAX = float(os.environ.get("LOCUST_ESPERA_MAX", "2"))


def payload_p1() -> dict:
    hora = datetime(2026, 4, 1) + timedelta(hours=random.randint(0, 24 * 180))
    lag_1h = random.randint(0, 30)
    return {
        "target_hour": hora.isoformat(),
        "lag_1h": lag_1h,
        "lag_24h": float(random.randint(0, 30)),
        "lag_168h": float(random.randint(0, 30)),
        "rolling_3h": round(random.uniform(0, 25), 2),
        "rolling_168h": round(random.uniform(0, 20), 2),
    }


def payload_p2() -> dict:
    capacidad = random.randint(12, 40)
    balance = random.randint(-capacidad, capacidad)
    dia = random.randint(0, 6)
    return {
        "hora_del_dia": random.randint(0, 23),
        "dia_semana": dia,
        "fin_de_semana": int(dia >= 5),
        "capacidad": capacidad,
        "salidas": random.randint(0, 15),
        "llegadas": random.randint(0, 15),
        "balance_acumulado_dia": balance,
        "acumulado_relativo": round(balance / capacidad, 4),
        "balance_ult_3h_relativo": round(random.uniform(-1, 1), 4),
        "riesgo_vacia": int(balance <= -0.5 * capacidad),
        "riesgo_llena": int(balance >= 0.5 * capacidad),
    }


def payload_p3() -> dict:
    # Cinco franjas (la madrugada no entra al modelo): las cuatro enviadas suman menos de 1
    pesos = [random.random() for _ in range(5)]
    total = sum(pesos)
    manana, mediodia, tarde, noche = (round(p / total, 4) for p in pesos[:4])
    return {
        "pct_manana": manana,
        "pct_mediodia": mediodia,
        "pct_tarde": tarde,
        "pct_noche": noche,
        "pct_fin_de_semana": round(random.uniform(0.05, 0.5), 4),
        "duracion_media": round(random.uniform(5, 40), 2),
        "viajes_dia_log": round(random.uniform(0.5, 5), 3),
        "balance_neto": round(random.uniform(-0.5, 0.5), 4),
    }


class _UsuarioMiBici(HttpUser):
    """Comportamiento común: muchas predicciones, consultas ocasionales de salud y algún error de cliente."""

    abstract = True
    wait_time = between(ESPERA_MIN, ESPERA_MAX)
    generar_payload = staticmethod(lambda: {})

    @task(20)
    def predecir(self):
        self.client.post("/predict", json=self.generar_payload(), name="/predict")

    @task(1)
    def predecir_invalido(self):
        datos = self.generar_payload()
        datos.pop(next(iter(datos)))  # falta un campo obligatorio
        with self.client.post(
            "/predict", json=datos, name="/predict [invalido]", catch_response=True
        ) as resp:
            if resp.status_code == 422:
                resp.success()
            else:
                resp.failure(f"Se esperaba 422 y llegó {resp.status_code}")

    @task(1)
    def salud(self):
        self.client.get("/health", name="/health")


class UsuarioP1(_UsuarioMiBici):
    host = HOST_P1
    generar_payload = staticmethod(payload_p1)


class UsuarioP2(_UsuarioMiBici):
    host = HOST_P2
    generar_payload = staticmethod(payload_p2)


class UsuarioP3(_UsuarioMiBici):
    host = HOST_P3
    generar_payload = staticmethod(payload_p3)
