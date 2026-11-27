from contextlib import asynccontextmanager

from fastapi import FastAPI

from proyecto_mibici.api import router_p1


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Se ejecuta al arrancar: precarga el modelo para que la primera petición no espere
    try:
        router_p1.cargar_modelo_p1()
    except Exception as e:
        print(f"[WARN] No se pudo precargar el modelo P1: {e}")
    yield


app = FastAPI(title="MiBici API - Equipo 20", lifespan=lifespan)
app.include_router(router_p1.router, prefix="/p1")