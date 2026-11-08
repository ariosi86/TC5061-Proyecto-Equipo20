FROM python:3.12-slim

# LightGBM (P1) necesita OpenMP en el sistema
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Dependencias primero: esta capa se cachea mientras requirements-api.txt no cambie
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Código de la API y modelo empaquetado
COPY proyecto_mibici/api ./proyecto_mibici/api
RUN touch proyecto_mibici/__init__.py
COPY models/p1 ./models/p1

ENV P1_MODEL_URI=/app/models/p1 \
    P1_MODEL_NAME=mibici-equipo20-p1-demanda \
    P1_MODEL_ALIAS=champion \
    P1_MODEL_VERSION=1

EXPOSE 8000
CMD ["uvicorn", "proyecto_mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]