#!/bin/sh
set -e

#!/bin/sh
set -e

MLFLOW_HOST="${MLFLOW_HOST:-mlflow}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"

echo "Esperando a MLflow en ${MLFLOW_HOST}:${MLFLOW_PORT}..."
# Este comando verifica la red de forma nativa y directa
until python -c "import socket; s = socket.socket(); s.settimeout(2); s.connect(('${MLFLOW_HOST}', ${MLFLOW_PORT}))" > /dev/null 2>&1; do
  echo "  MLflow no responde, reintentando en 2s..."
  sleep 2
done
echo "MLflow disponible."

echo "==> Paso 1: Ingesta de datos (dataset.py)"
python -m proyecto_mibici.dataset

echo "==> Paso 2: Entrenamiento P1 (lgbm_tweedie)"
python -m proyecto_mibici.modeling.train_p1 --model lgbm_tweedie

echo "==> Paso 3: Entrenamiento P2 (gbm)"
python -m proyecto_mibici.modeling.train_p2 --modelo gbm

echo "==> Paso 4: Entrenamiento P3 (kmeans k=4)"
python -m proyecto_mibici.modeling.train_p3 --modelo kmeans --n-clusters 4 --n-init 10

echo "==> Paso 5: Registro de campeones en Model Registry"
python run_mejor_modelo.py

echo "==> Prep completado: campeones registrados en MLflow"
