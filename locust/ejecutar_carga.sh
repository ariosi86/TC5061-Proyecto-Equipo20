#!/usr/bin/env bash
# Corre los escenarios de carga con Locust y registra en paralelo el consumo de los contenedores.
# Requiere el sistema levantado (docker compose -f docker/docker-compose.yml up -d) y locust instalado.
#
# Uso (desde la raíz del repo):  bash locust/ejecutar_carga.sh [prefijo] [duracion]
#   prefijo : antepone un nombre a los archivos (p. ej. "cache_" para comparar una optimización)
#   duracion: duración de cada escenario en formato de locust (por defecto 3m)
set -euo pipefail

PREFIJO="${1:-}"
DURACION="${2:-3m}"
SALIDA="reports/locust"
PYTHON="${PYTHON:-python}"
mkdir -p "$SALIDA"

segundos() {  # 3m -> 180, 90s -> 90
  case "$1" in *m) echo $(( ${1%m} * 60 ));; *s) echo "${1%s}";; *) echo "$1";; esac
}
SEG=$(segundos "$DURACION")

echo "==> Reposo: 60 s sin tráfico (consumo base)"
"$PYTHON" locust/monitor_recursos.py --salida "$SALIDA/${PREFIJO}reposo_recursos.csv" --duracion 60

# usuarios:tasa de arranque por segundo
for escenario in 10:2 50:5 100:10; do
  usuarios="${escenario%%:*}"
  tasa="${escenario##*:}"
  nombre="${PREFIJO}u${usuarios}"
  echo "==> Escenario $nombre: $usuarios usuarios, $tasa/s, $DURACION"
  "$PYTHON" locust/monitor_recursos.py --salida "$SALIDA/${nombre}_recursos.csv" --duracion "$((SEG + 5))" &
  monitor=$!
  "$PYTHON" -m locust -f locust/locustfile.py --headless -u "$usuarios" -r "$tasa" -t "$DURACION" \
    --csv "$SALIDA/$nombre" --html "$SALIDA/$nombre.html" --only-summary || true
  wait "$monitor"
  sleep 10  # deja que el sistema se estabilice entre escenarios
done

echo "==> Gráficas y resumen"
"$PYTHON" locust/graficas.py --escenarios "${PREFIJO}u10" "${PREFIJO}u50" "${PREFIJO}u100" --prefijo "$PREFIJO"
