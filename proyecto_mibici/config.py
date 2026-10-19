from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Rutas de artefactos de datos y metadatos
MANIFEST_PATH = DATA_DIR / "manifest.json"
STATIONS_PROCESSED_PATH = PROCESSED_DATA_DIR / "stations.parquet"
TRIPS_PROCESSED_PATH = PROCESSED_DATA_DIR / "trips.parquet"

TRIPS_TRAIN_PATH = PROCESSED_DATA_DIR / "train" / "trips_train_2025.parquet"
TRIPS_TEST_PATH = PROCESSED_DATA_DIR / "test" / "trips_test_2026_04_09.parquet"

# ==========================================
# 2. PARÁMETROS GLOBALES Y TRACKING
# ==========================================
RANDOM_SEED = 42
MLFLOW_EXPERIMENT_NAME = "mibici-equipo20"

URI_MLFLOW_TRACKING = "http://127.0.0.1:5000"

# ==========================================
# 3. FUENTES Y ENLACES DE DESCARGA DIRECTA
# ==========================================
#  los enlaces directos obtenidos de https://mibici.net/es/datos-abiertos/
MIBICI_HOST = "https://www.mibici.net"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-419,es;q=0.9,en;q=0.8",
    "Referer": "https://mibici.net/es/datos-abiertos/",
}

DATA_URLS = {
    # Nomenclatura más reciente
    "stations": f"{MIBICI_HOST}/site/assets/files/1118/nomenclatura_2026_09.csv",
    
    # Año 2025 completo (12 meses cerrados)
   
    "trips": {
    "2025-01": f"{MIBICI_HOST}/site/assets/files/97653/datos_abiertos_2025_01.csv",
    "2025-02": f"{MIBICI_HOST}/site/assets/files/99879/datos_abiertos_2025_02.csv",
    "2025-03": f"{MIBICI_HOST}/site/assets/files/101986/datos_abiertos_2025_03.csv",
    "2025-04": f"{MIBICI_HOST}/site/assets/files/103813/datos_abiertos_2025_04.csv",
    "2025-05": f"{MIBICI_HOST}/site/assets/files/105513/datos_abiertos_2025_05.csv",
    "2025-06": f"{MIBICI_HOST}/site/assets/files/107088/datos_abiertos_2025_06.csv",
    "2025-07": f"{MIBICI_HOST}/site/assets/files/108898/datos_abiertos_2025_07.csv",
    "2025-08": f"{MIBICI_HOST}/site/assets/files/111021/datos_abiertos_2025_08.csv",
    "2025-09": f"{MIBICI_HOST}/site/assets/files/113216/datos_abiertos_2025_09.csv",
    "2025-10": f"{MIBICI_HOST}/site/assets/files/115104/datos_abiertos_2025_10.csv",
    "2025-11": f"{MIBICI_HOST}/site/assets/files/116773/datos_abiertos_2025_11.csv",
    "2025-12": f"{MIBICI_HOST}/site/assets/files/118153/datos_abiertos_2025_12.csv",

    #Año 2026 (9 meses cerrados hasta septiembre)
    "2026-01": f"{MIBICI_HOST}/site/assets/files/120163/datos_abiertos_2026_01.csv",
    "2026-02": f"{MIBICI_HOST}/site/assets/files/121761/datos_abiertos_2026_02.csv",
    "2026-03": f"{MIBICI_HOST}/site/assets/files/123236/datos_abiertos_2026_03.csv",
    "2026-04": f"{MIBICI_HOST}/site/assets/files/124927/datos_abiertos_2026_04.csv",
    "2026-05": f"{MIBICI_HOST}/site/assets/files/125943/datos_abiertos_2026_05.csv",
    "2026-06": f"{MIBICI_HOST}/site/assets/files/127734/datos_abiertos_2026_06.csv",
    "2026-07": f"{MIBICI_HOST}/site/assets/files/129006/datos_abiertos_2026_07.csv",
    "2026-08": f"{MIBICI_HOST}/site/assets/files/130362/datos_abiertos_2026_08.csv",
    "2026-09": f"{MIBICI_HOST}/site/assets/files/131825/datos_abiertos_2026_09.csv",
    
    }
}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# ==========================================
# 4. P1: DEMANDA HORARIA
# ==========================================
P1_TARGET = "target_next_hour"
P1_MAX_PCT_VACIAS = 90

# Partición temporal, sobre la hora objetivo (target_hour)
P1_VAL_INICIO = "2026-01-01"
P1_TEST_INICIO = "2026-05-01"
P1_TEST_FIN = "2026-10-01"

P1_FEATURES_V1 = [
    "lag_1h", "lag_24h", "lag_168h",
    "target_hour_sin", "target_hour_cos", "target_is_weekend",
]
P1_FEATURES_V2 = P1_FEATURES_V1 + ["rolling_3h", "rolling_168h"]



# ==========================================
# CONFIGURACIÓN P2: CLASIFICACIÓN DE RIESGO
# ==========================================
P2_FIN_PERIODO = "2026-06-30 23:00"
P2_INICIO_VALIDACION = "2026-01-01"
P2_INICIO_PRUEBA = "2026-04-01"
P2_MES_ESTACION_NUEVA = "2026_04"
P2_HORA_INICIO_DIA = 3
P2_UMBRAL = 0.5

P2_VARIABLES = [
    "hora_del_dia", 
    "dia_semana", 
    "fin_de_semana", 
    "capacidad",
    "salidas", 
    "llegadas", 
    "balance_acumulado_dia", 
    "acumulado_relativo",
    "balance_ult_3h_relativo", 
    "riesgo_vacia", 
    "riesgo_llena", 
    "en_riesgo",
]
P2_OBJETIVO = "en_riesgo_siguiente"

P2_MODEL_TYPE = "LogisticRegression" # o "RandomForestClassifier", etc.
P2_RANDOM_STATE = 42


# ==========================================
# CONFIGURACIÓN P3: CLUSTERING DE ESTACIONES
# ==========================================
P3_TRAIN_INICIO = "2025-01-01"
P3_TRAIN_FIN = "2026-01-01"
P3_TEST_INICIO = "2026-05-01"
P3_TEST_FIN = "2026-10-01"

P3_VARIABLES = [
    "pct_manana",
    "pct_mediodia",
    "pct_tarde",
    "pct_noche",
    "pct_fin_de_semana",
    "duracion_media",
    "viajes_dia_log",
    "balance_neto",
]
P3_K_ELEGIDO = 4
P3_MIN_VIAJES_ESTACION = 100
P3_DURACION_MAX_MIN = 180
