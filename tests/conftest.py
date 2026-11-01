import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def viajes_chicos():
    """Viajes sintéticos: 1 estación (id=1), 1 viaje por hora durante 200 horas."""
    horas = pd.date_range("2023-01-01", periods=200, freq="h")
    return pd.DataFrame({
        "viaje_id": np.arange(200),
        "origen_id": 1,
        "inicio_del_viaje": horas,
    })


@pytest.fixture
def estaciones():
    return pd.DataFrame({"id": [1, 2]})


@pytest.fixture
def datos_modelado():
    """Tabla de modelado sintética (como la salida de GeneradorDemandaLagsP1) con ruido controlado."""
    from proyecto_mibici.config import P1_FEATURES_V2, P1_TARGET

    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "lag_1h": rng.poisson(3, n),
        "lag_24h": rng.poisson(3, n),
        "lag_168h": rng.poisson(3, n),
        "rolling_3h": rng.uniform(0, 6, n),
        "rolling_168h": rng.uniform(0, 6, n),
        "target_hour_sin": rng.uniform(-1, 1, n),
        "target_hour_cos": rng.uniform(-1, 1, n),
        "target_is_weekend": rng.integers(0, 2, n),
    })
    df[P1_TARGET] = rng.poisson(df["rolling_3h"].clip(0.1) + 1)
    return df[P1_FEATURES_V2], df[P1_TARGET]

@pytest.fixture
def carpeta_modelos(tmp_path, datos_modelado):
    """Crea una carpeta temporal con modelos de prueba entrenados para test_predict_p1."""
    from proyecto_mibici.config import P1_FEATURES_V1, P1_FEATURES_V2
    from proyecto_mibici.modeling.model_p1 import DemandaLinealP1

    X, y = datos_modelado
    
    # Entrenar y guardar modelo v1
    modelo_v1 = DemandaLinealP1(features=P1_FEATURES_V1).fit(X, y)
    modelo_v1.save(tmp_path / "p1_v1_model.pkl")

    # Entrenar y guardar modelo v2
    modelo_v2 = DemandaLinealP1(features=P1_FEATURES_V2).fit(X, y)
    modelo_v2.save(tmp_path / "p1_v2_model.pkl")

    return tmp_path