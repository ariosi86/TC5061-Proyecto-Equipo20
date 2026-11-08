from .schemas import P1Input, P1Output

P1_FEATURES = [
    "lag_1h", "lag_24h", "lag_168h",
    "target_hour_sin", "target_hour_cos", "target_is_weekend",
    "rolling_3h", "rolling_168h",
]


def predecir(modelo, datos: P1Input) -> P1Output:
    df = datos.to_dataframe()
    pred = modelo.predict(df)
    return P1Output(
        demanda_predicha=float(pred[0]),
    )
