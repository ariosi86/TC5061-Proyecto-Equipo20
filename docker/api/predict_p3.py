from .schemas import P3Input, P3Output

P3_FEATURES = [
    "pct_manana", "pct_mediodia", "pct_tarde", "pct_noche",
    "pct_fin_de_semana", "duracion_media", "viajes_dia_log", "balance_neto",
]


def predecir(modelo, datos: P3Input) -> P3Output:
    df = datos.to_dataframe()
    pred = modelo.predict(df)
    return P3Output(
        grupo_kmeans=int(pred[0]),
    )
