from .schemas import P2Input, P2Output

P2_FEATURES = [
    "hora_del_dia", "dia_semana", "fin_de_semana", "capacidad",
    "salidas", "llegadas", "balance_acumulado_dia", "acumulado_relativo",
    "balance_ult_3h_relativo", "riesgo_vacia", "riesgo_llena",
]


def predecir(modelo, datos: P2Input) -> P2Output:
    df = datos.to_dataframe()
    pred = modelo.predict(df)
    try:
        proba = modelo.predict_proba(df)[:, 1]
        prob_riesgo = float(proba[0])
    except Exception:
        prob_riesgo = float(pred[0])
    return P2Output(
        prob_riesgo=prob_riesgo,
        pred_riesgo=int(pred[0]),
    )
