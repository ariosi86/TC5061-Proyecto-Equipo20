from schemas import P2Input, P2Output


def predecir(modelo, datos: P2Input, ctx: dict) -> P2Output:
    """Predice si la estación estará en riesgo (vacía o llena) en la hora siguiente.

    `ctx` es el diccionario que devuelve `load_model_and_params()`; de ahí salen
    el nombre, la versión y el run_id del campeón que se reportan en la salida.
    """
    if not hasattr(modelo, "predict_proba"):
        raise TypeError(
            "El modelo de P2 no tiene predict_proba; se esperaba el clasificador de "
            "scikit-learn y no un wrapper pyfunc"
        )

    df = datos.to_dataframe()
    # Columna de predict_proba que corresponde a la clase "en riesgo" (True o 1),
    # sin suponer que siempre es la segunda
    col_riesgo = list(modelo.classes_).index(True)
    prob_riesgo = float(modelo.predict_proba(df)[0, col_riesgo])
    pred_riesgo = int(modelo.predict(df)[0])

    return P2Output(
        modelo=ctx["model_name"],
        version=str(ctx["version"]),
        run_id=ctx["run_id"],
        prob_riesgo=prob_riesgo,
        pred_riesgo=pred_riesgo,
    )