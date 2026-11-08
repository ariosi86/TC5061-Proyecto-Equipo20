from schemas import P3Input, P3Output


def predecir(modelo, datos: P3Input, ctx: dict) -> P3Output:
    """Asigna el grupo funcional (KMeans K=4) a una estación.

    `ctx` es el diccionario que devuelve `load_model_and_params()`; de ahí salen
    el nombre, la versión y el run_id del campeón que se reportan en la salida.
    """
    pred = modelo.predict(datos.to_dataframe())
    return P3Output(
        modelo=ctx["model_name"],
        version=ctx["version"],
        run_id=ctx["run_id"],
        grupo_kmeans=int(pred[0]),
    )
