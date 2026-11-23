"""Gráficas y tabla resumen de las pruebas de carga (Locust) y del consumo de recursos (docker stats).

Lee, para cada escenario, los archivos que generan locust (--csv) y monitor_recursos.py:
    reports/locust/<escenario>_stats.csv, <escenario>_stats_history.csv, <escenario>_recursos.csv

Uso (desde la raíz del repo):
    python locust/graficas.py --escenarios u10 u50 u100
Salidas en reports/locust/: resumen_escenarios.csv y figuras PNG.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DIR = Path("reports/locust")


def leer_csv(ruta: Path, **kwargs) -> pd.DataFrame:
    """En Windows locust escribe los CSV con la codificación del sistema (cp1252), no UTF-8."""
    try:
        return pd.read_csv(ruta, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(ruta, encoding="cp1252", **kwargs)


def cargar_historia(escenario: str) -> pd.DataFrame:
    df = leer_csv(DIR / f"{escenario}_stats_history.csv")
    df = df[df["Name"] == "Aggregated"].copy()
    df["t"] = df["Timestamp"] - df["Timestamp"].min()
    return df


def resumen_escenario(escenario: str) -> dict:
    stats = leer_csv(DIR / f"{escenario}_stats.csv")
    agg = stats[stats["Name"] == "Aggregated"].iloc[0]
    pred = stats[(stats["Name"] == "/predict")]
    fila = {
        "escenario": escenario,
        "usuarios": int(cargar_historia(escenario)["User Count"].max()),
        "peticiones": int(agg["Request Count"]),
        "fallas": int(agg["Failure Count"]),
        "tasa_error_pct": round(100 * agg["Failure Count"] / max(agg["Request Count"], 1), 2),
        "rps": round(agg["Requests/s"], 2),
        "p50_ms": agg["50%"],
        "p95_ms": agg["95%"],
        "p99_ms": agg["99%"],
        "predict_peticiones": int(pred["Request Count"].sum()),
        "predict_rps": round(pred["Requests/s"].sum(), 2),
    }
    recursos = DIR / f"{escenario}_recursos.csv"
    if recursos.exists():
        r = pd.read_csv(recursos)
        por = r.groupby("contenedor").agg(cpu_medio=("cpu_pct", "mean"), mem_max=("mem_mib", "max"))
        fila["cpu_total_medio_pct"] = round(por["cpu_medio"].sum(), 1)
        fila["mem_total_max_mib"] = round(por["mem_max"].sum(), 0)
        for c, v in por.iterrows():
            fila[f"cpu_{c}"] = round(v["cpu_medio"], 1)
            fila[f"mem_{c}"] = round(v["mem_max"], 0)
    return fila


def figura_series(escenarios: list[str], prefijo: str) -> None:
    fig, ejes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for e in escenarios:
        h = cargar_historia(e)
        ejes[0].plot(h["t"], h["Requests/s"], label=e)
        ejes[1].plot(h["t"], h["95%"], label=f"{e} p95")
        ejes[1].plot(h["t"], h["50%"], linestyle="--", alpha=0.6, label=f"{e} p50")
        tasa = 100 * h["Failures/s"] / h["Requests/s"].where(h["Requests/s"] > 0)
        ejes[2].plot(h["t"], tasa.fillna(0), label=e)
    ejes[0].set_ylabel("Peticiones/s")
    ejes[1].set_ylabel("Latencia (ms)")
    ejes[2].set_ylabel("Tasa de error (%)")
    ejes[2].set_xlabel("Segundos desde el inicio")
    for ax in ejes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=3)
    fig.suptitle("Pruebas de carga: throughput, latencia y errores")
    fig.tight_layout()
    fig.savefig(DIR / f"{prefijo}series.png", dpi=150)
    plt.close(fig)


def figura_resumen(res: pd.DataFrame, prefijo: str) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
    x = res["usuarios"].astype(str)
    ejes[0].bar(x, res["rps"], color="#3b6ea5")
    ejes[0].set_title("Peticiones por segundo")
    ejes[0].set_xlabel("Usuarios concurrentes")
    ejes[1].plot(x, res["p50_ms"], marker="o", label="p50")
    ejes[1].plot(x, res["p95_ms"], marker="o", label="p95")
    ejes[1].plot(x, res["p99_ms"], marker="o", label="p99")
    ejes[1].set_title("Latencia (ms)")
    ejes[1].set_xlabel("Usuarios concurrentes")
    ejes[1].legend()
    for ax in ejes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DIR / f"{prefijo}resumen.png", dpi=150)
    plt.close(fig)


def figura_recursos(escenarios: list[str], prefijo: str) -> None:
    disponibles = [e for e in escenarios if (DIR / f"{e}_recursos.csv").exists()]
    if not disponibles:
        return
    fig, ejes = plt.subplots(len(disponibles), 2, figsize=(11, 3.2 * len(disponibles)), squeeze=False)
    for i, e in enumerate(disponibles):
        r = leer_csv(DIR / f"{e}_recursos.csv", parse_dates=["timestamp"])
        r["t"] = (r["timestamp"] - r["timestamp"].min()).dt.total_seconds()
        for c, g in r.groupby("contenedor"):
            ejes[i][0].plot(g["t"], g["cpu_pct"], label=c)
            ejes[i][1].plot(g["t"], g["mem_mib"], label=c)
        ejes[i][0].set_ylabel(f"{e}\nCPU (%)")
        ejes[i][1].set_ylabel("Memoria (MiB)")
        for ax in ejes[i]:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    ejes[-1][0].set_xlabel("Segundos")
    ejes[-1][1].set_xlabel("Segundos")
    fig.suptitle("Consumo de recursos por contenedor durante la carga (docker stats)")
    fig.tight_layout()
    fig.savefig(DIR / f"{prefijo}recursos.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--escenarios", nargs="+", required=True)
    parser.add_argument("--prefijo", default="", help="Prefijo de los archivos de salida")
    args = parser.parse_args()

    res = pd.DataFrame([resumen_escenario(e) for e in args.escenarios])
    res.to_csv(DIR / f"{args.prefijo}resumen_escenarios.csv", index=False)
    print(res.to_string(index=False))
    figura_series(args.escenarios, args.prefijo)
    figura_resumen(res, args.prefijo)
    figura_recursos(args.escenarios, args.prefijo)
    print(f"Figuras en {DIR}/")


if __name__ == "__main__":
    main()
