"""Análisis FinOps: costo por inferencia a partir de las pruebas de carga y de docker stats.

Combina reports/locust/<prefijo>resumen_escenarios.csv (graficas.py) con precios de referencia
de nube y calcula el costo por inferencia con tres enfoques:

1. Instancia dedicada: precio por hora / inferencias por hora sostenidas en cada escenario.
2. Por consumo (tipo AWS Fargate): CPU-segundos y GB-segundos medidos por inferencia x precio unitario.
3. Tráfico real de MiBici: costo mensual fijo del sistema encendido 24/7 / inferencias del mes.

Los precios son on-demand de referencia (AWS us-east-1, Linux) y se pueden cambiar en PRECIOS.

Uso (desde la raíz del repo):
    python locust/finops.py --prefijos "" cache_
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DIR = Path("reports/locust")
HORAS_MES = 730

# Precios de referencia (USD). Verificar vigencia en la calculadora del proveedor antes de presentar.
INSTANCIAS = {
    # nombre: (vCPU, GiB, USD por hora)
    "t3.small": (2, 2, 0.0208),
    "t3.medium": (2, 4, 0.0416),
    "t3.large": (2, 8, 0.0832),
    "c6i.large": (2, 4, 0.0850),
    "m6i.xlarge": (4, 16, 0.1920),
}
FARGATE_VCPU_HORA = 0.04048
FARGATE_GB_HORA = 0.004445
DESCUENTO_SPOT = 0.65  # ahorro típico de instancias Spot frente a on-demand

# Tráfico real estimado: ~300 estaciones con una predicción por hora en cada pregunta (P1, P2)
# y una reasignación diaria de grupo (P3).
ESTACIONES = 300
INFERENCIAS_DIA = ESTACIONES * 24 * 2 + ESTACIONES


def costo_instancia(fila: pd.Series) -> dict:
    """Costo por inferencia si el sistema corre en una instancia dedicada al ritmo medido."""
    inferencias_hora = fila["predict_rps"] * 3600
    salida = {}
    for nombre, (_, _, usd_hora) in INSTANCIAS.items():
        salida[f"usd_por_1k_{nombre}"] = 1000 * usd_hora / inferencias_hora
    return salida


def costo_consumo(fila: pd.Series) -> dict:
    """Costo por inferencia pagando solo CPU y memoria usadas (modelo de precios de Fargate)."""
    cpu_seg_por_inf = (fila["cpu_total_medio_pct"] / 100) / fila["rps"]
    gb_seg_por_inf = (fila["mem_total_max_mib"] / 1024) / fila["rps"]
    usd = cpu_seg_por_inf * FARGATE_VCPU_HORA / 3600 + gb_seg_por_inf * FARGATE_GB_HORA / 3600
    return {
        "cpu_ms_por_inferencia": round(1000 * cpu_seg_por_inf, 1),
        "usd_por_1k_consumo": 1000 * usd,
    }


def figura_comparacion(tabla: pd.DataFrame) -> None:
    fig, ejes = plt.subplots(1, 3, figsize=(12, 3.6))
    for conf, g in tabla.groupby("configuracion", sort=False):
        x = g["usuarios"].astype(str)
        ejes[0].plot(x, g["rps_predict"], marker="o", label=conf)
        ejes[1].plot(x, g["p95_ms"], marker="o", label=conf)
        ejes[2].plot(x, g["cpu_ms_por_inferencia"], marker="o", label=conf)
    ejes[0].set_title("Inferencias por segundo")
    ejes[1].set_title("Latencia p95 (ms, log)")
    ejes[1].set_yscale("log")
    ejes[2].set_title("CPU por inferencia (ms)")
    for ax in ejes:
        ax.set_xlabel("Usuarios concurrentes")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(DIR / "comparacion_cache.png", dpi=150)
    plt.close(fig)


def figura_optimizaciones(opt: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.barh(opt["opcion"][::-1], opt["usd_mes"][::-1], color="#3b6ea5")
    for i, v in enumerate(opt["usd_mes"][::-1]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("USD por mes (tráfico real estimado)")
    ax.set_title("Costo mensual de servicio por opción de despliegue")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(DIR / "finops_optimizaciones.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefijos", nargs="+", default=[""], help='"" = línea base, cache_ = con caché')
    args = parser.parse_args()

    filas = []
    for prefijo in args.prefijos:
        res = pd.read_csv(DIR / f"{prefijo}resumen_escenarios.csv")
        for _, fila in res.iterrows():
            filas.append({
                "configuracion": "sin caché" if prefijo == "" else prefijo.rstrip("_"),
                "usuarios": fila["usuarios"],
                "rps_predict": fila["predict_rps"],
                "p95_ms": fila["p95_ms"],
                "tasa_error_pct": fila["tasa_error_pct"],
                "cpu_total_pct": fila["cpu_total_medio_pct"],
                "mem_total_mib": fila["mem_total_max_mib"],
                **costo_consumo(fila),
                **costo_instancia(fila),
            })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(DIR / "finops_costos.csv", index=False)
    pd.set_option("display.width", 220)
    print(tabla.round(4).to_string(index=False))

    # Tráfico real: el sistema encendido 24/7 en la instancia más chica en la que cabe (memoria)
    inferencias_mes = INFERENCIAS_DIA * 30
    print(f"\nTráfico real estimado: {INFERENCIAS_DIA:,} inferencias/día, {inferencias_mes:,}/mes")
    for nombre, (vcpu, gib, usd_hora) in INSTANCIAS.items():
        mensual = usd_hora * HORAS_MES
        print(
            f"  {nombre:11s} {vcpu} vCPU {gib:>2} GiB  USD {mensual:7.2f}/mes  "
            f"USD {1000 * mensual / inferencias_mes:.4f} por 1k inferencias"
        )

    # Capacidad máxima medida en la prueba de estrés (sin tiempo de espera)
    estres = DIR / "estres_resumen_escenarios.csv"
    if estres.exists():
        e = pd.read_csv(estres).iloc[0]
        vcpu_apis = sum(e[f"cpu_{c}"] for c in ("api-p1", "api-p2", "api-p3")) / 100
        por_vcpu = e["predict_rps"] / vcpu_apis
        print(
            f"\nCapacidad (estrés): {e['predict_rps']:.1f} inferencias/s con {vcpu_apis:.2f} vCPU de APIs "
            f"= {por_vcpu:.1f} inferencias/s por vCPU; p95 {e['p95_ms']:.0f} ms, errores {e['tasa_error_pct']}%"
        )
        for nombre, (vcpu, _, usd_hora) in INSTANCIAS.items():
            cap = por_vcpu * vcpu
            print(f"  {nombre:11s} capacidad ~{cap:5.0f} inf/s  USD {1e6 * usd_hora / (cap * 3600):.3f} por millón a plena carga")
        uso = INFERENCIAS_DIA / 86400 / (por_vcpu * 2)
        print(f"  Utilización con el tráfico real en 2 vCPU: {100 * uso:.2f}% de la capacidad")

    # Optimizaciones: costo mensual del sistema de servicio encendido 24/7 con el tráfico real
    usd = {k: v[2] * HORAS_MES for k, v in INSTANCIAS.items()}
    # CPU por inferencia medido con caché (última configuración pasada en --prefijos, mayor carga)
    cpu_s_por_inf = tabla.iloc[-1]["cpu_ms_por_inferencia"] / 1000
    consumo_cpu_h = inferencias_mes * cpu_s_por_inf / 3600
    opciones = [
        ("Referencia: t3.large on-demand (8 GiB de holgura)", usd["t3.large"]),
        ("Ajuste de tamaño: t3.medium (2.5 GiB medidos caben en 4)", usd["t3.medium"]),
        ("MLflow fuera de la ruta de servicio: t3.small (APIs ~0.6 GiB)", usd["t3.small"]),
        ("t3.small en Spot", usd["t3.small"] * (1 - DESCUENTO_SPOT)),
        ("Contenedor por consumo 0.25 vCPU / 0.5 GB 24/7 (Fargate)",
         HORAS_MES * (0.25 * FARGATE_VCPU_HORA + 0.5 * FARGATE_GB_HORA)),
        ("Solo CPU usada por las inferencias (escalar a cero, cota inferior)",
         consumo_cpu_h * FARGATE_VCPU_HORA),
    ]
    opt = pd.DataFrame(opciones, columns=["opcion", "usd_mes"])
    opt["usd_por_1k"] = 1000 * opt["usd_mes"] / inferencias_mes
    opt["ahorro_vs_referencia_pct"] = 100 * (1 - opt["usd_mes"] / opt["usd_mes"].iloc[0])
    opt.to_csv(DIR / "finops_optimizaciones.csv", index=False)
    print("\nOptimizaciones con el tráfico real:")
    print(opt.round(4).to_string(index=False))

    figura_comparacion(tabla)
    figura_optimizaciones(opt)

    # Figura: costo por 1k inferencias por escenario y configuración
    fig, ax = plt.subplots(figsize=(8, 4))
    for conf, g in tabla.groupby("configuracion", sort=False):
        ax.plot(g["usuarios"].astype(str), g["usd_por_1k_t3.large"], marker="o", label=f"{conf} (t3.large)")
    ax.set_yscale("log")
    ax.set_xlabel("Usuarios concurrentes")
    ax.set_ylabel("USD por 1,000 inferencias (log)")
    ax.set_title("Costo por inferencia en instancia dedicada")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DIR / "finops_costo_por_inferencia.png", dpi=150)
    print(f"\nTabla en {DIR / 'finops_costos.csv'} y figura finops_costo_por_inferencia.png")


if __name__ == "__main__":
    main()
