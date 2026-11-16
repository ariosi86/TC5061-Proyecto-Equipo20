"""Paquete de monitoreo del proyecto MiBici (Entregable 8).

Expone la clase reutilizable :class:`EvidentlyMonitor` para que cualquier
integrante del equipo genere reportes de drift con Evidently sobre sus propios
conjuntos de referencia/actual, sin depender de P1.
"""

from proyecto_mibici.monitoring.evidently_monitor import EvidentlyMonitor

__all__ = ["EvidentlyMonitor"]
