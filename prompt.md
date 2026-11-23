a ver vas a correr especificamente run_train.py del proyecto que corre todos los modelos, revisa el proyecto es de cookiecutter y estamos integrando mlproyect

vamos a analizar la consola de la corrida que integra todo y vamos a Seleccionar el mejor modelo: Con base en la comparación de métricas en la interfaz de MLflow, elijan de forma justificada el modelo de cada pregunta que pasará a producción. Consideren el desempeño respecto al baseline, la métrica ligada a la decisión de negocio de cada pregunta, la estabilidad de los resultados y el costo de entrenamiento o inferencia. y Una tabla comparativa por pregunta con el baseline y las corridas candidatas: modelo, hiperparámetros clave, métricas de validación y prueba, y mejora respecto al baseline.
El análisis de la comparación y la justificación del modelo elegido por pregunta.
El esquema de ciclo de vida utilizado y su justificación. , todo el analisis se escribe en mejor_modelo.md en raiz con el mejor modelo de cada pregunta,vas a crear un archivo run_mejor_modelo.py donde Registrar el modelo en el Model Registry: Registren el modelo elegido de cada pregunta con un nombre claro y consistente, por ejemplo mibici-equipoNN-p1-demanda, mibici-equipoNN-p2-riesgo y mibici-equipoNN-p3-grupos. Cada versión debe conservar su linaje: experimento, corrida de origen, commit y versión de datos. Agreguen una descripción a cada modelo registrado.

las preguntas originales son 

Preguntas del proyecto integrador
#	Pregunta de negocio	Tipo de tarea	Quién la usaría
P1	¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro?	Regresión	Operador del sistema, para planear la flota.
P2	¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro?	Clasificación	Equipo de rebalanceo de bicicletas.
P3	¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad?	Clustering	Planeación urbana y expansión del sistema.
