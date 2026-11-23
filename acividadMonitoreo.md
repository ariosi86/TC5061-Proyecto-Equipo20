Construye | Monitoreo y reporte de drift en el proyecto | Colaborativo
Fecha de entrega: dom 15 de nov de 2026 23:59
Intento 1
En progreso
SIGUIENTE: Presentar tarea

Unlimited Attempts Allowed

Objetivo(s)
 
Implementar el monitoreo de los datos del modelo del proyecto MiBici con Evidently.
Detectar, interpretar y reportar el drift entre los datos de entrenamiento y datos de un periodo posterior.
Comparar dos versiones del modelo registradas en MLFlow mediante un ejercicio de A/B testing, para decidir con evidencia cuál ofrece mejor desempeño.
Esta actividad corresponde a una fase de entrega del proyecto del curso (Entregable 8): el equipo añade al proyecto la capa de observabilidad que permite mantener el modelo vigente a lo largo del tiempo.

Seguiremos trabajando con el proyecto MiBici, construido a partir de los datos abiertos de MiBici: esta entrega se construye sobre el mismo repositorio del equipo y conserva el trabajo de las entregas anteriores (modelo registrado, pruebas y API en contenedor). Trabajen con al menos uno de los modelos del proyecto (regresión, clasificación o clustering).

 


Instrucciones
 
En esta actividad integradora, como equipo, dotarán al proyecto de la capacidad de vigilarse a sí mismo. El objetivo es detectar cuándo los datos de producción se alejan de los de entrenamiento y decidir, con evidencia, si una nueva versión del modelo es mejor que la actual.

Realiza, junto con tu equipo, los siguientes pasos:

Prepara los conjuntos de datos: Definan un conjunto de referencia (el periodo de datos de MiBici con el que se entrenó el modelo) y un conjunto actual que represente el comportamiento en producción: un periodo posterior de los datos abiertos de MiBici (otro mes, temporada o año). Si su API de la semana 8 registra las peticiones que recibe, también pueden usar esos registros. Documenten qué periodos eligieron y por qué.
Genera el reporte de drift con Evidently: Utilicen Evidently para comparar ambos conjuntos y producir un reporte que muestre el data drift por variable y la calidad de los datos.
Interpreta los resultados: Analicen el reporte e identifiquen qué variables presentan drift significativo y qué implicaciones tendría para el desempeño del modelo.
Entrena una versión alternativa: Generen una segunda versión del modelo (por ejemplo, reentrenada incluyendo datos del periodo actual o con distintos hiperparámetros) y regístrenla en el Model Registry de MLFlow como una nueva versión del mismo modelo.
Realiza el A/B testing: Comparen las dos versiones del modelo sobre el mismo conjunto de evaluación del periodo actual, contrastando sus métricas, y determinen de forma justificada cuál debería quedar como versión vigente.
Elabora el reporte: Integren en un documento el reporte de drift (con gráficas de Evidently), el resultado del A/B testing y una recomendación sobre si convendría reentrenar el modelo (Continuous Training).
 


Especificaciones de entrega
 
Para considerar completa la entrega, asegúrate de cumplir con lo siguiente:

Modalidad: Colaborativa (en equipo de 4 personas).
La evaluación es individual con base en el entregable del equipo y en la contribución evidenciada en el historial de Git.
Medio de realización/entrega:
Reporte en PDF subido a la plataforma Canvas, que incluya el enlace al repositorio de GitHub del equipo con el código del monitoreo y del A/B testing.
El reporte de Evidently puede adjuntarse como HTML o exportarse como imagen dentro del documento.
El código de monitoreo y de A/B testing se integra al repositorio existente del proyecto MiBici, sin borrar ni reemplazar las entregas anteriores.
Formato: Reporte en PDF, tipografía Arial 11 o equivalente, interlineado 1.15, de 3 a 5 cuartillas (sin contar anexos).
Debe incluir: 1) gráficas del reporte de drift de Evidently, 2) tabla comparativa del A/B testing entre las dos versiones, y 3) la recomendación final.
Nombre del entregable: TC5061_Drift_ABTest_Equipo[Número].pdf, por ejemplo: TC5061_Drift_ABTest_Equipo03.pdf.
 


Criterios de evaluación
 
La actividad se evaluará con base en la siguiente rúbrica (los pesos suman 100% del valor de la actividad):

Descripción de criterios
Criterio	Excelente (100%)	Satisfactorio (70%)	Insuficiente (<50%)	Peso
Generación del reporte de drift con Evidently (conjuntos de referencia y actual bien definidos, reporte correctamente producido).	Define con claridad y precisión los conjuntos de referencia y actual. Genera un reporte completo y estructurado con Evidently, incluyendo gráficas y métricas relevantes.	Define los conjuntos de referencia y actual de manera adecuada, aunque con ligeros errores o ambigüedades. El reporte es funcional, pero incompleto o poco claro.	Presenta conjuntos de datos mal definidos o ambiguos. El reporte es deficiente, con información incompleta, incorrecta o ausente.	30%
Interpretación del drift (identificación de variables afectadas e implicaciones para el modelo).	Identifica con precisión las variables afectadas por el drift. Analiza de manera detallada las implicaciones para el desempeño del modelo, mostrando un razonamiento profundo.	Identifica las variables afectadas, pero con análisis parcial o limitado. Las implicaciones son mencionadas de forma general, sin suficiente profundidad.	Omite identificar variables clave o realiza un análisis superficial que no permite extraer conclusiones claras sobre las implicaciones del drift.	20%
A/B testing entre versiones (comparación metodológicamente correcta de dos versiones del modelo).	Realiza una comparación metodológicamente sólida y bien justificada entre las dos versiones del modelo, detallando métricas y resultados con claridad.	Realiza la comparación entre versiones con metodología adecuada, aunque con errores menores o explicaciones poco claras.	Realiza una comparación incompleta, con errores metodológicos o sin justificar adecuadamente las diferencias entre las versiones.	30%
Recomendación y razonamiento sobre Continuous Training (decisión justificada con base en la evidencia).	Presenta una recomendación clara y bien fundamentada, respaldada por los datos y el análisis realizado, demostrando un razonamiento crítico.	Presenta una recomendación adecuada, pero con fundamentos o justificaciones limitadas o ambiguas.	La recomendación es vaga, sin justificación suficiente o basada en un análisis incompleto o incorrecto.	10%
Claridad del reporte y trabajo colaborativo evidenciado en Git.	El reporte es claro, bien estructurado y profesional. Evidencia claramente el trabajo colaborativo en Git, con contribuciones balanceadas y organizadas.	El reporte es adecuado, pero presenta problemas menores de claridad o estructura. La evidencia en Git muestra participación desigual o desorganizada.	El reporte es confuso o desorganizado. La evidencia en Git es insuficiente o no refleja una colaboración efectiva entre los integrantes del equipo.	10%
 


Recursos de apoyo
 
Para realizar esta actividad puedes apoyarte en los siguientes recursos:

Material y grabación de la Sesión sincrónica 4 (monitoreo, drift, A/B testing y Continuous Training) y las lecturas de la semana (Huyen capítulo 8, Treveil capítulo 7).
Documentación oficial de Evidently: https://docs.evidentlyai.com/introduction
Tutorial de Evidently (serie de videos): Tutorial de Evidently
Detección de data drift con Evidently en Python: Artículo en Medium.
Documentación de MLFlow para registrar y comparar versiones del modelo: MLFlow Model Registry.
Ejemplo oficial de detección de data drift con la versión actual de Evidently: Data and ML checks.
Repositorio del proyecto MiBici del equipo, con el modelo y la operación construidos en las semanas anteriores.
Fuente de datos del proyecto: Datos abiertos de MiBici.
 


Bibliografía
 
Evidently AI. (s.f.). Evidently documentation. Recuperado el 18 de junio de 2026.

Huyen, C. (2022). Designing machine learning systems: An iterative process for production-ready applications. O’Reilly Media.

Treveil, M., Omont, N., Stenac, C., Lefevre, K., Phan, D., Zentici, J., Lavoillotte, A., Miyazaki, M., & Heidmann, L. (2020). Introducing MLOps: How to scale machine learning in the enterprise. O’Reilly Media.

 

Actividad 10
Criterios	Calificaciones	Puntos
Generación del reporte de drift con Evidently (conjuntos de referencia y actual bien definidos, reporte correctamente producido).

Excelente
Define con claridad y precisión los conjuntos de referencia y actual. Genera un reporte completo y estructurado con Evidently, incluyendo gráficas y métricas relevantes.
30 pts

Satisfactorio
Define los conjuntos de referencia y actual de manera adecuada, aunque con ligeros errores o ambigüedades. El reporte es funcional, pero incompleto o poco claro.
22.5 pts

Insuficiente
Presenta conjuntos de datos mal definidos o ambiguos. El reporte es deficiente, con información incompleta, incorrecta o ausente.
14 pts
/30 pts
Interpretación del drift (identificación de variables afectadas e implicaciones para el modelo).

Excelente
Identifica con precisión las variables afectadas por el drift. Analiza de manera detallada las implicaciones para el desempeño del modelo, mostrando un razonamiento profundo.
20 pts

Satisfactorio
Identifica las variables afectadas, pero con análisis parcial o limitado. Las implicaciones son mencionadas de forma general, sin suficiente profundidad
15 pts

Insuficiente
Omite identificar variables clave o realiza un análisis superficial que no permite extraer conclusiones claras sobre las implicaciones del drift.
10 pts
/20 pts
A/B testing entre versiones (comparación metodológicamente correcta de dos versiones del modelo).

Excelente
Realiza una comparación metodológicamente sólida y bien justificada entre las dos versiones del modelo, detallando métricas y resultados con claridad.
30 pts

Satisfactorio
Realiza la comparación entre versiones con metodología adecuada, aunque con errores menores o explicaciones poco claras.
22.5 pts

Insuficiente
Realiza una comparación incompleta, con errores metodológicos o sin justificar adecuadamente las diferencias entre las versiones
14 pts
/30 pts
Recomendación y razonamiento sobre Continuous Training (decisión justificada con base en la evidencia).

Excelente
Presenta una recomendación clara y bien fundamentada, respaldada por los datos y el análisis realizado, demostrando un razonamiento crítico.
10 pts

Satisfactorio
Presenta una recomendación adecuada, pero con fundamentos o justificaciones limitadas o ambiguas.
7.5 pts

Insuficiente
La recomendación es vaga, sin justificación suficiente o basada en un análisis incompleto o incorrecto.
4 pts
/10 pts
Claridad del reporte y trabajo colaborativo evidenciado en Git.

Excelente
El reporte es claro, bien estructurado y profesional. Evidencia claramente el trabajo colaborativo en Git, con contribuciones balanceadas y organizadas.
10 pts

Satisfactorio
El reporte es adecuado, pero presenta problemas menores de claridad o estructura. La evidencia en Git muestra participación desigual o desorganizada.
7.5 pts

Insuficiente
El reporte es confuso o desorganizado. La evidencia en Git es insuficiente o no refleja una colaboración efectiva entre los integrantes del equipo.
4 pts
/10 pts
