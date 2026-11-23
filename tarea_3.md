Construye | Tests + Chequeos de seguridad básicos integrados al repositorio | Colaborativo
Fecha de entrega: dom 1 de nov de 2026 23:59
Intento 1
En progreso
SIGUIENTE: Presentar tarea

Unlimited Attempts Allowed

Objetivo(s)
 
Integrar al repositorio del proyecto integrador un conjunto de pruebas automatizadas (con PyTest y DeepChecks) y chequeos de seguridad básicos, con el fin de asegurar la calidad, confiabilidad y buenas prácticas de seguridad del código y los datos del proyecto.
Verificar mediante pruebas automatizadas que el entrenamiento de modelos sea reproducible (por ejemplo, con semillas fijas y resultados consistentes).
Esta actividad corresponde a la fase colaborativa del entregable de la semana y refuerza las competencias de testing y gobernanza técnica sobre el proyecto que el equipo ha venido construyendo.

 


Instrucciones
 
En esta actividad colaborativa, tu equipo añadirá una capa de calidad y seguridad al repositorio del proyecto. El objetivo es que el código cuente con pruebas que se puedan ejecutar de forma automatizada y que se hayan revisado los riesgos de seguridad más comunes.

Realiza, junto con tu equipo, los siguientes pasos:

Escribe pruebas unitarias con PyTest: Crea pruebas para al menos tres funciones clave del proyecto (procesamiento de datos, feature engineering, entrenamiento o utilidades), cubriendo tanto casos normales como casos límite. Incluye al menos una prueba que verifique la reproducibilidad del entrenamiento (por ejemplo, que dos ejecuciones con la misma semilla produzcan las mismas predicciones o métricas).
Integra validación de datos con DeepChecks: Añade chequeos que validen la integridad y la calidad del dataset (por ejemplo, tipos de datos, valores faltantes, duplicados o drift entre conjuntos de entrenamiento y prueba).
Realiza chequeos de seguridad básicos: Revisa que el repositorio no exponga credenciales, contraseñas ni datos sensibles (usa un archivo .gitignore y variables de entorno), y ejecuta al menos una herramienta de análisis de seguridad de dependencias o de código (por ejemplo, pip-audit o bandit).
Organiza las pruebas en el repositorio: Ubica las pruebas en una carpeta dedicada (por ejemplo, tests/) siguiendo la estructura del proyecto, de modo que se puedan ejecutar con un solo comando.
Documenta la ejecución: Actualiza el README con las instrucciones para correr las pruebas y los chequeos, e incluye evidencia (capturas) de su ejecución exitosa.
Refleja el trabajo colaborativo: Asegura que las contribuciones de todos los integrantes queden registradas en el historial de Git mediante ramas y commits significativos.
 


Especificaciones de entrega
 
Para considerar completa la entrega, asegúrate de cumplir con lo siguiente:

Modalidad: Colaborativa (en equipo de 4 personas).
La evaluación es individual con base en el entregable del equipo y en la contribución evidenciada en el historial de Git.
Medio de realización/entrega: Enlace al repositorio de GitHub del equipo, subido a la plataforma Canvas, con las pruebas y chequeos integrados.
El repositorio debe ser accesible para el profesor.
Formato: 1) Repositorio de GitHub con una carpeta de pruebas (por ejemplo, tests/), 2) código de PyTest y DeepChecks, 3) archivo .gitignore, 4) README actualizado con instrucciones de ejecución y evidencia (capturas) de la ejecución de pruebas y del chequeo de seguridad.
Nombre del entregable: Se entrega sobre el mismo repositorio del proyecto, TC5061-Proyecto-Equipo[Número].
Ejemplo: TC5061-Proyecto-Equipo03.
 


Criterios de evaluación
 
La actividad se evaluará con base en la siguiente rúbrica. Los pesos suman 100% y se aplican sobre el puntaje total de la actividad:

Descripción de criterios
Criterio	Excelente (100%)	Satisfactorio (70%)	Insuficiente (<50%)	Peso
Pruebas unitarias con PyTest: Las pruebas cubren funciones clave del proyecto, incluyen casos normales y límite, y se ejecutan correctamente.

Diseña y ejecuta pruebas unitarias que cubren todas las funciones clave del proyecto. Incluye casos normales y límite, evidenciando su correcta implementación.	Crea y ejecuta pruebas unitarias para la mayoría de las funciones clave del proyecto. Incluye algunos casos límite y asegura su funcionamiento.	Realiza pruebas unitarias básicas que cubren algunas funciones clave, pero no incluye casos límite o presenta fallos en la ejecución.	30%
Validación de datos con DeepChecks: Se integran chequeos pertinentes que validan la calidad e integridad del dataset.

Integra chequeos exhaustivos que validan la calidad e integridad del dataset. Detecta y documenta problemas como valores faltantes, duplicados o drift.	Implementa chequeos relevantes para validar la calidad e integridad del dataset, aunque no detecta todos los posibles problemas.	Realiza chequeos básicos, pero no abarca la validación completa de la calidad o integridad del dataset.	25%
Chequeos de seguridad básicos: El repositorio no expone datos sensibles y se ejecutó al menos una herramienta de análisis de seguridad, con evidencia de los resultados.

Implementa herramientas y prácticas de seguridad de manera completa, asegurando que no se expongan datos sensibles. Documenta evidencias claras de los resultados.	Utiliza herramientas y prácticas de seguridad, pero con algunas omisiones menores en la implementación o documentación.	Realiza chequeos básicos de seguridad, pero presenta omisiones importantes o no documenta adecuadamente.	25%
Organización y documentación: Las pruebas están bien organizadas y el README explica cómo ejecutarlas.

Organiza las pruebas en una estructura clara y funcional. Documenta detalladamente en el README cómo ejecutar las pruebas, incluyendo evidencia de su ejecución exitosa.	Organiza las pruebas en una estructura funcional, pero con detalles menores por mejorar. La documentación en el README es clara, aunque incompleta en algunos puntos.	La organización de las pruebas es confusa o poco funcional. La documentación en el README es limitada o carece de instrucciones claras.	10%
Trabajo colaborativo: El historial de Git evidencia la participación de todos los integrantes.

El historial de Git refleja una contribución equilibrada y significativa de todos los integrantes del equipo, con ramas y commits descriptivos.	El historial de Git evidencia la participación de la mayoría de los integrantes, aunque con contribuciones desiguales o commits poco detallados.	El historial de Git muestra poca participación de algunos integrantes o contribuciones mínimas y poco significativas.	10%
 

 


Recursos de apoyo
 
Para realizar esta actividad puedes apoyarte en los siguientes recursos:

Material de lectura de la semana (Wilson capítulo 15, Huyen capítulo 6, Grossmann et al.).
Documentación oficial de PyTest: https://docs.pytest.org/en/stable/
Documentación oficial de DeepChecks: https://docs.deepchecks.com/stable/getting-started/welcome.html
Herramientas de seguridad de código y dependencias: bandit y pip-audit.
Recursos en línea de la semana sobre PyTest aplicado a ciencia de datos.
Repositorio del proyecto del equipo, construido en las semanas anteriores.
 


Bibliografía
 
Grossmann, J., Wiesbrock, H., & Motta, M. (2021). Testing ML-Based Systems. ETSI.

Huyen, C. (2022). Designing machine learning systems: An iterative process for production-ready applications. O’Reilly Media.

Wilson, B. (2022). Machine Learning Engineering in Action. Manning Publications.

 

Actividad 7
Criterios	Calificaciones	Puntos
Pruebas unitarias con PyTest: Las pruebas cubren funciones clave del proyecto, incluyen casos normales y límite, y se ejecutan correctamente.

Excelente
Diseña y ejecuta pruebas unitarias que cubren todas las funciones clave del proyecto. Incluye casos normales y límite, evidenciando su correcta implementación.
30 pts

Satisfactorio
Crea y ejecuta pruebas unitarias para la mayoría de las funciones clave del proyecto. Incluye algunos casos límite y asegura su funcionamiento.
22.5 pts

Insuficiente
Realiza pruebas unitarias básicas que cubren algunas funciones clave, pero no incluye casos límite o presenta fallos en la ejecución.
14 pts
/30 pts
Validación de datos con DeepChecks: Se integran chequeos pertinentes que validan la calidad e integridad del dataset.

Excelente
Integra chequeos exhaustivos que validan la calidad e integridad del dataset. Detecta y documenta problemas como valores faltantes, duplicados o drift.
25 pts

Satisfactorio
Implementa chequeos relevantes para validar la calidad e integridad del dataset, aunque no detecta todos los posibles problemas.
18.75 pts

Insuficiente
Realiza chequeos básicos, pero no abarca la validación completa de la calidad o integridad del dataset.
11 pts
/25 pts
Chequeos de seguridad básicos: El repositorio no expone datos sensibles y se ejecutó al menos una herramienta de análisis de seguridad, con evidencia de los resultados.

Excelente
Implementa herramientas y prácticas de seguridad de manera completa, asegurando que no se expongan datos sensibles. Documenta evidencias claras de los resultados.
25 pts

Satisfactorio
Utiliza herramientas y prácticas de seguridad, pero con algunas omisiones menores en la implementación o documentación.
18.75 pts

Insuficiente
Realiza chequeos básicos de seguridad, pero presenta omisiones importantes o no documenta adecuadamente.
11 pts
/25 pts
Organización y documentación: Las pruebas están bien organizadas y el README explica cómo ejecutarlas.

Excelente
Organiza las pruebas en una estructura clara y funcional. Documenta detalladamente en el README cómo ejecutar las pruebas, incluyendo evidencia de su ejecución exitosa.
10 pts

Satisfactorio
Organiza las pruebas en una estructura funcional, pero con detalles menores por mejorar. La documentación en el README es clara, aunque incompleta en algunos puntos.
7.5 pts

Insuficiente
La organización de las pruebas es confusa o poco funcional. La documentación en el README es limitada o carece de instrucciones claras.
4 pts
/10 pts
Trabajo colaborativo: El historial de Git evidencia la participación de todos los integrantes.

Excelente
El historial de Git refleja una contribución equilibrada y significativa de todos los integrantes del equipo, con ramas y commits descriptivos.
10 pts

Satisfactorio
El historial de Git evidencia la participación de la mayoría de los integrantes, aunque con contribuciones desiguales o commits poco detallados.
7.5 pts

Insuficiente
El historial de Git muestra poca participación de algunos integrantes o contribuciones mínimas y poco significativas
4 pts
/10 pts
