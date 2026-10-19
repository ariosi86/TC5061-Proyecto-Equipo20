# Guía de Inicio Rápido (Getting Started)

Esta guía paso a paso contiene todo lo necesario para que cualquier integrante del equipo clone el repositorio, configure su entorno virtual, ejecute la ingesta de datos y comience a trabajar con el proyecto **MiBici**.

---

## 1. Clonar el Repositorio y Configurar el Entorno

Asegúrate de tener instalado **Python 3.10 o 3.11** en tu sistema.

### Paso 1: Clonar el proyecto
```bash
git clone https://github.com/ariosi86/TC5061-Proyecto-Equipo20.git
cd TC5061-Proyecto-Equipo20
```

### Paso 2: Crear el entorno virtual
```bash
python -m venv .venv
```

### Paso 3: Activar el entorno virtual

* **En macOS y Linux:**
  ```bash
  source .venv/bin/activate
  ```
* **En Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **En Windows (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

### Paso 4: Instalar dependencias y registrar el kernel
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name=mibici-venv --display-name="Python (MiBici .venv)"
```

---

## 2. Ingesta y Procesamiento de Datos

Ejecuta el módulo de ingesta para descargar los registros originales y generar los archivos Parquet canónicos:

```bash
python -m proyecto_mibici.dataset
```

Esto generará automáticamente:
* `data/processed/trips.parquet` (Maestro de viajes)
* `data/processed/stations.parquet` (Catálogo de estaciones)
* Particiones de entrenamiento y prueba en `data/processed/train/` y `data/processed/test/`
* `data/manifest.json` (Manifiesto con hashes SHA-256)

---

## 3. Ejecución del Orquestador End-to-End

Para correr todo el flujo de manera automatizada (Ingesta, Entrenamientos con MLflow, Inferencia y Pytest):

```bash
python -m proyecto_mibici.pipeline
```

