"""
Autor: Luis A. García
Email: luisgarcia.oq95@gmail.com
Creado: 7 de noviembre del 2025

Este código está diseñado para levantar la API y hacer predicciones
con el modelo de clasificación
sus funciones principales son:
    • Configura el sistema de registros que permite monitorear
        el comportamiento de la aplicación durante la ejecución.
    • Establece el ciclo de vida (lifespan) para la aplicación
        lo que permite controlar las acciones que ocurren antes
        de que la aplicación comience a antender peticiones y 
        después de que se apague.
      usadas en entrenamiento.
    • Crea la API con FastAPI
    • Permite reportar el estado del servidor MLflow
    • Permite cargar el modelo en producción y hacer predicciones
"""


# =============================================
# 1. IMPORTACIONES GENERALES (lo básico de FastAPI)
# =============================================
from fastapi import FastAPI,HTTPException,UploadFile,File,Request
from contextlib import asynccontextmanager # General: para manejar ciclo de vida
import logging                             # General: para obtener los log de la aplicación
import time
import json
from pathlib import Path
import os

# =============================================
# 2. Importar los ódigos creados para comunicarnos con MLflow y validar datos  de entrada
# =============================================
# Usemos las dos clases que creamos MlflowHandler y ClassificationRequest
from helpers.schemas import ClassificationRequest
from registry.mlflow.mlflow_handler import MlflowHandler

# =============================================
# 3. CONFIGURACIÓN INICIAL
# =============================================
# Crear un sistema de registro que permite rastrear lo que sucede en la aplicación mientras se ejecuta.
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
logging.basicConfig(format = log_format, level = logging.INFO)

# configuremos las variables globales (caché)
ml_models = {}
service_handlers = {}
# carga credenciales de minio
credential_path = lambda file: os.path.join(os.getcwd(),'secrets',file)
credential_path = credential_path('credentials_minio.json')
with open(credential_path, 'r') as file:
    credentials_minio = json.load(file)
# =============================================
# 4. CICLO DE VIDA DE LA APP
# =============================================
@asynccontextmanager
async def lifespan(app: FastAPI): # primera función función que al iniciar la aplicación ejecuta los siguientes servicios
    """
    Ciclo de vida de la aplicación FastAPI.

    Inicializa los servicios necesarios al arrancar la aplicación, 
    incluyendo la conexión con el servidor MLflow y la carga del modelo 
    de producción en memoria caché.  
    Al detener la aplicación, libera los recursos y limpia los servicios 
    registrados.

    Parameters
    ----------
    app : FastAPI
        Instancia principal de la aplicación FastAPI.

    Process
    --------
    - Crea una instancia de `MlflowHandler` configurada con las credenciales de MinIO.
    - Carga el modelo `DecisionTree_CreditRiskModel` desde la etapa *Production* 
      de MLflow y lo almacena en caché.
    - Registra los eventos relevantes en el sistema de logs.
    - Tras finalizar la ejecución de la aplicación, limpia las variables 
      globales `service_handlers` y `ml_models`.

    Yields
    ------
    None
        Control temporal devuelto al manejador de contexto de FastAPI 
        durante la ejecución de la aplicación.

    Raises
    -----------
    Exception
        Si ocurre un error al intentar cargar el modelo desde MLflow 
        durante la inicialización.
    """
    service_handlers['mlflow'] = MlflowHandler(
        tracking_uri="http://localhost:5000",
        s3_endpoint=credentials_minio["endpoint_url"],
        aws_access_key=credentials_minio["aws_access_key_id"],
        aws_secret_key=credentials_minio["aws_secret_access_key"]
)
    logging.info("Inicializado MlflowHandler {}".format(type(service_handlers['mlflow'])))
    # cargar el modelo
    model_name = "DecisionTree_CreditRiskModel"
    try:
        ml_models[model_name] = service_handlers['mlflow'].get_production_sklearn_model(model_name)
        logging.info(f"Modelo '{model_name}' cargado y almacenado en caché.")
    except Exception as e:
        logging.error(f"No se pudo cargar el modelo '{model_name}' al iniciar: {e}")
    
    yield
    # lo que se ejecuta después de cerrar la aplicación
    service_handlers.clear()
    ml_models.clear()
    logging.info("Servicios Handlers y ml models limpiados")

# =============================================
# 5. CREAR LA APLICACIÓN FASTAPI
# =============================================
app = FastAPI(
    title="API de Evaluación de Riesgo Crediticio",
    description="Servicio que estima la probabilidad de incumplimiento " \
    "de un préstamo dentro de los 12 meses" \
    "posteriores a su adquisición.",
    lifespan=lifespan  # Conectar el ciclo de vida
)

# =============================================
# 6. DEFINIR EL ENDPOINT HTTP GET PARA COMPROBAR ESTADO DEL SERVIDOR MLFLOW
# =============================================
@app.get("/health/", status_code=200)
async def healthcheck():
    """
    Verifica el estado general del servicio y del servidor de MLflow.

    Este endpoint permite comprobar que la API está activa y que los 
    servicios asociados a MLflow (tanto el servidor de tracking como 
    el registro de modelos) están funcionando correctamente. 
    Además, devuelve la lista de modelos actualmente en la etapa 
    *Production* dentro del registro de MLflow.

    Returns:
        dict: Un diccionario con información sobre el estado de la API 
        y del entorno de MLflow, incluyendo:
            - **serviceStatus** (`str`): Estado general del servicio (`"OK"` si está activo).
            - **modelTrackingHealth** (`bool`): Estado del servidor de tracking de MLflow.
            - **modelRegistryHealth** (`bool`): Estado del registro de modelos.
            - **productionModels** (`list[str]`): Lista de modelos en la etapa *Production*.
    """
    return {
        "serviceStatus": "OK",
        "modelTrackingHealth": service_handlers['mlflow'].check_mlflow_health(),
        "modelRegistryHealth": service_handlers['mlflow'].check_registry_health(),
        "productionModels": service_handlers['mlflow'].list_production_models()
    }
# =============================================
# 7. DEFINIR EL ENDPOINT HTTP GET PARA OBTENER INFORMACIÓN COMPLETA DEL SERVIDOR MLFLOW
# =============================================
@app.get("/debug/mlflow/", status_code=200)
async def debug_mlflow():
    return service_handlers['mlflow'].debug_registry()

# =============================================
# 8. DEFINIR EL ENDPOINT HTTP POST PARA OBTENER LAS PREDICCIONES
# =============================================
@app.post("/classify/", status_code=200)
async def classify(file: UploadFile | None = File(None), request: Request = None):
    """
    Endpoint de clasificación de riesgo crediticio.

    Este endpoint recibe datos de entrada en formato CSV o JSON, los transforma en un 
    DataFrame válido mediante `ClassificationRequest`, y utiliza un modelo de 
    `scikit-learn` cargado desde caché para generar predicciones de probabilidad 
    por clase. 

    Si el modelo no se encuentra en caché, se lanza un error HTTP 500.  
    En caso de error en el procesamiento de los datos o la inferencia, 
    se devuelve un error HTTP 400.

    Parameters
    ----------
    file : UploadFile | None, opcional
        Archivo CSV cargado por el usuario con las observaciones a clasificar.
    request : Request, opcional
        Objeto de solicitud HTTP, usado para detectar si el contenido proviene de un
        archivo o de un cuerpo JSON.

    Return
    -------
    dict
        Un diccionario con la siguiente información:
        - **n_observaciones**: número de registros clasificados.
        - **clases**: lista con los nombres de las clases del modelo.
        - **predicciones**: lista de diccionarios con las probabilidades por clase.
        - **datos**: representación de los datos de entrada como diccionario de listas.
        - **tiempo_inferencia_seg**: tiempo total de inferencia en segundos.

    Raises
    -----------
    HTTPException
        - 400: Si ocurre un error al procesar los datos o realizar la clasificación.
        - 500: Si el modelo no se encuentra disponible en caché.
    """
    try:
        content_type = request.headers.get("content-type", "")
        if file:
            if file.filename.endswith(".csv"):
                content = await file.read()
                decoded = content.decode("utf-8")
                df = ClassificationRequest.from_input(decoded)
        
        elif "application/json" in request.headers.get("content-type", ""):
            data = await request.json()
            df = ClassificationRequest.from_input(data)

        else:
            return {"error": "Formato no reconocido"}
        
        
        model_name = "DecisionTree_CreditRiskModel"
        if model_name not in ml_models:
            raise HTTPException(status_code=500, detail="Modelo no encontrado en caché")
        model = ml_models[model_name]

        start = time.time()
        probabilities = model.predict_proba(df)

        classes = getattr(model, "clase_", [f"Clase_{i}" for i in range(probabilities.shape[1])])
        results = [
                {cls: float(prob) for cls, prob in zip(classes, row)}
                for row in probabilities
                ]
        inference_time = round(time.time() - start, 4)
        return {
                "n_observaciones": len(df),
                "clases": list(classes),
                "predicciones": results,
                "datos": df.to_dict(orient="list"),
                "tiempo_inferencia_seg": inference_time
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la clasificación: {e}")

