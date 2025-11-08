"""
Autor: Luis A. García
Email: luisgarcia.oq95@gmail.com
Creado: 6 de noviembre del 2025

Este código está diseñado para comunicarse con mlflow 
sus funciones principales son:
    • Comprobar si el servidor MLflow está disponible y responde correctamente 
    • Comprobar si existen modelos en fase de producción
    • Listar los modelos que tienen versiones en producción
    • Obtener detalles de los modelos registrados en MLflow.
"""
import mlflow
from mlflow.client import MlflowClient
import logging
import os

logger = logging.getLogger(__name__)
class MlflowHandler:
    """
    Se encarga de establecer conexión al registro de modelos de MLflow
    para seleccionar los modelos por verisón y etapas.
    """
    def __init__(self, tracking_uri="http://localhost:5000",
                 s3_endpoint=None, aws_access_key=None, aws_secret_key=None):
        """
        Configura la conexión con el servidor de MLflow y opcionalmente S3/MinIO.

        Attributes
        ----------
        tracking_uri: str
            URL del servidor MLflow, por defecto es http://localhost:5000
        s3_endpoint: str, optional
            Endpoint de S3/MinIO
        aws_access_key: str, optional
            Access key de S3/MinIO
        aws_secret_key: str, optional
            Secret key de S3/MinIO
        """
        # Configurar credenciales de S3/MinIO antes de inicializar MLflow
        if s3_endpoint and aws_access_key and aws_secret_key:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint
            os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key

        self.client = MlflowClient(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

    def check_mlflow_health(self):
        """
        Comprueba si el servidor MLflow está disponible y responde
        correctamente.

        Returns
        -------
        dict: Estado del servicio con estructura {
            'healthy': bool,
            'message': str,
            'experiment_count': int
        }
        """
        try:
            experiments = self.client.search_experiments()
            return {
                'healthy': True,
                'message': 'El servicio MLflow es receptivo',
                'experiment_count': len(experiments)
            }
        except ConnectionError as e:
            return {
                'healthy': False,
                'message': f'Connection error: {e}',
                'experiment_count': 0
            }
    def check_registry_health(self):
        """
        Este método consulta el registro de modelos de mlflow para comprobar si existe 
        al menos una versión de modelo en el estado "Production".

        Returns
        -------
         bool
            - True si al menos un modelo tiene una versión en producción.
            - False si no existen versiones en producción.

        Raise
        -----
        RuntimeError
            Si ocurre un error al intentar acceder o consultar el registro de modelos.
        """
        try:
            registered_models = self.client.search_registered_models()
            has_production_models = False
            for model in registered_models:
                # Verificar si este modelo tiene versión en producción
                for version in model.latest_versions:
                    if version.current_stage == "Production":
                        has_production_models = True
                        break
                if has_production_models:
                    break
            return has_production_models
        
        except Exception as e:
            raise RuntimeError("Error al chequear registro") from e  
    
    def list_production_models(self):
        """
        Lista todos los modelos registrados que tienen versiones en producción.

        Este método consulta el registro de modelos en MLflow y devuelve 
        información básica sobre cada modelo que cuenta con al menos una 
        versión en el estado "Production".

        Returns
        -------
        list[dict]:
            Una lista de diccionarios, donde cada elemento representa un modelo
            con las siguientes claves:
    
            - "name" (str): nombre del modelo registrado.  
            - "version" (str): número de versión en producción.  
            - "description" (str | None): descripción asociada a la versión. 
        
        Raises
        ------
        Exception
            Si ocurre algún error
        """
        try:
            models = self.client.search_registered_models()
            production_models = []
            
            for model in models:
                for version in model.latest_versions:
                    if version.current_stage == "Production":
                        production_models.append({
                            "name": model.name,
                            "version": version.version,
                            "description": version.description
                        })
            return production_models
        except Exception as e:
            return [f"Error: {e}"]    
    
    def get_production_model(self, model_name: str):
        """
        Carga un modelo específico desde el Model Registry en la etapa "Production".

        Este método busca el modelo indicado en el registro de MLflow y carga 
        la versión que se encuentra en el estado "Production". 
        Si el modelo no existe o no se puede cargar, se lanza una excepción.

        Attributes
        ----------
        model_name: str
            nombre del modelo registrado en MLflow.
        
        Returns
        -------
            mlflow.pyfunc.PyFuncModel: Instancia del modelo cargado desde MLflow.
        
        Raises
        ------
            ValueError: Si el modelo no existe en el registro.
            RuntimeError: Si ocurre un error al intentar cargar el modelo desde MLflow.
        """
        try:
            model_details = self.get_model_details(model_name)
            if not model_details:
                raise Exception(f"Modelo {model_name} no encontrado en el registry")
            
            model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/production")
            logger.info(f"Modelo '{model_name}' cargado exitosamente desde Production.")
            return model
            
        except Exception as e:
            logger.error(f"Error cargando modelo '{model_name}' desde Production: {e}")
            raise RuntimeError(f"Error cargando modelo '{model_name}' desde Production.") from e
        
    def get_production_sklearn_model(self, model_name: str):
        """
        Carga un modelo de scikit-learn desde el Model Registry de MLflow 
        en la etapa "Production".

        Este método obtiene el modelo registrado en MLflow usando el flavor
        específico de `mlflow.sklearn`, devolviendo el objeto original de 
        scikit-learn. A diferencia del modelo genérico `PyFuncModel`, este 
        permite acceder a métodos nativos como `predict_proba()`, `score()` 
        y otros disponibles en la clase base de scikit-learn.

        Parameters
        ----------
        model_name : str
            Nombre del modelo registrado en MLflow que se desea cargar.

        Returns
        -------
        sklearn.base.BaseEstimator
            Instancia del modelo de scikit-learn cargado desde MLflow en la 
            etapa "Production".

        Raises
        ------
        RuntimeError
            Si ocurre un error durante la carga del modelo o si no puede 
            accederse al registro de MLflow.
        """
        try:
            model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
            logger.info(f"Modelo sklearn '{model_name}' cargado desde Production.")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo sklearn '{model_name}': {e}")
            raise RuntimeError(f"Error cargando modelo sklearn '{model_name}'") from e

    def get_model_details(self, model_name: str):
        """
        Obtiene los detalles de un modelo específico registrado en MLflow.

        Este método consulta el registro de modelos (Model Registry) de MLflow
        para recuperar información del modelo identificado por `model_name`.
        Retorna un diccionario con el nombre del modelo y una lista de sus
        versiones más recientes, incluyendo su etapa (stage) y descripción.

        Attributes
        ----------
        model_name: str
            Nombre del modelo registrado en MLflow
        
        Return
        ------
        dict
            Un diccionario con información relevante del modelo
        
        Raises
        ------
        None explícitamente.
            (Los errores se registran mediante `logger.error` y la función devuelve `None`.)
        """
        try:
            model = self.client.get_registered_model(model_name)
            details = {
                "name": model.name,
                "versions": []
            }
            
            for version in model.latest_versions:
                details["versions"].append({
                    "version": version.version,
                    "stage": version.current_stage,
                    "description": version.description
                })
            
            return details
        except Exception as e:
            logger.error(f"Error '{model_name}' no encontrado: {e}")
            return None 
           
    def debug_registry(self):
        """
        Obtiene información completa del estado actual del Model Registry de MLflow.

        Esta función se utiliza con fines de depuración o auditoría para inspeccionar 
        el estado completo del registro de modelos. Recupera todos los modelos 
        registrados y sus versiones más recientes, incluyendo su nombre, descripción, 
        estado, etapa (stage) y versiones que están actualmente en producción.

        Returns
        -------
            dict: Diccionario con información general y detallada del registro.

        Raises
        ------
        None explícitamente.
            (Los errores se capturan internamente y se devuelven en la respuesta como texto.)
        """
        try:
            all_models = self.client.search_registered_models()
            debug_info = {
                "total_models": len(all_models),
                "models": [],
                "production_models": []
            }
            
            for model in all_models:
                model_info = {
                    "name": model.name,
                    "description": model.description,
                    "versions": []
                }
                
                for version in model.latest_versions:
                    version_info = {
                        "version": version.version,
                        "stage": version.current_stage,
                        "status": version.status
                    }
                    model_info["versions"].append(version_info)
                    
                    if version.current_stage == "Production":
                        debug_info["production_models"].append(f"{model.name} v{version.version}")
                
                debug_info["models"].append(model_info)
            
            return debug_info
            
        except Exception as e:
            return {"error": f"Debug failed: {e}"}