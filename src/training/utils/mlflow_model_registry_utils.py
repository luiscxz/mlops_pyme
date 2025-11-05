"""
Autor: Luis A. García
Email: luisgarcia.oq95@gmail.com
Creado: 5 de noviembre del 2025

Este código está diseñado establecer cargar los modelos en mlflow, 
sus funciones principales son:
    • Establecer conexión con mlflow
    • Cargar el modelo a mlflow
    • Guardar el modelo en el registro de mlflow
    • Promocionar el modelo a producción
"""
import mlflow
from datetime import datetime
from pathlib import Path
import json

class MLflowModelRegister:
    """
    Sube el modelo al servidor mlflow y permite asignar a producción.
    """
    def __init__(self,tracking_uri_path):
        """
        Inicializa la clase con con la ruta del archivo json que contiene el uri de mlflow

        Attributes
        ----------
        tracking_uri_path: str
            Ruta del archivo json que contiene la uri de mlflow

        Raises
        ------
        FileNotFoundError
            Si no se encuentra el archivo 
        """
        self.tracking_uri_path = Path(tracking_uri_path)
        if not self.tracking_uri_path.exists():
            raise FileNotFoundError(f"No existe la ruta {self.tracking_uri_path}")
        self.tracking_uri = None
    
    def load_tracking_uri_mlflow(self):
        """
        Carga el archivo json que contiene la uri de mlflow

        Return
        ------
        self
        """
        with open(self.tracking_uri_path, 'r') as file:
            self.tracking_uri = json.load(file)
        return self
        
    def create_mlflow_experiment(self, experiment_name = "Experimento_1"):
        """
        Crea experimentos en mlflow

        Attributes
        ----------
        experiment_name: str
            Nombre del experimento a crear, por defecto es "Experimento_1"
        
        Return
        ------
        self
        """
        tracking_uri = self.tracking_uri['tracking_uri']
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        return self

    def log_pipeline(
        self,
        pipeline,
        metrics: dict,
        params: dict,
        model_name: str = "sklearn-pipeline",
        framework: str = "scikit-learn",
        type: str = "classification",
        tags: dict = None,
        register_in_registry: bool = True
    ):
        """
        Registra los modelos en mlflow

        Attributes
        ----------
        pipeline: model
            Modelo entrenado y validado
        metric: dict
            Diccionario con las métricas de evaluación del modelo
        params: dict
            Diccionario con los parámetros de entrenamiento del modelo
        model_nade: str
            Nombre del modelo, por defecto es "sklearn-pipeline"
        framework: str
            Librería de entrenamiento del modelo, por defecto es "scikit-learn"
        type: str
            Tipo de modelo, por defecto es "classification"
        
        tags: dict
            opcional, por defecto es None
        """
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params(params)

            if metrics:
                mlflow.log_metrics(metrics)

            # para registrar directamente en el Model Registry
            if register_in_registry:
                mlflow.sklearn.log_model(
                    pipeline,
                    name="model",
                    registered_model_name=model_name  
                )
            else:
                mlflow.sklearn.log_model(
                    pipeline,
                    name=model_name,
                )

            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("framework", framework)
            mlflow.set_tag("type", type)

            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(k, v)
        return self 