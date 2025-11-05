"""
Autor: Luis A. García
Email: luisgarcia.oq95@gmail.com
Creado: 5 de noviembre del 2025

Este código está diseñado establecer hacer uso del paquete utils y 
sus funciones principales son:
    • Entrenar el modelo 
    • Conetarse a minio
    • Conectarse a Mlflow
    • Cargar los modelos y registrarlos en producción.
"""
from utils import TrainModel, MLflowModelRegister, MinioMlflowBucketCreator
from pathlib import Path
import os
from mlflow.client import MlflowClient


"""
Sección correspondiente a entrenar y evaluar el modelo
"""
project_root = next(p for p in Path.cwd().parents if (p / 'data').exists())
file_path = lambda file: os.path.join(project_root,'data/processed',file)
train_model = TrainModel(
    train_file_path=file_path('covalto_sme_credit_train.csv'),
    test_file_path=file_path('covalto_sme_credit_test.csv')
)
train_model.load_dataset()
train_model.load_test_dataset()
modelo = train_model.train_pipeline(
    target = 'default_12m',
    parameters = {
        'criterion': 'gini', 
        'max_depth': 2, 
        'min_samples_split': 8, 
        'min_samples_leaf': 19, 
        'max_features': None, 
        'class_weight': None}
)
metrics = train_model.test_pipeline()



"""
Sección correspondiente a establecer conexión con minio y crear bucket si no
existe.
"""
project_secrets = next(p for p in Path.cwd().parents if (p / 'secrets').exists())
credential_path = lambda file: os.path.join(project_root,'secrets',file)
minio = MinioMlflowBucketCreator(
    credential_path = credential_path('credentials_minio.json')
)
minio.load_minio_credentials()
minio.conection_minio()
minio.create_bucket(bucket_name='mlflow')


"""
Sección correspondiente a establecer conexión con mlflow
crear experimento y registrar los modelos.
"""
tracking_uri_path = lambda file: os.path.join(project_root,'secrets',file)

mlflow_register = MLflowModelRegister(
    tracking_uri_path = tracking_uri_path ('tracking_uri_mlflow.json')
)


mlflow_register = mlflow_register.load_tracking_uri_mlflow()
mlflow_register = mlflow_register.create_mlflow_experiment(
    experiment_name = "riesgo_crediticio"
)

# Variables de entorno para minio
creds = minio.credentials  # ya cargadas con load_minio_credentials()

os.environ["MLFLOW_S3_ENDPOINT_URL"] = creds['endpoint_url']
os.environ["AWS_ACCESS_KEY_ID"] = creds['aws_access_key_id']
os.environ["AWS_SECRET_ACCESS_KEY"] = creds['aws_secret_access_key']

mlflow_register = mlflow_register.log_pipeline(
    pipeline = modelo,
    metrics = metrics,
    params={
    'criterion': 'gini', 
    'max_depth': 2, 
    'min_samples_split': 8, 
    'min_samples_leaf': 19, 
    'max_features': None, 
    'class_weight': None},
    model_name = "DecisionTree_CreditRiskModel",
    tags={"author": "Luis Garcia", "use_case": "credit-risk"},
    register_in_registry=True)

# Pasar a producción
client = MlflowClient()
client.set_registered_model_alias(
    name="DecisionTree_CreditRiskModel",
    alias="Production",
    version=2
)