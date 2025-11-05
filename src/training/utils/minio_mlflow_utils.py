"""
Autor: Luis A. García
Email: luisgarcia.oq95@gmail.com
Creado: 5 de noviembre del 2025

Este código está diseñado establecer conexión con minio, 
sus funciones principales son:
    • Establecer conexión minio
    • Crear bucket en minio
"""
import json
import boto3
from pathlib import Path

class MinioMlflowBucketCreator:
    """
    Se conecta a minio y puede crear buckest
    """
    def __init__(self,credential_path):
        """
        Función que inicializa la clase con la ruta del archivo y verifica
        que exista.

        Attributes
        ----------
        credential_path: str
            Ruta del las credenciales de minio

        Raise
        FileNotFoundError
            Si el archivo no existe.
        
        """
        self.credential_path = Path(credential_path)
        if not self.credential_path.exists():
            raise FileNotFoundError(f"No existe la ruta {self.credential_path}")
        self.credentials = None
        self.client = None

    def load_minio_credentials(self):
        """
        Carga el archivo json que contiene las credenciales.

        return
        ------
        json
            json con credenciales de minio.

        """
        with open(self.credential_path, 'r') as file:
            self.credentials = json.load(file)
        return self.credentials
    
    def conection_minio(self):
        """
        Configura el cliente de conexión a minio

        return
        ------
            Retona el cliente configurado para conectarse a minio
        """
        s3 = boto3.client(
            "s3",
            endpoint_url = self.credentials['endpoint_url'],
            aws_access_key_id = self.credentials['aws_access_key_id'],
            aws_secret_access_key = self.credentials['aws_secret_access_key']
        )
        self.client = s3
        return self.client
    
    def create_bucket(self, bucket_name ='mlflow'):

        """
        Crea buckets en minio

        Attributes
        ----------
        bucket_name: str
            Nombre del bucket a crear, por decfecto es 'mlflow'
        
        Raise
        -----
        ConectionError
            Si no ha establecido conexión con minio
        
        RuntimeError
            Si no se puede crear el bucket
        """
        if self.client is None:
            raise ConnectionError("Primero debes conectarte a MinIO con conection_minio().")
        try:
            existing_buckets = [b['Name'] for b in self.client.list_buckets().get('Buckets', [])]
            if bucket_name not in existing_buckets:
                self.client.create_bucket(Bucket=bucket_name)
            else:
                pass
        except:
            raise RuntimeError(f"Error al crear el bucket {bucket_name}")
