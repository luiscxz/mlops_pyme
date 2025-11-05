"""
Autor: Luis A. García
Email: luisgarcia.oq95@gmail.com
Creado: 5 de noviembre del 2025

Este código está diseñado entrenar un modelo de clasificación, 
sus funciones principales son:
    • Entrenar el modelo de clasificación con pipeline de sklearn
    • Evaluar el modelo entrenado
"""

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score


# Clase que se encarga de entrenar el modelo
class TrainModel:
    """
    Clase encargada de entrenar y evaluar modelos de ml
    """
    def __init__(self,train_file_path,test_file_path=None):
        """
        Función que inicializa la clase con la ruta el archivo y verifica
        que exista.

        Attributes
        ----------
        file_path: str
            Ruta del archivo csv

        Raise
        FileNotFoundError
            Si el archivo no existe.
        """
        self.train_file_path = Path(train_file_path)
        self.test_file_path = Path(test_file_path) if test_file_path else None
        if not self.train_file_path.exists():
            raise FileNotFoundError(f"No existe la ruta {self.train_file_path}")

        if self.test_file_path and not self.test_file_path.exists():
            raise FileNotFoundError(f"No existe la ruta {self.test_file_path}")
        
        self.data = None
        self.test_data =None
        self.pipeline = None
    
    def load_dataset(self):
        """
        Lee el archivo csv de entrenamiento indicado en el self.train_file_path y lo carga como un
        dataframe de pandas.

        return
        ------
        pd.DataFrame
            DataFrame que contiene los datos leidos del archivo csv.

        Raises
        ------
        pandas.errors.EmptyDataError
            Si el archivo está vacío

        pandas.errors.ParserError
            Si el archivo CSV tiene errores de formato o no puede ser parseado correctamente.
        """

        self.data = pd.read_csv(self.train_file_path)
        return self.data
    
    def load_test_dataset(self):

        """
        Carga el dataset de test (si se proporcionó).

        return
        ------
        pd.DataFrame
            DataFrame que contiene los datos leidos del archivo csv.

        Raises
        ------
        ValueError
            Si no se proporciona la ruta del dataset de prueba
        
        pandas.errors.EmptyDataError
            Si el archivo está vacío

        pandas.errors.ParserError
            Si el archivo CSV tiene errores de formato o no puede ser parseado correctamente.
        """

        if self.test_file_path is None:
            raise ValueError("No se proporcionó la ruta del dataset de test.")
        self.test_data = pd.read_csv(self.test_file_path)

        return self.test_data
    
    def train_pipeline(self, target ='default_12m',parameters = None):
        """
        Entrena un pipeline con RobustScaler y DecisionTreeClassifier

        Attributes
        ----------
        target: str
            Nombre de la columna objetivo, por decfecto es 'defaul_12m'

        parameters: dict
            Diccionario con los mejores parámetros para entrenar el modelo

        Raise
        -----
        ValueError
            Si la variable target no existe en el dataset

        Return
        ------
        sklearn.pipeline
            pipeline de skalearn entrenada

        """
        if target not in self.data.columns:
            raise ValueError(f"La variable objetivo '{target}' no se encuentra en los datos.")
        
        independientes = self.data.drop(columns=[target],axis=1)
        objetivo = self.data[target]

        parameters = parameters or {}
        pipeline = Pipeline([ 
            ('scaler',RobustScaler()), 
            ('classifier',DecisionTreeClassifier(**parameters)) ])

        pipeline.fit(independientes, objetivo)
        self.pipeline = pipeline
    
        return pipeline
    
    def test_pipeline(self,target = 'default_12m',pos_label=1,print_metrics=True):
        """
        Se encarga de evaluar el modelo entrenado

        Attributes
        ----------
        target: str
            Nombre de la columna objetivo, por decfecto es 'defaul_12m'

        pos_label: int
            Clase que se considera positiva, por defecto es 1

        print_metrics: Booleano
            Indica si desea imprimir métricas de rendimiento
        
        Raises
        ------
        ValueError
            Si los datos de test no están cargados, si el pipeline no está entrenado y si 
            la variable objetivo no está en los datos de test.

        Return
        ------
        dict
            Diccionario con las métricas de evaluación del modelo

        """
        if self.pipeline is None:
            raise ValueError("El pipeline no está entrenado. Ejecuta primero 'train_pipeline()'.")
        if self.test_data is None:
            raise ValueError("Los datos de test no están cargados. Ejecuta 'load_test_dataset()'.")
        if target not in self.test_data.columns:
            raise ValueError(f"La variable objetivo '{target}' no se encuentra en los datos de test.")
        
        X_test = self.test_data.drop(columns=[target])
        y_test = self.test_data[target]
        
        y_pred = self.pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred, pos_label=pos_label)
        precision = precision_score(y_test, y_pred, pos_label=pos_label)
        recall = recall_score(y_test, y_pred, pos_label=pos_label)

        metrics = {
            'f1': f1,
            'precision': precision,
            'recall': recall
            }
        
        if print_metrics:
            print("=== MÉTRICAS DEL MODELO ===")
            print(f"F1-Score: {f1:.4f}")
            print(f"Precisión: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
        return metrics