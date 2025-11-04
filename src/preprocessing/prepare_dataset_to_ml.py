"""
Autor: Luis A. García
Email: luisgarcia.oq95@gmail.com
Creado: 22 de octubre del 2025

Este código está diseñado para preparar los datos en entrenamiento y prueba del proyecto 
MLOPS_PYME, sus funciones principales son:
    • Cargar el dataset preprocesado 
    • Feature engginering
    • Seleccionar las variables que consideramos importantes
    • Codificar variables ordinales
    • Balancear clases sacando una muestra representativa
    • Dividir los datos en entrenamiento y prueba
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

class MlDataPreprocessor:
    """
    Clase encargada de preparar y dividir los datos en entranimiento y test. 

    Attributes
    ----------
    file_path: str or Path
        Ruta del archivo csv a limpiar
    """
    def __init__(self,file_path):
        """
        Función que inicializa la clase con la ruta del archivo y verifica que
        el archivo exista.

        Attributes
        ----------
        file_path: str
            Ruta del archivo csv.

        Raises
        ------
        FileNotFoundError
            Si el archivo no existe

        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"No existe la ruta: {self.file_path}")
        self.data = None
        self.data_train = None
        self.data_test = None

    def load_dataset(self):

        """
        Lee el archivo csv indicado en el self.file_path y lo carga como un
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

        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def create_features(self):
        """
        Crea las variables más imporantes que deben ser agregadas al dataset

        Raise
        -----
        ValueError
            Si el dataset no ha sido cargado
        Return
        ------
        pd.DataFrame
            Dataframe con las nuevas caracteísticas calculadas
        """
        if self.data is None:
            raise ValueError("El dataset no ha sido cargado. Ejecute load_dataset() primero.")
        
        self.data['ratio_deuda_ingresos'] = self.data['deuda_total_mxn']/self.data['ingresos_anuales_mxn']
        self.data['carga_total_ingresos'] = (self.data['deuda_total_mxn'] + self.data['monto_solicitado_mxn'])/self.data['ingresos_anuales_mxn']
        return self.data
    
    def select_features(self,list_features=[]):
        """
        Selecciona las variables consideradas importantes para entrenar el modelo

        Attributes
        ----------
        list_features: list
            Lista de nombres de columnas

        Return
        ------
        pd.Dataframe
            Dataframe compuesto solamente pos la columnas de la lista
        """
        if len(list_features)>1:
            self.data = self.data[list_features]
        return self.data
    
    def encode_ordinal(self,name_col_ordinal ='calificacion_buro',dict_map ={}):
        """
        Mapea una variable ordinal en base al diccionario ingresado

        Attributes
        ----------
        name_col_ordinal: str
            Nombre de la columna ordinal a mapear
            Por defecto es 'calificacion_buro'
        dict_map: dict
            Diccionario con las respectivar ordenes de mapeo.

        Raise
        -----
        ValueError
            Si el diccionario está vacio

        Return
        pd.DataFrame
            Dataframe con la columna ordinal mapeada.
        """
        if len(dict_map)>1:
            self.data[name_col_ordinal] = self.data[name_col_ordinal].map(dict_map)
        else:
            raise ValueError("El diccionario de codificación ordinal está vacío.")
        return self.data
    
    def extract_sample(self, target='default_12m'):
        """
        Extrae una muestra del DataFrame garantizando que las clases 0 y 1 
        estén balanceadas, mediante estratificación en una columna de interés. 

        Attributes
        ----------
        target: str
            Nombre de la columna que contiene las clases desbalanceadas
            Por defecto es 'default_12m'

        Raise
        -----
        KeyError
            Si la columna ingresada no existe en el DataFrame

        Return
        ------
        pd.DataFrame
            Dataframe con las clases balanceadas
        """
        if target in self.data.columns:
            class_false = self.data[self.data[target]==0]
            class_true = self.data[self.data[target]==1]

            class_false_sub, _ = train_test_split(
                class_false, # clase a submuestrear
                train_size = len(class_true),
                stratify = class_false['sector_industrial'], # estractifica por sector industrial
                random_state = 43,
                shuffle = True 
            )

            data_sample = pd.concat([class_false_sub,class_true],axis=0)
            self.data = data_sample.copy()
        else:
            raise KeyError(f"La columna {target} no existe en el DataFrame.")
        return self.data
    
    def split_data(self,target='default_12m',test_size=0.05):
        """
        Separa los datos en entrenamiento y  test

        Attributes
        ----------
        target: str
            Nombre de la columna objetivo, por defecto es 'default_12m'.capitalize

        test_size: float
            Tamaño en porcentaje del dataset de prueba
            por defecto es 5%
        
        Raise
        -----
        KeyError
            Si la columna ingresada no existe en el dataframe

        Return
        ------
        pd.DataFrame
            DataFrames para entrenamiento y test
        """
        if target in self.data.columns:

            independientes = self.data.drop(columns=[target],axis=1)
            objetivo = self.data[target]

            X_train, X_test, Y_train, Y_test = train_test_split(
            independientes,
            objetivo,
            test_size=test_size,
            random_state=42,
            shuffle=True,
            stratify=self.data[target]
            )

            data_train = pd.concat([X_train,Y_train],axis=1)
            data_test = pd.concat([X_test,Y_test],axis=1)

            self.data_train = data_train.copy()
            self.data_test = data_test.copy()
        else:
            raise KeyError(f"La columna {target} no existe en el DataFrame.")
        return self.data_train, self.data_test 

# Código en acción
if __name__=="__main__":
    # 1. Obteniene la ruta general donde están los datos
    project_root = next(p for p in Path.cwd().parents if (p / 'data').exists())
    file_path = lambda file : os.path.join(project_root,'data/processed',file)

    # 2. Carga los datos
    prepare_data = MlDataPreprocessor(file_path('covalto_sme_credit_data_clean.csv'))
    data = prepare_data.load_dataset()

    # 3. Crea variables imporantes
    data = prepare_data.create_features()

    # 4. Codifica las variables ordinales
    data = prepare_data.encode_ordinal(
        name_col_ordinal ='calificacion_buro',
        dict_map={
            np.nan:0,
            'A':1,
            'B':2,
            'C':3,
            'D':4
        }
        )
    # 5. Extrae una muestra con clases balanceadas
    data = prepare_data.extract_sample(
        target='default_12m')
    
    # 6. Selecciona las características importantes econtradas durante fase de exploración
    data = prepare_data.select_features(list_features=[
        'historial_pagos_atrasados',
        'calificacion_buro',
        'monto_solicitado_mxn',
        'ratio_deuda_ingresos',
        'carga_total_ingresos',
        'default_12m'
    ])

    # 7. Divide los datos en entrenamiento y prueba
    data_train, data_test = prepare_data.split_data(
        target='default_12m',
        test_size=0.03
    )

    # 8. Guarda los datos en archivos scv
    dir_save_data = project_root/'data/processed'
    data_train.to_csv(dir_save_data/ 'covalto_sme_credit_train.csv', index=False)
    data_test.to_csv(dir_save_data/ 'covalto_sme_credit_test.csv', index=False)