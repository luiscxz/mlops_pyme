"""
Autor: Luis A. García
Email: luisgarcia.oq95@gmail.com
Creado: 22 de octubre del 2025

Este código está diseñado para limpiar los datos del proyecto 
MLOPS_PYME, sus funciones principales son:
    • Limpiar valores atípicos
    • Estandarizar la varibale categória "Retail"
"""
import pandas as pd
import numpy as np
from statsmodels.stats.stattools import medcouple
from pathlib import Path
import os

class CovaltoCsvDataCleaner:
    """
    Clase encargada de limpiar los datos provenientes de un archivo csv de la 
    base de datos de covalto.

    Attributes
    ----------
    file_path: str or Path
        Ruta del archivo csv a limpiar
    """

    def __init__(self,file_path):

        """
        función que inicializa la clase con la ruta del archivo y verifica que
        el archivo exista.
        
        Attributes
        ----------
        file_path: Ruta del archivo csv a limpiar

        Raises
        ------
        FileNotFoundError
            Si el archivo específicado no existe

        """

        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"El archivo no existe: {self.file_path}")
        
        self.data = None                    
        

    
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
    
    
    def remove_outliers(self, col_interes='ingresos_anuales_mxn',max_retries=3):

        """
        Elimina los valores atípicos de una columna numérica usando el estadístico MedCouple (MC).

        Este método ajusta los límites inferior y superior de detección de outliers 
        en función de la asimetría de la distribución, medida con el estadístico MedCouple.

        Parameters
        ----------
        covalto_dataset_raw: pd.DataFrame
            DataFrame que contiene los datos a procesar.

        col_interes: str 
            Nombre de la columna sobre la cual se eliminaran los valores atípicos.
            Por defecto es 'ingresos_anuales_mxn'.

        max_retries : int
            Número máximo de intentos para ingresar una columna válida.
            Por defecto es 3.
        
        Return
        ------
        pandas.DataFrame
            Copia del DataFrame original sin los valores atípicos en la columna indicada.

        Raises
        ------
        ValueError
            Si no se proporciona una columna válida después de varios intentos.
        
        Notes
        -----
            - Si el MedCouple (MC) es positivo, se ajusta más el límite inferior.
            - Si el MC es negativo, se ajusta más el límite superior.
            - Si el MC es 0, se aplica la regla estándar del rango intercuartílico (IQR).

        """

        if self.data is None:
            raise ValueError("El dataset no ha sido cargado. Ejecute load_dataset() primero.")
        
        attempt = 0
        while attempt<max_retries:
            try:
                # Intenta acceder a la columna
                data = self.data.copy()
                _=data[col_interes]

                # Calcular estadísticas
                resumen = data[col_interes].describe()
                Q1, Q2, Q3 = resumen.iloc[4], resumen.iloc[5], resumen.iloc[6]
                RI = Q3 - Q1
                MC = medcouple(data[col_interes].to_numpy())

                # Determinar límites
                if MC > 0:
                    lim_inf_MC = Q1 - 1.5 * np.exp(-3.5 * MC) * RI
                    lim_sup_MC = Q3 + 1.5 * np.exp(4 * MC) * RI
                elif MC < 0:
                    lim_inf_MC = Q1 - 1.5 * np.exp(-4 * MC) * RI
                    lim_sup_MC = Q3 + 1.5 * np.exp(3.5 * MC) * RI
                else:
                    lim_inf_MC = Q1 - 1.5 * RI
                    lim_sup_MC = Q3 + 1.5 * RI

                data = data[
                    (data[col_interes] >= lim_inf_MC) &
                    (data[col_interes] <= lim_sup_MC) |
                    (data[col_interes].isna())
                ]
                self.data = data.copy()
                return self.data
            
            except KeyError:
                print(f"La columna '{col_interes}' no existe en el DataFrame.")
                attempt += 1
                if attempt < max_retries:
                    col_interes = input("Por favor, ingrese el nombre correcto de la columna: ")
                else:
                    raise ValueError("No se proporcionó una columna válida después de varios intentos.")

    def standardize_sector_column(self,col_name='sector_industrial',max_retries=3):
        """
        Estandariza los nombres del sector industrial en el DataFrame.
        
        Reemplaza valores inconsistentes o en minúsculas por versiones normalizadas.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame a procesar.
        
        col_name: str
            Nobre de la columna sobre la cual se estarizarán las categorías
        
        max_retries: int
            Número máximo de intentos para ingregar una columna válida
            Por defecto es 3.
        Returns
        -------
        pandas.DataFrame
            Copia del DataFrame con los nombres del sector estandarizados.
        """
        
        attempt = 0
        while attempt<max_retries:
            try:
                self.data[col_name] = self.data[col_name].replace({
                 'retail': 'Retail'   
                })
                return self.data
            except KeyError:
                print(f"la columna {col_name} no existe en el DataFrame.")
                attempt +=1
                if attempt<max_retries:
                    col_name = input('Ingrese el nombre correcto de la columna: ')
                else:
                    ValueError("No se proporcionó una columna válida después de varios intentos")


# Código en acción
if __name__ == "__main__":
    # 1. Obteniene la ruta general donde están los datos
    project_root = next(p for p in Path.cwd().parents if (p / 'data').exists()) 
    file_path = lambda file : os.path.join(project_root,'data/raw',file)

    # 2. Carga los datos
    cleaner = CovaltoCsvDataCleaner(file_path('covalto_sme_credit_data.csv'))   # Crear instancia
    data = cleaner.load_dataset()    

    # 3. Limpia Outliers
    data = cleaner.remove_outliers(
        col_interes ='ingresos_anuales_mxn',
        max_retries=3
    )

    # 4. Reemplaza retail por Retail en la columna 'sector_industrial'
    data = cleaner.standardize_sector_column(
        col_name = 'sector_industrial',
        max_retries = 3)

    # 5. Guarda el archivo preprocesado

    dir_save_data = project_root/'data/processed'
    data.to_csv(dir_save_data/ 'covalto_sme_credit_data_clean.csv', index=False)