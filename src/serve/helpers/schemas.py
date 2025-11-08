"""
Autor: Luis A. García
Email: luisgarcia.oq95@gmail.com
Creado: 6 de noviembre del 2025

Este código está diseñado recibir los datos de entrada y crear el DataFrame
para hacer predicciones con el modelo de clasificación
sus funciones principales son:
    • Validar que los tipos de datos recibidos sean los correctos 
    • Verificar que los datos de entrada tegan las mismas variables 
      usadas en entrenamiento.
    • Convertir los datos de entrada a DataFrame
"""

from pydantic import BaseModel, Field
from typing import Optional, Union
import pandas as pd
import numpy as np
from io import StringIO

class ClassificationRequest(BaseModel):
    """
    Modelo de datos para solicitudes de clasificación crediticia.

    Esta clase define la estructura esperada de la información de entrada
    para un modelo de clasificación, utilizando validación de tipos y
    descripciones semánticas a través de Pydantic. Cada campo representa
    una variable relevante para evaluar el riesgo crediticio del cliente.

    Attributes
    ----------
    historial_pagos_atrasados : Optional[int]
        Número de pagos atrasados del cliente.
    calificacion_buro : Optional[int]
        Puntaje del buró de crédito del cliente.
    monto_solicitado_mxn : Optional[float]
        Monto del préstamo solicitado en pesos mexicanos.
    ratio_deuda_ingresos : Optional[float]
        Relación entre la deuda total del cliente y sus ingresos.
    carga_total_ingresos : Optional[float]
        Porcentaje de carga total sobre los ingresos del cliente.

    Métodos
    -------
    from_input(data)
        Convierte diferentes tipos de entrada (DataFrame, diccionario, lista o CSV)
        en un DataFrame con las columnas requeridas por el modelo.
    """
    historial_pagos_atrasados: Optional[int] = Field(None, description="Número de pagos atrasados del cliente")
    calificacion_buro: Optional[int] = Field(None, description="Puntaje del buró de crédito")
    monto_solicitado_mxn: Optional[float] = Field(None, description="Monto del préstamo solicitado en pesos")
    ratio_deuda_ingresos: Optional[float] = Field(None, description="Relación entre deuda total e ingresos")
    carga_total_ingresos: Optional[float] = Field(None, description="Porcentaje de carga total sobre los ingresos")

    @classmethod
    def from_input(cls, data: Union[str, pd.DataFrame, dict, list, bytes]):
        """
        Convierte una entrada de datos en un DataFrame validado con las columnas del modelo.

        Este método acepta múltiples formatos de entrada —como DataFrame, diccionario,
        lista de registros o archivo CSV (en texto o bytes)— y los transforma en un
        `pandas.DataFrame` con las columnas definidas en la clase. Si faltan columnas
        requeridas o el formato no es soportado, se lanza una excepción descriptiva.

        Parameters
        ----------
        data : Union[str, pd.DataFrame, dict, list, bytes]
            Datos de entrada que representan uno o varios registros. Puede ser:
            - `pd.DataFrame`: datos ya tabulares.
            - `dict`: un solo registro o un diccionario con listas de valores.
            - `list`: lista de diccionarios (cada uno una fila).
            - `str` o `bytes`: contenido de un archivo CSV.

        Return
        -------
        pd.DataFrame
            DataFrame con las columnas requeridas por el modelo `ClassificationRequest`.
        
        Raises
        ------
        ValueError
            Si no se puede leer el archivo csv.
            Si faltan las columnas requeridas
        TypeError
            Si el tipo de datos no es soportado.
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()

        elif isinstance(data, dict):
            try:
                # Convierte cualquier dict con listas (incluso de distinto largo) a DataFrame
                df = pd.DataFrame.from_dict(data, orient="index").T
            except Exception:
                # Si no son listas (o falla), tratamos como un solo registro
                df = pd.DataFrame([data])

        elif isinstance(data, list):
            df = pd.DataFrame(data)

        elif isinstance(data, (str, bytes)):
            try:
                df = pd.read_csv(StringIO(data))
            except Exception as e:
                raise ValueError(f"No se pudo leer el CSV: {e}")
        else:
            raise TypeError("El tipo de entrada no es soportado. Debe ser DataFrame, dict, lista o CSV.")

        required_fields = list(cls.model_fields.keys())
        missing_columns = [f for f in required_fields if f not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Faltan columnas requeridas: {missing_columns}. "
                f"Columnas disponibles: {list(df.columns)}"
            )
        
        df = df[required_fields]
        type_map = {
            "historial_pagos_atrasados": "Int64",  # Entero nulo seguro
            "calificacion_buro": "Int64",
            "monto_solicitado_mxn": "float64",
            "ratio_deuda_ingresos": "float64",
            "carga_total_ingresos": "float64",
        }
        try:
            df = df.astype(type_map)
        except Exception as e:
            raise ValueError(f"No se pudieron convertir los tipos correctamente: {e}")
        return df.replace({np.nan:None})
