import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Leer series temporales desde un archivo CSV (con o sin encabezado).
    :param filepath: Ruta al archivo CSV.
    :return: Serie temporal como un array numpy.
    """
    try:
        data = pd.read_csv(filepath)

        # Eliminar columna índice si no tiene nombre
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        # Buscar columnas numéricas válidas
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            raise ValueError("No se encontró una columna numérica válida.")

        values = data[numeric_cols[0]].astype(float).values

        # Eliminar posibles NaN
        values = values[~np.isnan(values)]

        return values


    except Exception as e:
        print(f"Error al cargar el archivo {filepath}: {e}")
        return None



def normalize_data(data):
    """
    Normalizar valores entre [0, 1].

    :param data: Array de valores.
    :return: Array normalizado entre 0 y 1.
    """
    if len(data) == 0:
        raise ValueError("Los datos proporcionados están vacíos.")


    min_val = np.min(data)
    max_val = np.max(data)

    return (data - min_val) / (max_val - min_val)
