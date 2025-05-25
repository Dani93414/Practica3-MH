import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Leer series temporales desde un archivo CSV.
    Se espera al menos una columna numérica.

    :param filepath: Ruta al archivo CSV.
    :return: Serie temporal como array numpy o None si falla.
    """
    try:
        data = pd.read_csv(filepath)

        # Eliminar columna índice automática (por ejemplo: 'Unnamed: 0')
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        # Buscar la primera columna numérica válida
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            raise ValueError("No se encontró ninguna columna numérica válida.")

        values = data[numeric_cols[0]].astype(float).values
        values = values[~np.isnan(values)]  # Eliminar NaNs

        if values.size == 0:
            raise ValueError("Todos los valores son NaN o la columna está vacía.")

        return values

    except Exception as e:
        print(f"Error al cargar el archivo {filepath}: {e}")
        return None


def normalize_data(data):
    """
    Normalizar valores entre 0 y 1 usando min-max scaling.

    :param data: Array de valores.
    :return: Array normalizado.
    """
    if len(data) == 0:
        raise ValueError("Los datos proporcionados están vacíos.")

    min_val = np.min(data)
    max_val = np.max(data)

    if max_val == min_val:
        # Evitar división por cero si todos los valores son iguales
        return np.zeros_like(data)

    return (data - min_val) / (max_val - min_val)
