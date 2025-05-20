import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Leer series temporales desde un archivo.
    
    :param filepath: Ruta al archivo CSV.
    :return: Serie temporal como un array numpy.
    """
    try:
        # Leer el archivo CSV
        data = pd.read_csv(filepath, delimiter=None, engine="python")
        
        # Identificar posibles columnas de valores
        possible_columns = ['value', 'Value', 'V1']
        selected_column = None
        
        for col in data.columns:
            if col.strip() in possible_columns:
                selected_column = col
                break
        
        if selected_column is None:
            raise ValueError("No se encontró una columna válida para los valores.")
        
        # Extraer los valores
        values = data[selected_column].values
        
        return np.array(values)
    
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

