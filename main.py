import argparse
import numpy as np
from preprocessing import load_data, normalize_data
from algorithm import run_evolutionary_algorithm, find_similar_windows
from visualization import plot_series, plot_detected_patterns

def main():
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Detección de patrones en series temporales mediante algoritmos evolutivos.")
    parser.add_argument("csv_file", type=str, help="Ruta al archivo CSV con la serie temporal.")
    parser.add_argument("--generations", type=int, default=100, help="Número de generaciones del algoritmo evolutivo.")
    parser.add_argument("--population_size", type=int, default=20, help="Tamaño de la población.")
    parser.add_argument("--window_length", type=int, default=50, help="Longitud de las ventanas deslizantes.")
    args = parser.parse_args()

    # Paso 1: Cargar y normalizar datos
    print("Cargando datos...")
    data = load_data(args.csv_file)
    if data is None:
        print("Error al cargar los datos. Asegúrate de que el archivo CSV es válido.")
        return

    print("Normalizando datos...")
    normalized_data = normalize_data(data)

    # Paso 2: Ejecutar el algoritmo evolutivo
    print("Ejecutando el algoritmo evolutivo...")
    best_window = run_evolutionary_algorithm(
        data=normalized_data,
        generations=args.generations,
        population_size=args.population_size,
        window_length=args.window_length
    )

    # Paso 3: Visualización de resultados
    print("Visualizando resultados...")
    start_idx, end_idx = best_window
    pattern = normalized_data[start_idx:end_idx]
    similar_windows = find_similar_windows(pattern, normalized_data, threshold=0.8)
    detected_patterns = [(start_idx, end_idx)]

    # Graficar la serie original
    plot_series(normalized_data)

    # Mostrar resultados por consola
    print(f"Patrón base: índice {start_idx}-{end_idx}")
    print("Patrones similares encontrados:")
    for i, (start, end) in enumerate(similar_windows):
        print(f"  {i+1}. Índices {start}-{end} (longitud {end-start})")

    # Visualización de todos los patrones encontrados
    plot_detected_patterns(normalized_data, similar_windows)

    print(f"Mejor ventana detectada: Inicio = {start_idx}, Fin = {end_idx}")

if __name__ == "__main__":
    main()
