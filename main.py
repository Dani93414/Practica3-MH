import argparse
import numpy as np
from preprocessing import load_data, normalize_data
from algorithm import run_evolutionary_algorithm, find_similar_windows, run_multiple_window_sizes
from visualization import plot_series, plot_detected_patterns

def main():
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Detección de patrones en series temporales mediante algoritmos evolutivos.")
    parser.add_argument("csv_file", type=str, help="Ruta al archivo CSV con la serie temporal.")
    parser.add_argument("--generations", type=int, default=100, help="Número de generaciones del algoritmo evolutivo.")
    parser.add_argument("--population_size", type=int, default=20, help="Tamaño de la población.")
    parser.add_argument("--window_length", type=int, default=50, help="Longitud de la ventana si se usa un único tamaño.")
    parser.add_argument("--n_windows", type=int, default=3, help="Número de ventanas por individuo.")
    parser.add_argument("--window_sizes", type=str, default=None, help="Lista de tamaños de ventana separados por comas (por ejemplo, '30,50,100').")
    args = parser.parse_args()

    # Paso 1: Cargar y normalizar datos
    print("Cargando datos...")
    data = load_data(args.csv_file)
    if data is None:
        print("Error al cargar los datos. Asegúrate de que el archivo CSV es válido.")
        return

    print("Normalizando datos...")
    normalized_data = normalize_data(data)

    # Paso 2: Selección del método
    if args.window_sizes:
        print("Ejecutando el algoritmo evolutivo con múltiples tamaños de ventana...")
        window_sizes = [int(size) for size in args.window_sizes.split(",")]
        all_patterns = run_multiple_window_sizes(
            data=normalized_data,
            window_sizes=window_sizes,
            generations=args.generations,
            population_size=args.population_size,
            n_windows=args.n_windows
        )

        print("Visualizando resultados...")
        plot_series(normalized_data)
        plot_detected_patterns(normalized_data, all_patterns)

        print("Patrones similares encontrados:")
        for i, (start, end) in enumerate(all_patterns):
            print(f"  {i+1}. Índices {start}-{end} (longitud {end - start})")

    else:
        print("Ejecutando el algoritmo evolutivo con un único tamaño de ventana...")
        best_windows = run_evolutionary_algorithm(
            data=normalized_data,
            generations=args.generations,
            population_size=args.population_size,
            window_length=args.window_length,
            n_windows=args.n_windows
        )

        all_matches = []
        for (start_idx, end_idx) in best_windows:
            pattern = normalized_data[start_idx:end_idx]
            matches = find_similar_windows(pattern, normalized_data, threshold=0.8)
            all_matches.extend(matches)

        print("Visualizando resultados...")
        plot_series(normalized_data)
        plot_detected_patterns(normalized_data, all_matches)

        print("Ventanas base encontradas:")
        for i, (start, end) in enumerate(best_windows):
            print(f"  {i+1}. Índices {start}-{end} (longitud {end - start})")

        print("Patrones similares encontrados:")
        for i, (start, end) in enumerate(all_matches):
            print(f"  {i+1}. Índices {start}-{end} (longitud {end - start})")

if __name__ == "__main__":
    main()
