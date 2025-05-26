import random
import numpy as np
from fastdtw import fastdtw
from joblib import Parallel, delayed


# =============================================
# Inicialización de la Población (con múltiples ventanas)
# =============================================
def initialize_population(population_size, window_length, data_length, data, n_windows=3):
    population = []
    for _ in range(population_size):
        individual = []
        for _ in range(n_windows):
            start_idx = random.randint(0, data_length - window_length)
            end_idx = start_idx + window_length
            individual.append((start_idx, end_idx))
        population.append(individual)
    return population


# =============================================
# Evaluación de Fitness (múltiples ventanas)
# =============================================
def fitness_function(individual, data):
    total_length = len(data)
    coverage_mask = np.zeros(total_length, dtype=bool)
    match_count = 0
    min_pattern_length = 5

    for start, end in individual:
        pattern = data[start:end]

        # Ignorar patrones triviales
        if len(pattern) < min_pattern_length or np.std(pattern) < 0.01:
            continue

        # Buscar repeticiones del patrón
        matches = find_similar_windows(pattern, data, threshold=0.85, shift_tolerance=3)

        # Marcar en el "coverage mask" las posiciones cubiertas
        for m_start, m_end in matches:
            coverage_mask[m_start:m_end] = True
        match_count += 1 if matches else 0

    coverage_score = np.sum(coverage_mask) / total_length  # Fracción del tiempo cubierto

    # Penalizar el número de patrones (más patrones = peor)
    if match_count == 0:
        return -np.inf  # Sin repeticiones, solución inútil

    penalty = match_count  # Puedes probar con log(match_count + 1) si quieres suavizar

    # Fitness: cobertura menos penalización por número de patrones
    return coverage_score - 0.01 * penalty


# =============================================
# Selección por Torneo
# =============================================
def tournament_selection(population, fitness_scores, tournament_size=3):
    tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
    winner = max(tournament, key=lambda x: x[1])
    return winner[0]


# =============================================
# Cruce de Individuos con múltiples ventanas
# =============================================
def single_point_crossover(parent1, parent2):
    if len(parent1) < 2:  # No se puede hacer crossover si solo hay una ventana
        return parent1[:], parent2[:]
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


# =============================================
# Mutación de Individuos
# =============================================
def mutation(individual, data_length, max_shift=10):
    mutated = []
    for start, end in individual:
        length = end - start
        shift = random.randint(-max_shift, max_shift)
        new_start = max(0, min(data_length - length, start + shift))
        mutated.append((new_start, new_start + length))
    return mutated


# =============================================
# Algoritmo Evolutivo Principal
# =============================================
def run_evolutionary_algorithm(data, generations=100, population_size=20, window_length=50, n_windows=3):
    population = initialize_population(population_size, window_length, len(data), data, n_windows)

    for generation in range(generations):
        fitness_scores = Parallel(n_jobs=-1)(
            delayed(fitness_function)(ind, data) for ind in population
        )

        new_population = []
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            child1, child2 = single_point_crossover(parent1, parent2)
            child1 = mutation(child1, len(data))
            child2 = mutation(child2, len(data))

            new_population.extend([child1, child2])

        population = new_population[:population_size]

    fitness_scores = Parallel(n_jobs=-1)(
        delayed(fitness_function)(ind, data) for ind in population
    )
    best_individual = max(zip(population, fitness_scores), key=lambda x: x[1])
    return best_individual[0]  # Retorna el mejor conjunto de ventanas


# =============================================
# Correlación Segura
# =============================================
def safe_corrcoef(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


# =============================================
# Buscar ventanas similares a patrones
# =============================================
def find_similar_windows(pattern, data, threshold=0.8, shift_tolerance=3):
    matches = []
    win_len = len(pattern)

    if np.std(pattern) < 0.01:
        return matches

    for i in range(len(data) - win_len + 1):
        for shift in range(-shift_tolerance, shift_tolerance + 1):
            shifted_start = max(0, min(len(data) - win_len, i + shift))
            window = data[shifted_start:shifted_start + win_len]

            if np.std(window) < 0.01:
                continue

            corr = safe_corrcoef(pattern, window)
            if corr >= threshold and not np.isnan(corr):
                matches.append((shifted_start, shifted_start + win_len))

    return matches


# =============================================
# Ejecutar para múltiples tamaños de ventana
# =============================================
def run_multiple_window_sizes(data, window_sizes, generations=100, population_size=20, n_windows=3):
    all_patterns = []
    for window_length in window_sizes:
        if window_length > len(data):
            print(f"Aviso: Tamaño de ventana {window_length} es mayor que la longitud de los datos ({len(data)}). Se omite.")
            continue

        best_individual = run_evolutionary_algorithm(
            data, generations, population_size, window_length, n_windows
        )

        for (start, end) in best_individual:
            pattern = data[start:end]
            similar_windows = find_similar_windows(pattern, data)
            all_patterns.extend(similar_windows)
# Detectar patrones planos
    flat_patterns = detect_flat_patterns(data, window_length=80, min_std=0.005)
    print("Patrones planos detectados:")
    for i, (start, end) in enumerate(flat_patterns):
        print(f"  {i+1}. Índices {start}-{end} (longitud {end - start})")
# Mezclar todos los patrones
    all_patterns.extend(flat_patterns)
    
    return all_patterns


# =============================================
# Detección de Patrones Planos
# =============================================
def detect_flat_patterns(data, window_length=80, min_std=0.005):
    flat_patterns = []
    i = 0
    while i < len(data) - window_length:
        window = data[i:i + window_length]
        if np.std(window) < min_std:
            # Expandir la región plana mientras siga cumpliendo la condición
            start = i
            while i < len(data) - window_length and np.std(data[i:i + window_length]) < min_std:
                i += 1
            end = i + window_length
            flat_patterns.append((start, min(end, len(data))))
        else:
            i += 1
    return flat_patterns
