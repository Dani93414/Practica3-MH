import random
import numpy as np

# =============================================
# Inicialización de la Población
# =============================================
def initialize_population(pop_size, window_length, data_length):
    """
    Genera una población inicial de individuos representados como índices
    de fragmentos en las series temporales.
    
    :param pop_size: Número de individuos en la población
    :param window_length: Longitud de las ventanas en la serie temporal
    :param data_length: Longitud total de la serie temporal
    :return: Lista de individuos (ventanas representadas como índices)
    """
    population = []
    for _ in range(pop_size):
        start_idx = random.randint(0, data_length - window_length - 1)
        individual = (start_idx, start_idx + window_length)
        population.append(individual)
    return population

# =============================================
# Evaluación de Fitness
# =============================================
def fitness_function(individual, data):
    """
    Calcula la función de fitness para un individuo basado en la similitud
    de la ventana que representa con el resto de la serie temporal.
    
    :param individual: Tuple (start_idx, end_idx)
    :param data: Serie temporal (array numpy)
    :return: Valor de fitness (float)
    """
    start, end = individual
    window = data[start:end]
    
    if len(window) < 5:  # Correlación no válida
        return -np.inf

    similarities = []
    for i in range(0, len(data) - len(window)):
        comp_window = data[i:i + len(window)]
        if len(comp_window) == len(window):
            corr = np.corrcoef(window, comp_window)[0, 1]
            if not np.isnan(corr):
                similarities.append(corr)

    if similarities:
        return np.mean(similarities)
    else:
        return -np.inf

# =============================================
# Operador de Selección
# =============================================
def tournament_selection(population, fitness_scores, tournament_size=3):
    """
    Selección por torneo.
    
    :param population: Lista de individuos
    :param fitness_scores: Lista de valores de fitness para los individuos
    :param tournament_size: Tamaño del torneo
    :return: Individuo ganador del torneo
    """
    tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
    winner = max(tournament, key=lambda x: x[1])  # Seleccionar el mejor fitness
    return winner[0]

# =============================================
# Operador de Cruce
# =============================================
def single_point_crossover(parent1, parent2, min_window_length=30):
    """
    Cruce de un punto entre dos padres.
    
    :param parent1: Tuple (start_idx, end_idx)
    :param parent2: Tuple (start_idx, end_idx)
    :return: Dos hijos resultantes del cruce
    """
    p1_start, p1_end = parent1
    p2_start, p2_end = parent2

    len1 = p1_end - p1_start
    len2 = p2_end - p2_start
    min_len = min(len1, len2)

    if min_len <= min_window_length:
        return parent1, parent2

    crossover_point = random.randint(min_window_length, min_len - 1)

    child1 = (p1_start, p1_start + crossover_point)
    child2 = (p2_start, p2_start + crossover_point)

    return child1, child2

# =============================================
# Operador de Mutación
# =============================================
def mutation(individual, data_length, max_shift=5):
    """
    Mutación por cambio de índice en un individuo.
    
    :param individual: Tuple (start_idx, end_idx)
    :param data_length: Longitud de la serie temporal
    :param max_shift: Máxima cantidad de cambio permitida
    :return: Individuo mutado
    """
    start, end = individual
    length = end - start
    shift = random.randint(-max_shift, max_shift)

    new_start = max(0, min(data_length - length, start + shift))
    new_end = new_start + length

    return new_start, new_end

# =============================================
# Algoritmo Evolutivo Principal
# =============================================
def run_evolutionary_algorithm(data, generations=100, population_size=20, window_length=50):
    """
    Ejecuta el algoritmo evolutivo para detectar patrones en la serie temporal.
    
    :param data: Serie temporal (array numpy)
    :param generations: Número de generaciones
    :param population_size: Tamaño de la población
    :param window_length: Longitud de las ventanas
    :return: Individuo con el mejor fitness
    """
    # Inicialización de la población
    population = initialize_population(population_size, window_length, len(data))

    for generation in range(generations):
        # Evaluación de fitness
        fitness_scores = [fitness_function(ind, data) for ind in population]

        # Selección, cruce y mutación
        new_population = []
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            child1, child2 = single_point_crossover(parent1, parent2)
            child1 = mutation(child1, len(data))
            child2 = mutation(child2, len(data))

            new_population.extend([child1, child2])

        population = new_population[:population_size]

    # Devolver el mejor individuo encontrado
    best_individual = max(zip(population, fitness_scores), key=lambda x: x[1])
    return best_individual[0]

# =============================================
# Encontrar ventanas similares
# =============================================
def find_similar_windows(pattern, data, threshold=0.8):
    """
    Encuentra ventanas similares a un patrón base.
    :param pattern: Fragmento patrón (array)
    :param data: Serie completa
    :param threshold: Umbral de similitud (por ejemplo, 0.8)
    :return: Lista de tuplas (inicio, fin) de ventanas similares
    """
    matches = []
    win_len = len(pattern)
    for i in range(len(data) - win_len + 1):
        window = data[i:i + win_len]
        corr = np.corrcoef(pattern, window)[0, 1]
        if corr >= threshold:
            matches.append((i, i + win_len))
    return matches
