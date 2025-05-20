import matplotlib.pyplot as plt

def plot_series(data):
    """
    Graficar la serie temporal original.
    
    :param data: Serie temporal (array o lista de valores).
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data, label="Serie Original", color="blue")
    plt.title("Serie Temporal")
    plt.xlabel("Tiempo")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_detected_patterns(data, patterns):
    """
    Graficar la serie temporal con los patrones detectados resaltados.
    
    :param data: Serie temporal (array o lista de valores).
    :param patterns: Lista de patrones detectados [(start_idx, end_idx), ...].
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data, label="Serie Original", color="blue")
    
    # Resaltar cada patrón detectado
    for start, end in patterns:
        plt.axvspan(start, end, color="red", alpha=0.3, label="Patrón Detectado")
    
    # Asegurar que la leyenda no se repita
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.title("Serie Temporal con Patrones Detectados")
    plt.xlabel("Tiempo")
    plt.ylabel("Valor")
    plt.grid(True)
    plt.show()