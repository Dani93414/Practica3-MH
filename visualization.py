import matplotlib.pyplot as plt

def plot_series(data):
    """
    Graficar la serie temporal original.

    :param data: Serie temporal como lista o array.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data, label="Serie Original", color="blue")
    plt.title("Serie Temporal")
    plt.xlabel("Índice de Tiempo")
    plt.ylabel("Valor Normalizado")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_detected_patterns(data, patterns):
    """
    Graficar la serie con ventanas resaltadas como patrones detectados.

    :param data: Serie temporal como lista o array.
    :param patterns: Lista de tuplas (start_idx, end_idx) de patrones detectados.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data, label="Serie Original", color="blue")

    for idx, (start, end) in enumerate(patterns):
        plt.axvspan(start, end, color=f"C{idx % 10}", alpha=0.3, label=f"Patrón {idx + 1}")

    plt.title("Serie Temporal con Patrones Detectados")
    plt.xlabel("Índice de Tiempo")
    plt.ylabel("Valor Normalizado")
    plt.grid(True)
    if patterns:
        plt.legend()
    plt.tight_layout()
    plt.show()
