import matplotlib.pyplot as plt
from collections import defaultdict


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
    Graficar la serie con ventanas resaltadas como patrones detectados del mismo grupo con el mismo color.

    :param data: Serie temporal como lista o array.
    :param patterns: Lista de tuplas (start_idx, end_idx) de patrones detectados.
    """
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.plot(data, label="Serie Original", color="blue")

    # Agrupar patrones por longitud como criterio simple (puedes mejorar esto con clustering o hashing de la señal)
    grouped_patterns = defaultdict(list)
    for start, end in patterns:
        length = end - start
        grouped_patterns[length].append((start, end))

    # Pintar cada grupo con un color diferente
    for idx, (length, group) in enumerate(grouped_patterns.items()):
        color = f"C{idx % 10}"
        for start, end in group:
            ax.axvspan(start, end, color=color, alpha=0.3, label=f"Patrón tipo {idx + 1}")

    ax.set_title("Serie Temporal con Patrones Detectados")
    ax.set_xlabel("Índice de Tiempo")
    ax.set_ylabel("Valor Normalizado")
    ax.grid(True)

    # Eliminar etiquetas duplicadas en la leyenda
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.show()