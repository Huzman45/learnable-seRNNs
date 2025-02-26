import pickle
import numpy as np
import matplotlib.pyplot as plt

import utils

def plot_histories(histories: list[list[dict[list[float]]]],
    legend: list[str],
    metric: str,
    title: str,
) -> None:
    plt.figure()
    for history, label in zip(histories, legend):
        vals = np.array([net[metric] for net in history])
        mean = np.mean(vals, axis=0)
        std = np.std(vals, axis=0)

        epochs = range(1, len(mean) + 1)
        plt.plot(epochs, mean, label=label)
        plt.fill_between(
            epochs,
            [m - s for m, s in zip(mean, std)],
            [m + s for m, s in zip(mean, std)],
            alpha=0.2,
        )

    plt.xlabel("Epochs")
    plt.ylabel(title)
    plt.legend()
    plt.savefig(f"plots/{title}.png")

def plot_local_statistics(
    local_statistics: list[np.ndarray], legend: list[str], metric: int, title: str
) -> None:
    plt.figure()
    for statistics, label in zip(local_statistics, legend):
        mean = np.mean(statistics[:, :, :, metric], axis=(0, 2))
        std = np.std(statistics[:, :, :, metric], axis=(0, 2))
        epochs = range(1, len(mean) + 1)
        plt.plot(epochs, mean, label=label)
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Epochs")
    plt.ylabel(title)
    plt.legend()
    plt.savefig(f"plots/{title}.png")

def plot_global_statistics(
    global_statistics: list[np.ndarray], legend: list[str], metric: int, title: str
) -> None:
    plt.figure()
    for statistics, label in zip(global_statistics, legend):
        mean = np.mean(statistics[:, :, metric], axis=0)
        std = np.std(statistics[:, :, metric], axis=0)
        epochs = range(1, len(mean) + 1)
        plt.plot(epochs, mean, label=label)
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Epochs")
    plt.ylabel(title)
    plt.legend()
    plt.savefig(f"plots/{title}.png")


if __name__ == "__main__":
    history_path = "data/history.pickle"
    history = utils.load_history(history_path)
    local_stats = utils.load_local_statistics()
    global_stats = utils.load_global_statistics()

    # plot_histories([history, history_2], ["RNN", "Rand"], "accuracy", "Accuracy")
    # plot_histories([history, history_2], ["RNN", "Rand"], "loss", "Loss")
    # plot_histories(
    #     [history, history_2], ["RNN", "Rand"], "val_accuracy", "Validation Accuracy"
    # )
    # plot_histories([history, history_2], ["RNN", "Rand"], "val_loss", "Validation Loss")

    plot_local_statistics([local_stats], ["seRNN"], 0, "Strength")
    plot_local_statistics([local_stats], ["seRNN"], 1, "Clustering")
    plot_local_statistics([local_stats], ["seRNN"], 2, "Betweenness")
    plot_local_statistics([local_stats], ["seRNN"], 3, "Weighted Edge Length")
    plot_local_statistics([local_stats], ["seRNN"], 4, "Communicability")
    plot_local_statistics([local_stats], ["seRNN"], 5, "Matching")

    plot_global_statistics([global_stats], ["seRNN"], 0, "Total Weight")
    plot_global_statistics([global_stats], ["seRNN"], 1, "Total Weighted Edge Length")
    plot_global_statistics([global_stats], ["seRNN"], 2, "Global Efficiency")
    plot_global_statistics([global_stats], ["seRNN"], 3, "Homophily per Weight")
    plot_global_statistics([global_stats], ["seRNN"], 4, "Modularity")
    plot_global_statistics([global_stats], ["seRNN"], 5, "Efficiency per Weight")
    plot_global_statistics([global_stats], ["seRNN"], 6, "Corr(Weight, Distance)")
    plot_global_statistics([global_stats], ["seRNN"], 7, "Small Worldness")
