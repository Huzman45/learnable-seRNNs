import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import utils

os.makedirs("plots/visualisations", exist_ok=True)

# Plots (multiple networks across all epochs)


def plot_histories(histories: list[list[dict[list[float]]]],
                   legend: list[str],
                   metric: str,
                   ) -> None:
    titles = {
        "accuracy": "Accuracy",
        "loss": "Loss",
        "val_accuracy": "Validation Accuracy",
        "val_loss": "Validation Loss",
    }
    if metric not in titles:
        raise ValueError(
            f"Unknown metric: {metric}. Must be one of {titles.keys()}")
    title = titles[metric]

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
    local_statistics: list[np.ndarray], legend: list[str], metric_name: str
) -> None:
    metrics = {
        "strength": (0, "Strength"),
        "clustering": (1, "Clustering"),
        "betweenness": (2, "Betweenness"),
        "weighted_edge_length": (3, "Weighted Edge Length"),
        "communicability": (4, "Communicability"),
        "matching": (5, "Matching"),
    }
    if metric_name.lower() not in metrics:
        raise ValueError(
            f"Unknown metric: {metric_name}. Must be one of {metrics.keys()}"
        )
    metric, title = metrics[metric_name.lower()]

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
    global_statistics: list[np.ndarray], legend: list[str], metric_str: str
) -> None:
    metrics = {
        "total_weight": (0, "Total Weight"),
        "total_weighted_edge_length": (1, "Total Weighted Edge Length"),
        "global_efficiency": (2, "Global Efficiency"),
        "homophily_per_weight": (3, "Homophily per Weight"),
        "modularity": (4, "Modularity"),
        "efficiency_per_weight": (5, "Efficiency per Weight"),
        "corr_weight_distance": (6, "Corr(Weight, Distance)"),
    }
    if metric_str.lower() not in metrics:
        raise ValueError(
            f"Unknown metric: {metric_str}. Must be one of {metrics.keys()}"
        )
    metric, title = metrics[metric_str.lower()]

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


# Visualisation (single network at a single epoch)

def visualise_network(
    net: np.ndarray,
    name: str,
    epoch: int = -1,
    threshold: float = 0.05,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    coordinates = np.asarray(net["coordinates"])[epoch]
    weight_matrix = np.asarray(net["weight_matrix"])[epoch]

    ax.scatter(
        coordinates[0],
        coordinates[1],
        coordinates[2],
        c="black",
        marker="o",
        s=15,
    )
    alpha = np.abs(weight_matrix)
    for i in range(len(coordinates[0])):
        for j in range(i + 1, len(coordinates[0])):
            if alpha[i, j] > threshold:
                ax.plot(
                    [coordinates[0, i], coordinates[0, j]],
                    [coordinates[1, i], coordinates[1, j]],
                    [coordinates[2, i], coordinates[2, j]],
                    c="red" if weight_matrix[i, j] > 0 else "blue",
                    alpha=alpha[i, j],
                )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    epoch_str = "epoch_" + str(epoch if epoch != -
                                1 else len(net["coordinates"]) - 1)
    ax.set_title(f"{name}@{epoch_str}")
    plt.legend()
    os.makedirs(f"plots/visualisations/{name}", exist_ok=True)
    plt.savefig(f"plots/visualisations/{name}/{epoch_str}.png")


def visualise_weight_matrix(
    net: np.ndarray,
    name: str,
    epoch: int = -1,
) -> None:
    weight_matrix = np.asarray(net["weight_matrix"])[epoch]

    fig, ax = plt.subplots()
    im = ax.imshow(weight_matrix, cmap="berlin", interpolation="nearest")
    ax.set_xlabel("Node")
    ax.set_ylabel("Node")
    epoch_str = "epoch_" + str(epoch if epoch != -
                                1 else len(net["coordinates"]) - 1)
    ax.set_title(f"Weight matrix of {name}@{epoch_str}")
    plt.colorbar(im, ax=ax)
    os.makedirs(f"plots/visualisations/{name}", exist_ok=True)
    plt.savefig(f"plots/visualisations/{name}/weight_matrix@{epoch_str}.png")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dirs",
        nargs="+",
        default=["seRNNs"],
        help="Subdirectories within /data to load the nets from",
    )

    args = argparser.parse_args()
    histories = []
    local_stats = []
    global_stats = []
    for dir in args.dirs:
        history_path = os.path.join("data", dir, "history.pickle")
        local_stats_path = os.path.join("data", dir, "local_statistics.pickle")
        global_stats_path = os.path.join(
            "data", dir, "global_statistics.pickle")

        histories.append(utils.load_history(history_path))
        local_stats.append(utils.load_local_statistics(local_stats_path))
        global_stats.append(utils.load_global_statistics(global_stats_path))

    # Plot histories
    plot_histories(histories, args.dirs, "accuracy")
    plot_histories(histories, args.dirs, "loss")
    plot_histories(histories, args.dirs, "val_accuracy")
    plot_histories(histories, args.dirs, "val_loss")

    # Plot local statistics
    plot_local_statistics(local_stats, args.dirs, "strength")
    plot_local_statistics(local_stats, args.dirs, "clustering")
    plot_local_statistics(local_stats, args.dirs, "betweenness")
    plot_local_statistics(local_stats, args.dirs, "weighted_edge_length")
    plot_local_statistics(local_stats, args.dirs, "communicability")
    plot_local_statistics(local_stats, args.dirs, "matching")

    # Plot global statistics
    plot_global_statistics(global_stats, args.dirs, "total_weight")
    plot_global_statistics(global_stats, args.dirs,
                           "total_weighted_edge_length")
    plot_global_statistics(global_stats, args.dirs, "global_efficiency")
    plot_global_statistics(global_stats, args.dirs, "homophily_per_weight")
    plot_global_statistics(global_stats, args.dirs, "modularity")
    plot_global_statistics(global_stats, args.dirs, "efficiency_per_weight")
    plot_global_statistics(global_stats, args.dirs, "corr_weight_distance")

    # Visualise networks

    for net_name, history in zip(args.dirs, histories):
        epochs = len(history[0]["weight_matrix"])
        regularization_strength = len(history)//2/len(history) 
        for epoch in [0, epochs//2, epochs - 1]:
            visualise_network(
                history[len(history)//2],
                f"{net_name}@regstrength_{regularization_strength:.2f}",
                epoch=epoch,
                threshold=0.05,
            )
            visualise_weight_matrix(
                history[len(history)//2],
                f"{net_name}@regstrength_{regularization_strength:.2f}",
                epoch=epoch,
            )
