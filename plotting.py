import argparse
import os
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import bct
from typing import List, Dict, Optional

import utils

default_font_size = {"title": 14, "label": 11, "legend": 10}
style = "seaborn-v0_8-paper"
file_format = "pdf"
dpi = 300
figsize: tuple = ((8, 6),)

plt.style.use(style)
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage[T1]{fontenc}"

kelly_colors = [
    np.array([0.94901961, 0.95294118, 0.95686275]),  # White/Light Gray
    np.array([0.13333333, 0.13333333, 0.13333333]),  # Black/Dark Gray
    np.array([0.95294118, 0.76470588, 0.0]),  # Yellow
    np.array([0.52941176, 0.3372549, 0.57254902]),  # Purple
    np.array([0.95294118, 0.51764706, 0.0]),  # Orange
    np.array([0.63137255, 0.79215686, 0.94509804]),  # Light Blue
    np.array([0.74509804, 0.0, 0.19607843]),  # Red
    np.array([0.76078431, 0.69803922, 0.50196078]),  # Beige
    np.array([0.51764706, 0.51764706, 0.50980392]),  # Medium Gray
    np.array([0.0, 0.53333333, 0.3372549]),  # Green
    np.array([0.90196078, 0.56078431, 0.6745098]),  # Pink
    np.array([0.0, 0.40392157, 0.64705882]),  # Dark Blue
    np.array([0.97647059, 0.57647059, 0.4745098]),  # Light Orange
    np.array([0.37647059, 0.30588235, 0.59215686]),  # Dark Purple
    np.array([0.96470588, 0.65098039, 0.0]),  # Gold
    np.array([0.70196078, 0.26666667, 0.42352941]),  # Maroon
    np.array([0.8627451, 0.82745098, 0.0]),  # Olive
    np.array([0.53333333, 0.17647059, 0.09019608]),  # Brown
    np.array([0.55294118, 0.71372549, 0.0]),  # Lime Green
    np.array([0.39607843, 0.27058824, 0.13333333]),  # Dark Brown
    np.array([0.88627451, 0.34509804, 0.13333333]),  # Burnt Orange
    np.array([0.16862745, 0.23921569, 0.14901961]),  # Dark Green
]

# Plots (multiple networks across all epochs)


def plot_histories(
    histories: List[List[Dict[str, List[float]]]],
    legend: List[str],
    metric: str,
    subdir: Optional[str] = None,
    figsize: tuple = (8, 6),
    include_legend: bool = False,
) -> None:
    """
    Plot training histories with mean and standard deviation for a given metric.

    Args:
        histories: List of lists of dictionaries, where each dictionary contains
            metric values (e.g., 'loss', 'accuracy') for each epoch across runs.
        legend: List of labels for each history in the plot.
        metric: The metric to plot (e.g., 'loss', 'accuracy', 'val_loss').
        subdir: Optional subdirectory within 'plots' to save the figure.
        file_format: File format for saving the plot (e.g., 'png', 'pdf', 'svg').
        dpi: Resolution of the saved figure in dots per inch.
        figsize: Tuple of (width, height) for the figure size in inches.
        style: Matplotlib style for professional appearance (e.g., 'seaborn', 'ggplot').

    Raises:
        ValueError: If metric is unknown, inputs are empty, or lengths mismatch.
        OSError: If directory creation or file saving fails.

    Returns:
        None: Saves the plot to a file in the specified directory.
    """
    # Input validation
    if not histories or not legend:
        raise ValueError("histories and legend must not be empty")
    if len(histories) != len(legend):
        raise ValueError(
            f"Length of histories ({len(histories)}) must match legend ({len(legend)})"
        )
    metric = metric.lower()
    titles = {
        "accuracy": "Accuracy",
        "loss": "Loss",
        "val_accuracy": "Validation Accuracy",
        "val_loss": "Validation Loss",
    }
    if metric not in titles:
        raise ValueError(
            f"Unknown metric: {metric}. Must be one of {list(titles.keys())}"
        )

    fig, ax = plt.subplots(figsize=figsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # Plot mean and std for each history
    for history, label in zip(histories, legend):
        if not history or not all(metric in net for net in history):
            raise ValueError(f"Metric '{metric}' missing in history for {label}")
        vals = np.array([net[metric] for net in history])
        mean = np.mean(vals, axis=0)
        std = np.std(vals, axis=0)

        epochs = range(1, len(mean) + 1)
        ax.plot(epochs, mean, label=label, linewidth=2)
        ax.fill_between(
            epochs,
            [m - s for m, s in zip(mean, std)],
            [m + s for m, s in zip(mean, std)],
            alpha=0.1,
        )

    # Enhance plot aesthetics
    ax.set_xlim(left=1)
    ax.set_xlabel("Epochs", fontsize=default_font_size["label"])
    ax.set_ylabel(titles[metric], fontsize=default_font_size["label"])
    if include_legend:
        ax.legend(fontsize=default_font_size["legend"], loc="best", frameon=False)

    # Save plot
    save_dir = "plots" if subdir is None else os.path.join("plots", subdir)
    try:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{metric}.{file_format}")
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi, format=file_format)
        plt.close()
    except OSError as e:
        raise OSError(f"Failed to save plot to {save_path}: {e}")


def plot_local_statistics(
    local_statistics: list[dict[str, np.ndarray]],
    legend: list[str],
    metric: str,
    subdir: str | None = None,
    verbose: bool = False,
    figsize: tuple = (8, 6),
    include_legend: bool = False,
) -> None:
    metric = metric.lower()
    title = metric.capitalize().replace("_", " ")

    fig, ax = plt.subplots(figsize=figsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    for statistics, label in zip(local_statistics, legend):
        if verbose:
            print(f"Plotting local statistic {metric} for {label}")
        if metric not in statistics.keys():
            raise ValueError(
                f"Metric {metric} not found in local statistics for {label}"
            )
        mean = np.mean(statistics[metric], axis=(0, 2))
        std = np.std(statistics[metric], axis=(0, 2))
        epochs = range(1, len(mean) + 1)
        ax.plot(epochs, mean, label=label)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.1)

    ax.set_xlim(left=1)
    ax.set_xlabel("Epochs", fontsize=default_font_size["label"])
    ax.set_ylabel(title, fontsize=default_font_size["label"])
    if include_legend:
        ax.legend(fontsize=default_font_size["legend"], loc="best", frameon=False)

    save_dir = "plots" if subdir is None else os.path.join("plots", subdir)
    save_dir = os.path.join(save_dir, f"{metric}.{file_format}")
    plt.savefig(save_dir, bbox_inches="tight", dpi=dpi, format=file_format)
    plt.close()


def plot_global_statistics(
    global_statistics: list[dict[str, np.ndarray]],
    legend: list[str],
    metric: str,
    subdir: str | None = None,
    verbose: bool = False,
    figsize: tuple = (8, 6),
    include_legend: bool = False,
) -> None:
    metric = metric.lower()
    title = (
        "Corr(Weight, Distance)"
        if metric == "corr_weight_distance"
        else metric.capitalize().replace("_", " ")
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    for statistics, label in zip(global_statistics, legend):
        if verbose:
            print(f"Plotting global statistic {metric} for {label}")
        if metric not in statistics.keys():
            raise ValueError(
                f"Metric {metric} not found in global statistics for {label}"
            )
        mean = np.mean(statistics[metric], axis=0)
        std = np.std(statistics[metric], axis=0)
        epochs = range(1, len(mean) + 1)
        ax.plot(epochs, mean, label=label)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.1)

    ax.set_xlim(left=1)
    ax.set_xlabel("Epochs", fontsize=default_font_size["label"])
    ax.set_ylabel(title, fontsize=default_font_size["label"])
    if include_legend:
        ax.legend(fontsize=default_font_size["legend"], loc="best", frameon=False)

    save_dir = "plots" if subdir is None else os.path.join("plots", subdir)
    save_dir = os.path.join(save_dir, f"{metric}.{file_format}")
    plt.savefig(save_dir, bbox_inches="tight", dpi=dpi, format=file_format)
    plt.close()


def plot_all(
    output_dir: str | None,
    networks: list[str],
    histories: list[list[dict[list[float]]]],
    local_statistics: list[dict[str, np.ndarray]],
    global_statistics: list[dict[str, np.ndarray]],
    verbose: bool = False,
) -> None:
    if output_dir is not None:
        os.makedirs(os.path.join("plots", output_dir), exist_ok=True)

    # Assume all networks have the same local and global statistics
    local_stats = local_statistics[0].keys()
    global_stats = global_statistics[0].keys()

    # Plot histories
    plot_histories(histories, networks, "accuracy", output_dir)
    plot_histories(histories, networks, "loss", output_dir)
    plot_histories(histories, networks, "val_accuracy", output_dir, include_legend=True)
    plot_histories(histories, networks, "val_loss", output_dir)

    # Plot local statistics
    for stat in local_stats:
        plot_local_statistics(local_statistics, networks, stat, output_dir, verbose)

    # Plot global statistics
    for stat in global_stats:
        plot_global_statistics(global_statistics, networks, stat, output_dir, verbose)


# Visualisation (single network at a single epoch)

os.makedirs("plots/visualisations", exist_ok=True)


def visualise_network(
    net: np.ndarray,
    name: str,
    epoch: int = -1,
    proportion: float = 0.5,
    normalise: bool = True,
    limits: np.ndarray | None = None,
    subdir: str | None = None,
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
    alpha = (alpha - np.mean(alpha)) / np.std(alpha) if normalise else alpha
    alpha = np.clip(alpha, 0, 1)
    thresh = bct.threshold_proportional(
        alpha,
        proportion,
        copy=True,
    )
    for i in range(len(coordinates[0])):
        for j in range(i + 1, len(coordinates[0])):
            if thresh[i, j]:
                ax.plot(
                    [coordinates[0, i], coordinates[0, j]],
                    [coordinates[1, i], coordinates[1, j]],
                    [coordinates[2, i], coordinates[2, j]],
                    c="red" if weight_matrix[i, j] > 0 else "blue",
                    alpha=alpha[i, j],
                )
    if limits is not None:
        ax.set_xlim(limits[0, 0], limits[1, 0])
        ax.set_ylim(limits[0, 1], limits[1, 1])
        ax.set_zlim(limits[0, 2], limits[1, 2])

        ax.set_box_aspect(
            (
                min(limits[1, 0] - limits[0, 0], 10),
                min(limits[1, 1] - limits[0, 1], 10),
                min(limits[1, 2] - limits[0, 2], 10),
            )
        )
    else:
        ax.set_box_aspect(
            (
                min(np.max(coordinates[0]) - np.min(coordinates[0]), 10),
                min(np.max(coordinates[1]) - np.min(coordinates[1]), 10),
                min(np.max(coordinates[2]) - np.min(coordinates[2]), 10),
            )
        )
    epoch_str = "epoch_" + str(epoch if epoch != -1 else len(net["coordinates"]) - 1)
    if subdir:
        name = os.path.join(subdir, name)
    os.makedirs(f"plots/visualisations/{name}", exist_ok=True)
    plt.savefig(
        f"plots/visualisations/{name}/{epoch_str}.{file_format}",
        bbox_inches="tight",
        dpi=dpi,
        format=file_format,
    )
    plt.close(fig)


def visualise_weight_matrix(
    net: np.ndarray,
    name: str,
    epoch: int = -1,
    subdir: str | None = None,
) -> None:
    weight_matrix = np.asarray(net["weight_matrix"])[epoch]

    fig, ax = plt.subplots()
    im = ax.imshow(weight_matrix, cmap="berlin", interpolation="nearest")
    ax.set_xlabel("Node", fontsize=default_font_size["label"])
    ax.set_ylabel("Node", fontsize=default_font_size["label"])
    epoch_str = "epoch_" + str(epoch if epoch != -1 else len(net["coordinates"]) - 1)
    # ax.set_title(f"Weight matrix of {name}@{epoch_str}")
    cbar = plt.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Weight Strength", fontsize=default_font_size["label"])
    if subdir:
        name = os.path.join(subdir, name)
    os.makedirs(f"plots/visualisations/{name}", exist_ok=True)
    plt.savefig(
        f"plots/visualisations/{name}/weight_matrix@{epoch_str}.{file_format}",
        bbox_inches="tight",
        dpi=dpi,
        format=file_format,
    )
    plt.close(fig)


def visualise_variances(
    net: np.ndarray,
    variances1: np.ndarray,
    variances2: np.ndarray,
    name: str,
    epoch: int = -1,
    subdir: str | None = None,
) -> None:
    coordinates = np.asarray(net["coordinates"])[epoch]

    for var1, var2 in zip(variances1, variances2):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            coordinates[0],
            coordinates[1],
            coordinates[2],
            c=var2,
            cmap="Reds",
            marker="o",
            s=var1 * 15,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()


def visualise_tsne(
    modularity: np.ndarray,
    tsne_results: np.ndarray,
    name: str,
    epoch: int = -1,
    subdir: str | None = None,
):
    structural_clusters = modularity[epoch]
    num_structural_clusters = len(np.unique(structural_clusters))
    if num_structural_clusters > 20:
        print(
            f"Too many structural clusters ({num_structural_clusters}) to visualize with a single color map."
        )

    fig, ax = plt.subplots()
    plt.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=structural_clusters,
        cmap="tab20",
        s=50,
        alpha=0.7,
    )
    plt.title("t-SNE of Functional Task Variance Colored by Structural Clustering")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.show()
    plt.close()


def visualise_cluster_score(
    silhouette_scores: np.ndarray,
    optimal_k: int,
    figsize: tuple = (8, 6),
    subdir: str | None = None,
):
    fig, ax = plt.subplots(figsize=figsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Silhouette scores: {silhouette_scores}")
    ax.plot(
        range(2, 30),
        silhouette_scores,
        marker="o",
    )
    ax.set_xlim(left=2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Number of Clusters", fontsize=default_font_size["label"])
    ax.set_ylabel("Silhouette Score", fontsize=default_font_size["label"])

    if subdir is not None:
        save_dir = os.path.join("plots", subdir)
        os.makedirs(save_dir, exist_ok=True)
        save_dir = os.path.join(save_dir, f"cluster_score.{file_format}")
        plt.savefig(save_dir, bbox_inches="tight", dpi=dpi, format=file_format)
        print(f"Saved silhouette score plot to {save_dir}")

    plt.show()
    plt.close(fig)


def visualise_task_variances(
    clusters_list: list[np.ndarray],
    titles: list[str],
    variances_list: list[np.ndarray],
    task_names: list[str],
    figsize: tuple = [10, 7],
    subdir: str | None = None,
) -> None:
    if len(task_names) <= 20:
        figsize = (10, 5)

    n_subplots = len(clusters_list)

    gs = gridspec.GridSpec(
        nrows=2,
        ncols=1 + n_subplots // 2,  # Heatmap + colorbar
        width_ratios=[5] * (n_subplots // 2) + [0.2],
        wspace=0.1,
        hspace=0.25 if len(task_names) <= 20 else 0.2,
    )
    fig = plt.figure(figsize=(figsize[0] * n_subplots // 2, figsize[1] * 2))
    axes = [fig.add_subplot(gs[j, i]) for j in range(2) for i in range(n_subplots // 2)]

    # Position for labels and rectangles
    if len(task_names) > 20:
        text_y_position = -5
        rect_y_position = -3
        rect_height = 1.7
        title_pad = 35
    else:
        text_y_position = -2.5
        rect_y_position = -1.6
        rect_height = 0.7
        title_pad = 33

    for ix, (ax, clusters, title, variances) in enumerate(
        zip(axes, clusters_list, titles, variances_list)
    ):
        row = ix >= n_subplots // 2
        # Cluster processing
        sorted_cluster_indices = np.argsort(clusters)
        ordered_clusters = clusters[sorted_cluster_indices]
        ordered_norm_vars = variances[:, sorted_cluster_indices]
        num_clusters = len(np.unique(clusters))
        print(f"Number of clusters: {num_clusters}")

        im = ax.imshow(ordered_norm_vars, aspect="auto", cmap="viridis")

        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Task (y-axis) labels
        if ix == 0 or ix == n_subplots // 2:
            ax.set_yticks(np.arange(len(task_names)))
            ax.set_yticklabels(task_names)
            ax.set_ylabel("Tasks", fontsize=default_font_size["label"])
        else:
            ax.set_yticks([])

        # Cluster (x-axis) labels
        ax.set_xticks([])
        if row == 1:
            ax.set_xlabel(
                "Nodes (sorted by cluster)", fontsize=default_font_size["label"]
            )
        ax.set_title(title, fontsize=default_font_size["title"], pad=title_pad)

        # Cluster segmentation line
        cluster_boundaries = np.where(np.diff(ordered_clusters) != 0)[0]
        for boundary in cluster_boundaries:
            ax.axvline(x=boundary, color="black", linewidth=1)

        # Cluster labels and rectangles
        unique_clusters = np.unique(ordered_clusters)
        cluster_start_indices = {}
        cluster_end_indices = {}

        for cluster_val in unique_clusters:
            indices = np.where(ordered_clusters == cluster_val)[0]
            if len(indices) > 0:
                cluster_start_indices[cluster_val] = indices.min()
                cluster_end_indices[cluster_val] = indices.max()

        for jx, cluster_val in enumerate(
            sorted(unique_clusters)
        ):  # Ensure numerical order for plotting
            start_idx = cluster_start_indices[cluster_val] - 1
            end_idx = cluster_end_indices[cluster_val] - 1
            center_idx = (start_idx + end_idx) / 2

            # Place the cluster number label
            ax.text(
                center_idx,
                text_y_position,
                str(cluster_val + 1),
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=default_font_size["label"],
                color=kelly_colors[jx + 1],
            )

            # Coloured rectangle
            rect = patches.Rectangle(
                (start_idx, rect_y_position),
                (end_idx - start_idx + 1),
                rect_height,
                facecolor=kelly_colors[jx + 1],
                edgecolor="black",
                linewidth=1,
                clip_on=False,
            )
            ax.add_patch(rect)

        if ix == n_subplots // 2 - 1 or ix == len(clusters_list) - 1:
            cbar_ax = fig.add_subplot(gs[row, n_subplots // 2])
            cbar = fig.colorbar(im, cax=cbar_ax, shrink=0.9)
            cbar.set_ticks([0, 1])
            cbar.set_label(
                "Normalised Task Variance", fontsize=default_font_size["label"]
            )

    if subdir:
        save_dir = os.path.join("plots/visualisations", subdir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"task_variances.{file_format}")
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi, format=file_format)

    plt.show()
    plt.close(fig)


def visualise_lesioned_performance(
    accuracy_drops: np.ndarray, task_names: list[str], subdir: str | None = None
):
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(accuracy_drops.T, aspect="auto", cmap="viridis")

    # Task (y-axis) labels
    ax.set_yticks(np.arange(len(task_names) + 1))
    ax.set_yticklabels(["full"] + task_names)
    ax.set_ylabel("Tasks")

    # Colorbar for normalized variance
    plt.colorbar(
        im, ax=ax, label="Performance drop", shrink=0.7
    )  # Adjust shrink for height
    # cbar.set_ticks([0,-0.5]) # Set colorbar ticks if needed

    ax.set_xlabel("Clusters")
    ax.set_xticks(np.arange(accuracy_drops.shape[0])[:-1])
    ax.set_xticklabels(np.arange(accuracy_drops.shape[0])[1:])
    ax.set_title("Accuracy Drop After Lesioning Clusters")

    plt.tight_layout()
    plt.show()


def generate_visualisations(
    output_dir: str | None,
    names: list[str],
    histories: list[list[dict[list[float]]]],
    reg_strengths_list: list[np.ndarray],
    normalise: bool = True,
    proportion: float = 0.5,
) -> None:
    if output_dir is not None:
        os.makedirs(os.path.join("plots/visualisations", output_dir), exist_ok=True)

    lims = np.zeros((2, 3))
    for net_name, history, reg_strengths in zip(names, histories, reg_strengths_list):
        net = history[len(history) // 2]
        epochs = len(net["weight_matrix"])

        for epoch in [0, epochs // 2, epochs - 1]:
            coordinates = np.asarray(net["coordinates"])[epoch]
            max_x = np.max(coordinates[0])
            min_x = np.min(coordinates[0])
            max_y = np.max(coordinates[1])
            min_y = np.min(coordinates[1])
            max_z = np.max(coordinates[2])
            min_z = np.min(coordinates[2])
            comp_lim = np.array([[min_x, min_y, min_z], [max_x, max_y, max_z]])
            lims[0] = np.minimum(lims[0], comp_lim[0]) if lims[0].any() else comp_lim[0]
            lims[1] = np.maximum(lims[1], comp_lim[1]) if lims[1].any() else comp_lim[1]

    for net_name, history, reg_strengths in zip(names, histories, reg_strengths_list):
        net = history[len(history) // 2]
        strength = reg_strengths[len(history) // 2]
        strength_str = utils.strength_to_str(strength)

        print(
            f"Visualising {net_name} with reg_strength {strength_str} from options {reg_strengths}"
        )
        epochs = len(net["weight_matrix"])

        for epoch in [0, epochs // 2, epochs - 1]:
            visualise_network(
                net,
                f"{net_name}@regstrength_{strength_str}",
                epoch=epoch,
                proportion=proportion,
                normalise=normalise,
                limits=lims,
                subdir=output_dir,
            )
            visualise_weight_matrix(
                net,
                f"{net_name}@regstrength_{strength_str}",
                epoch=epoch,
                subdir=output_dir,
            )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Name of the subdirectory in plots/ and plots/visualisations/ where graphs will be output.",
    )
    argparser.add_argument(
        "--models",
        nargs="+",
        default=["seRNNs"],
        help="Model subdirectories within data/ to load the nets from",
    )
    argparser.add_argument(
        "--stats",
        action="store_true",
        help="Plot the stats of the networks",
    )
    argparser.add_argument(
        "--visualise",
        action="store_true",
        help="Visualise the networks",
    )
    argparser.add_argument(
        "--normalise",
        action="store_true",
        help="Normalise the weights in the visualisation",
    )
    argparser.add_argument(
        "--proportion",
        type=float,
        default=0.5,
        help="Proportion of heaviest edges to keep in the visualisation",
    )

    args = argparser.parse_args()

    if not args.stats and not args.visualise:
        print("No action specified. Use --stats or --visualise.")
        exit(1)

    histories = []
    reg_strengths_list = []
    local_stats = []
    global_stats = []
    for dir in args.models:
        history = utils.load_history(dir)
        reg_strengths = [hist["reg_strength"] for hist in history]
        history, reg_strengths = utils.threshold_history_by_accuracy(
            history, reg_strengths, threshold=0.9
        )

        histories.append(history)
        reg_strengths_list.append(reg_strengths)
        if args.stats:
            local_stats.append(utils.load_local_statistics(dir))
            global_stats.append(utils.load_global_statistics(dir))

    # Plot all statistics
    if args.stats:
        plot_all(args.output_dir, args.models, histories, local_stats, global_stats)

    # Visualise networks
    if args.visualise:
        generate_visualisations(
            args.output_dir,
            args.models,
            histories,
            reg_strengths_list,
            args.normalise,
            args.proportion,
        )
