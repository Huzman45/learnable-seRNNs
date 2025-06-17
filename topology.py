import math
import os
import scipy.linalg
import scipy.spatial
import utils
import bct
import numpy as np
import pickle
import argparse
import time


def small_worldness(binary_weight_matrix: np.ndarray, n_nodes: int) -> float:
    # Empirical clustering and path length
    A = binary_weight_matrix
    avg_degree = np.mean(np.sum(A, axis=0))
    # Compute the clustering coefficient and path length
    clu = np.mean(bct.clustering_coef_bu(A))
    pth = bct.efficiency_bin(A)
    # Compute the null model
    clunull = avg_degree / n_nodes
    pthnull = math.log(n_nodes) / math.log(avg_degree)
    # Compute the small worldness
    smw = np.divide(np.divide(clu, clunull), np.divide(pth, pthnull))
    return smw


valid_local_stats = [
    "strength",
    "clustering",
    "betweenness",
    "weighted_edge_length",
    "communicability",
    "matching",
    "clusters",
]
valid_global_stats = [
    "total_weight",
    "total_weighted_edge_length",
    "global_efficiency",
    "homophily_per_weight",
    "modularity",
    "efficiency_per_weight",
    "corr_weight_distance",
    "small_worldness",
]


def compute_statistics(
    history: list[dict[list[float]]],
    local_stats: list[str] = valid_local_stats,
    global_stats: list[str] = valid_global_stats,
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Compute local and global statistics from the history of the networks.
    """

    num_nets = len(history)
    num_epochs = len(history[0]["accuracy"])
    n_nodes = history[0]["weight_matrix"][0].shape[0]

    thresh = 0.1

    local_statistics = {}
    global_statistics = {}

    for stat in local_stats:
        if stat not in valid_local_stats:
            raise ValueError(f"Invalid local statistic: {stat}")
        local_statistics[stat] = np.zeros((num_nets, num_epochs, n_nodes))
    for stat in global_stats:
        if stat not in valid_global_stats:
            raise ValueError(f"Invalid global statistic: {stat}")
        global_statistics[stat] = np.zeros((num_nets, num_epochs))

    for i in range(num_nets):
        print(f"Network {i + 1}/{num_nets}")
        profile = {}
        for epoch in range(num_epochs):
            if verbose:
                print("_" * 20)
                print(f"Epoch {epoch + 1}/{num_epochs}")

            net: np.ndarray = history[i]["weight_matrix"][epoch + 1]
            abs_net: np.ndarray = np.abs(net)
            distance: np.ndarray = history[i]["coordinates"][epoch + 1]
            t1 = time.perf_counter()
            distance = scipy.spatial.distance.pdist(
                np.transpose(distance), metric="euclidean"
            )
            distance = scipy.spatial.distance.squareform(distance).astype("float32")
            t2 = time.perf_counter()
            diff = t2 - t1
            profile["distance"] = profile.get("distance", 0) + diff
            if verbose:
                print(
                    f"Distance calculation took {diff:.4f} seconds for network {i + 1}/{num_nets}"
                )
            binarised = bct.threshold_proportional(abs_net, thresh)
            binarised = (binarised > 0).astype("float32")
            t3 = time.perf_counter()
            diff = t3 - t2
            profile["binarised"] = profile.get("binarised", 0) + diff
            if verbose:
                print(
                    f"Binarisation took {diff:.4f} seconds for network {i + 1}/{num_nets}"
                )

            # Precalculate specific stats that are used in multiple calculations
            if (
                "weighted_edge_length" in local_stats
                or "total_weighted_edge_length" in global_stats
            ):
                t1 = time.perf_counter()
                weighted_edge_length = np.sum(abs_net * distance)
                t2 = time.perf_counter()
                diff = t2 - t1
                profile["weighted_edge_length"] = (
                    profile.get("weighted_edge_length", 0) + diff
                )
                profile["total_weighted_edge_length"] = (
                    profile.get("total_weighted_edge_length", 0) + diff
                )
                if verbose:
                    print(
                        f"Weighted edge length calculation took {diff:.4f} seconds for network {i + 1}/{num_nets}"
                    )
            if "matching" in local_stats or "homophily_per_weight" in global_stats:
                t1 = time.perf_counter()
                matching = np.mean(bct.matching_ind(binarised) * 2)
                t2 = time.perf_counter()
                diff = t2 - t1
                profile["matching"] = profile.get("matching", 0) + diff
                profile["homophily_per_weight"] = (
                    profile.get("homophily_per_weight", 0) + diff
                )
                if verbose:
                    print(
                        f"Matching calculation took {diff:.4f} seconds for network {i + 1}/{num_nets}"
                    )
            if (
                "total_weight" in global_stats
                or "efficiency_per_weight" in global_stats
            ):
                t1 = time.perf_counter()
                total_weight = np.sum(abs_net)
                t2 = time.perf_counter()
                diff = t2 - t1
                profile["total_weight"] = profile.get("total_weight", 0) + diff
                profile["efficiency_per_weight"] = (
                    profile.get("efficiency_per_weight", 0) + diff
                )
                if verbose:
                    print(
                        f"Total weight calculation took {diff:.4f} seconds for network {i + 1}/{num_nets}"
                    )
            if (
                "global_efficiency" in global_stats
                or "efficiency_per_weight" in global_stats
            ):
                t1 = time.perf_counter()
                global_efficiency = bct.efficiency_wei(abs_net)
                t2 = time.perf_counter()
                diff = t2 - t1
                profile["global_efficiency"] = (
                    profile.get("global_efficiency", 0) + diff
                )
                profile["efficiency_per_weight"] = (
                    profile.get("efficiency_per_weight", 0) + diff
                )
                if verbose:
                    print(
                        f"Global efficiency calculation took {diff:.4f} seconds for network {i + 1}/{num_nets}"
                    )
            if "clusters" in local_stats or "modularity" in global_stats:
                t1 = time.perf_counter()
                clusters, modularity = bct.modularity_dir(
                    abs_net,
                    gamma=1.0,  # B="negative_asym"
                )
                t2 = time.perf_counter()
                diff = t2 - t1
                profile["clusters"] = profile.get("clusters", 0) + diff
                profile["modularity"] = profile.get("modularity", 0) + diff
                if verbose:
                    print(
                        f"Clusters calculation took {diff:.4f} seconds for network {i + 1}/{num_nets}"
                    )

            # Compute local statistics
            for stat in local_stats:
                match stat:
                    case "strength":
                        t1 = time.perf_counter()
                        local_statistics[stat][i, epoch, :] = bct.strengths_und(abs_net)
                        t2 = time.perf_counter()
                        diff = t2 - t1
                        profile["strength"] = profile.get("strength", 0) + diff
                        if verbose:
                            print(
                                f"Strength calculation took {diff:.4f} seconds for network {i + 1}/{num_nets}"
                            )
                    case "clustering":
                        t1 = time.perf_counter()
                        local_statistics[stat][i, epoch, :] = bct.clustering_coef_wu(
                            abs_net
                        )
                        t2 = time.perf_counter()
                        diff = t2 - t1
                        profile["clustering"] = profile.get("clustering", 0) + diff
                        if verbose:
                            print(
                                f"Clustering calculation took {diff:.4f} seconds for network {i + 1}/{num_nets}"
                            )
                    case "betweenness":
                        t1 = time.perf_counter()
                        local_statistics[stat][i, epoch, :] = bct.betweenness_wei(
                            abs_net
                        )
                        t2 = time.perf_counter()
                        diff = t2 - t1
                        profile["betweenness"] = profile.get("betweenness", 0) + diff
                        if verbose:
                            print(
                                f"Betweenness calculation took {diff:.4f} seconds for network {i + 1}/{num_nets}"
                            )
                    case "weighted_edge_length":
                        local_statistics[stat][i, epoch, :] = weighted_edge_length
                    case "communicability":
                        t1 = time.perf_counter()
                        local_statistics[stat][i, epoch, :] = scipy.sparse.linalg.expm(
                            abs_net
                        ).mean()
                        t2 = time.perf_counter()
                        diff = t2 - t1
                        profile["communicability"] = (
                            profile.get("communicability", 0) + diff
                        )
                        if verbose:
                            print(
                                f"Communicability calculation took {diff:.4f} seconds for network {i + 1}/{num_nets}"
                            )
                    case "matching":
                        local_statistics[stat][i, epoch, :] = matching
                    case "clusters":
                        local_statistics[stat][i, epoch, :] = clusters

            # Compute global statistics
            for stat in global_stats:
                match stat:
                    case "total_weight":
                        global_statistics[stat][i, epoch] = total_weight
                    case "total_weighted_edge_length":
                        t1 = time.perf_counter()
                        global_statistics[stat][i, epoch] = weighted_edge_length.mean()
                        t2 = time.perf_counter()
                        diff = t2 - t1
                        profile["total_weighted_edge_length"] = (
                            profile.get("total_weighted_edge_length", 0) + diff
                        )
                        if verbose:
                            print(
                                f"Total weighted edge length calculation took {diff:.4f} seconds for network {i + 1}/{num_nets}"
                            )
                    case "global_efficiency":
                        global_statistics[stat][i, epoch] = global_efficiency
                    case "homophily_per_weight":
                        t1 = time.perf_counter()
                        global_statistics[stat][i, epoch] = (
                            np.mean(matching.squeeze()) / total_weight
                        )
                        t2 = time.perf_counter()
                        diff = t2 - t1
                        profile["homophily_per_weight"] = (
                            profile.get("homophily_per_weight", 0) + diff
                        )
                        if verbose:
                            print(
                                f"Homophily per weight calculation took {diff:.4f} seconds for network {i + 1}/{num_nets}"
                            )
                    case "modularity":
                        global_statistics[stat][i, epoch] = modularity
                    case "efficiency_per_weight":
                        global_statistics[stat][i, epoch] = (
                            global_efficiency / total_weight
                        )
                    case "corr_weight_distance":
                        t1 = time.perf_counter()
                        global_statistics[stat][i, epoch] = np.corrcoef(
                            distance[abs_net > 0], abs_net[abs_net > 0]
                        )[0, 1]
                        t2 = time.perf_counter()
                        diff = t2 - t1
                        profile["corr_weight_distance"] = (
                            profile.get("corr_weight_distance", 0) + diff
                        )
                        if verbose:
                            print(
                                f"Correlation calculation took {diff:.4f} seconds for network {i + 1}/{num_nets}"
                            )
                    case "small_worldness":
                        t1 = time.perf_counter()
                        global_statistics[stat][i, epoch] = small_worldness(
                            binarised, n_nodes
                        )
                        t2 = time.perf_counter()
                        diff = t2 - t1
                        profile["small_worldness"] = (
                            profile.get("small_worldness", 0) + diff
                        )
                        if verbose:
                            print(
                                f"Small worldness calculation took {diff:.4f} seconds for network {i + 1}/{num_nets}"
                            )

        if verbose:
            print(f"Network {i + 1}/{num_nets}")
            for key in profile:
                profile[key] /= num_epochs
                print(f"{key}: {profile[key]:.4f}")

    return local_statistics, global_statistics


def save_statistics(local_stats, global_stats, path):
    """
    Save the statistics to a file.

    Args:
        local_stats: The local statistics to save.
        global_stats: The global statistics to save.
        path: The path to save the statistics to.
    """
    path = os.path.join("data", path)
    os.makedirs(path, exist_ok=True)

    with open(f"{path}/local_statistics.pickle", "wb") as file:
        pickle.dump(local_stats, file)

    with open(f"{path}/global_statistics.pickle", "wb") as file:
        pickle.dump(global_stats, file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dir",
        type=str,
        default="seRNNs",
        help="Name of the subdirectory in data/ where history.pickle is contained.",
    )
    args = argparser.parse_args()
    history, _ = utils.threshold_history_by_accuracy(utils.load_history(args.dir))

    local_statistics, global_statistics = compute_statistics(history)
    save_statistics(local_statistics, global_statistics, args.dir)
