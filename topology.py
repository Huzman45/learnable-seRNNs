import os
import scipy.linalg
import scipy.spatial
import utils
import bct
import numpy as np
import pickle
import argparse


def small_worldness(binary_weight_matrix: np.ndarray, n_nodes: int = 100) -> float:
    # Empirical clustering and path length
    A = binary_weight_matrix
    clu = np.mean(bct.clustering_coef_bu(A))
    pth = bct.efficiency_bin(A)
    # Run nperm null models
    nperm = 1000
    cluperm = np.zeros((nperm, 1))
    pthperm = np.zeros((nperm, 1))
    for perm in range(nperm):
        Wperm = np.random.rand(n_nodes, n_nodes)
        # Make it into a matrix
        Wperm = np.matrix(Wperm)
        # Make symmetrical
        Wperm = Wperm + Wperm.T
        Wperm = np.divide(Wperm, 2)
        # Binarise
        threshold, upper, lower = 0.7, 1, 0
        Aperm = np.where(Wperm > threshold, upper, lower)
        # Take null model
        cluperm[perm] = np.mean(bct.clustering_coef_bu(Aperm))
        pthperm[perm] = bct.efficiency_bin(Aperm)
    # Take the average of the nulls
    clunull = np.mean(cluperm)
    pthnull = np.mean(pthperm)
    # Compute the small worldness
    smw = np.divide(np.divide(clu, clunull), np.divide(pth, pthnull))
    return smw


def compute_statistics(history: list[dict[list[float]]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute local and global statistics from the history of the networks.
    """
    num_nets = len(history)
    num_epochs = len(history[0]["accuracy"])

    # Local statistics = [strength, clustering, betweenness, weighted_edge_length, communicability, matching]
    n_l_stats = 6
    n_nodes = history[0]["weight_matrix"][0].shape[0]
    local_statistics = np.zeros((num_nets, num_epochs, n_nodes, n_l_stats))
    thresh = 0.1

    # Global statistics = [total_weight, total_weighted_edge_length, global_efficiency,
    # homophily_per_weight, modularity, efficiency_per_weight, corr(weight,distance), small_worldness]
    n_g_stats = 8
    global_statistics = np.zeros((num_nets, num_epochs, n_g_stats))

    for i in range(num_nets):
        for epoch in range(num_epochs):
            net: np.ndarray = np.abs(history[i]["weight_matrix"][epoch+1])
            distance: np.ndarray = history[i]["coordinates"][epoch+1]
            distance = scipy.spatial.distance.pdist(
                np.transpose(distance), metric="euclidean")
            distance = scipy.spatial.distance.squareform(
                distance).astype("float32")
            binarised = bct.threshold_proportional(net, thresh)
            binarised = (binarised > 0).astype("float32")

            local_statistics[i, epoch, :, 0] = bct.strengths_und(net)
            local_statistics[i, epoch, :, 1] = bct.clustering_coef_wu(net)
            local_statistics[i, epoch, :, 2] = bct.betweenness_wei(net)
            local_statistics[i, epoch, :, 3] = np.sum(net * distance)
            local_statistics[i, epoch, :, 4] = scipy.linalg.expm(net).mean()
            local_statistics[i, epoch, :, 5] = np.mean(
                bct.matching_ind(binarised) * 2)

            global_statistics[i, epoch, 0] = np.sum(net)
            global_statistics[i, epoch, 1] = (net * distance).mean()
            global_statistics[i, epoch, 2] = bct.efficiency_wei(net)
            global_statistics[i, epoch, 3] = np.mean(
                local_statistics[i, epoch, :, 5].squeeze())/global_statistics[i, epoch, 0]
            global_statistics[i, epoch, 4] = bct.modularity_und(net)[1]
            global_statistics[i, epoch, 5] = (
                global_statistics[i, epoch, 2] / global_statistics[i, epoch, 0]
            )
            global_statistics[i, epoch, 6] = np.corrcoef(
                distance[net > 0], net[net > 0])[0, 1]
            global_statistics[i, epoch, 7] = small_worldness(
                binarised, n_nodes)

    return local_statistics, global_statistics


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dir",
        type=str,
        default="seRNNs",
        help="Name of the subdirectory in data/ where history.pickle is contained.",
    )
    args = argparser.parse_args()
    history: np.ndarray = utils.load_history(
        os.path.join("data", args.dir, "history.pickle"))
    local_statistics, global_statistics = compute_statistics(history)

    with open(f"data/{args.dir}/local_statistics.pickle", "wb") as file:
        pickle.dump(local_statistics, file)

    with open(f"data/{args.dir}/global_statistics.pickle", "wb") as file:
        pickle.dump(global_statistics, file)
