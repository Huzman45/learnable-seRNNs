import pickle
import numpy as np


# history: list[dict[list[float]]] = [['weight_matrix', 'coordinates', 'accuracy', 'loss', 'val_accuracy', 'val_loss']]
# history.shape = (num_nets, 6, num_epochs)
def load_history(file_path: str = "data/history.pickle") -> np.ndarray:
    with open(file_path, "rb") as file:
        history = pickle.load(file)

    return history

# data: list[list[list[float]]]
# data.shape = (num_nets, num_epochs, num_nodes, num_local_statistics)
# local_statistics = [strength, clustering, betweenness, weighted_edge_length, communicability, matching]


def load_local_statistics(file_path: str = "data/local_statistics.pickle") -> np.ndarray:
    with open(file_path, "rb") as file:
        local_statistics = pickle.load(file)

    return local_statistics

# data: list[list[[float]]
# data.shape = (num_nets, num_epochs, num_global_statistics)
# global_statistics = [total_weight, total_weighted_edge_length, global_efficiency,
# homophily_per_weight, modularity, efficiency_per_weight, corr(weight,distance), small_worldness]


def load_global_statistics(file_path: str = "data/global_statistics.pickle") -> np.ndarray:
    with open(file_path, "rb") as file:
        global_statistics = pickle.load(file)

    return global_statistics
