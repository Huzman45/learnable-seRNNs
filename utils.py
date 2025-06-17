import json
import os
import pickle
import numpy as np


# history: list[dict[str, Any]] = [['weight_matrix', 'coordinates', 'accuracy', 'loss',
# 'val_accuracy', 'val_loss', 'reg_strength']]
# history.shape = (num_nets, 6, num_epochs)
def load_history(dir: str) -> list[dict[str, list[float]]]:
    """
    Load the history of the networks from the given subdirectory within /data.
    The history is saved as a pickle file.
    """

    file_path = os.path.join("data", dir, "history.pickle")
    with open(file_path, "rb") as file:
        history = pickle.load(file)

    return history


# data: dict[str, list[list[list[float]]]] = [strength?, clustering?, betweenness?, weighted_edge_length?, communicability?, matching?]
# data.shape = (num_local_statistics, num_nets, num_epochs, num_nodes)
def load_local_statistics(dir: str) -> dict[str, np.ndarray]:
    """
    Load the local statistics of the networks from the given subdirectory within /data.
    The local statistics are saved as a pickle file.
    """

    file_path = os.path.join("data", dir, "local_statistics.pickle")
    with open(file_path, "rb") as file:
        local_statistics = pickle.load(file)

    return local_statistics


# data: dict[str, list[[float]]] = [total_weight?, total_weighted_edge_length?, global_efficiency?,
# homophily_per_weight?, modularity?, efficiency_per_weight?, corr(weight,distance), small_worldness?]
# data.shape = (num_global_statistics, num_nets, num_epochs)
def load_global_statistics(dir: str) -> dict[str, np.ndarray]:
    """
    Load the global statistics of the networks from the given subdirectory within /data.
    The global statistics are saved as a pickle file.
    """

    file_path = os.path.join("data", dir, "global_statistics.pickle")
    with open(file_path, "rb") as file:
        global_statistics = pickle.load(file)

    return global_statistics


def load_activations(dir: str, reg_strength: float) -> np.ndarray:
    """
    Load the activations of the networks from the given subdirectory within /data.
    The activations are saved as a numpy file.
    """

    strength_str = strength_to_str(reg_strength)
    file_path = os.path.join("data", dir, f"model_{strength_str}_activations.npy")
    activations = np.load(file_path)

    return activations


def threshold_history_by_accuracy(
    history: list[dict[list[float]]], reg_strengths: np.ndarray, threshold: float = 0.9
) -> tuple[list[dict[list[float]]], np.ndarray]:
    """
    Threshold the history by validation accuracy.
    """

    filtered_history = []
    filtered_reg_strengths = []
    for i in range(len(history)):
        # print(f"Network {i}: {history[i]['val_accuracy'][-1]}")
        if history[i]["val_accuracy"][-1] > threshold:
            filtered_history.append(history[i])
            filtered_reg_strengths.append(reg_strengths[i])

    print(
        f"Retained {len(filtered_history)} networks with final accuracy > {threshold} from {len(history)}"
    )
    return filtered_history, np.array(filtered_reg_strengths)


def strength_to_str(strength: float) -> str:
    """
    Convert the regularization strength to a string.
    """
    try:
        strength_str = f"{strength:.5f}"
    except TypeError:
        strength_str = str(strength)
    return strength_str


def load_hp(model_dir):
    """Load the hyper-parameter file of model save_name"""
    fname = os.path.join(model_dir, "hp.json")
    if not os.path.isfile(fname):
        fname = os.path.join(model_dir, "hparams.json")  # backward compat
        if not os.path.isfile(fname):
            return None

    with open(fname, "r") as f:
        hp = json.load(f)

    # Use a different seed aftering loading,
    # since loading is typically for analysis
    hp["rng"] = np.random.RandomState(hp["seed"] + 1000)
    return hp
