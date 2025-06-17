import os
from typing import Callable, OrderedDict
import keras
import numpy as np
import sklearn.cluster
from sklearn.metrics import silhouette_score
import tensorflow as tf
import modcog_gen
import models
from task import generate_trials
import utils


# OLD WAY OF CALCULATING ACTIVATIONS


def get_model_recurrent_weights(
    imported_model: keras.Model, example_x_data
) -> tuple[list[np.ndarray], int, str | Callable]:
    # Extract weights and number of units for recording_model
    transfer_weights = imported_model.get_weights()
    layer = imported_model.get_layer(index=1)
    no_units = layer.units

    target_shape = (example_x_data.shape[-1], no_units)
    index = 0
    for i, w in enumerate(transfer_weights):
        if w.shape == target_shape:
            index = i
            break
    transfer_weights = transfer_weights[index : index + 3]

    # Set activation function for recording_model
    recording_activation_fun = layer.activation

    return transfer_weights, no_units, recording_activation_fun


def setup_recording_model(
    transfer_weights: list[np.ndarray],
    no_units: int,
    example_x_data: np.ndarray,
    activation_fun: str | Callable = "relu",
) -> keras.Model:
    """
    Function imports model defined by modelpath from disk and creates a second model which
    removes the last layer, so that the activities of the recurrent layer are exposed as
    outputs.
    """

    # Assemble recording_model
    recording_model = keras.models.Sequential(
        [
            keras.layers.SimpleRNN(
                no_units,
                recurrent_initializer="orthogonal",
                activation=activation_fun,
                return_sequences=True,
            ),
        ]
    )
    print(f"example_x_data shape: {example_x_data.shape}")
    recording_model.build(input_shape=example_x_data.shape)
    recording_model.set_weights(transfer_weights)

    return recording_model


def compute_batch_of_trials(
    dataset: tf.data.Dataset, recording_model: keras.Model, steps: int = 5
) -> np.ndarray:
    """
    Uses computes the activations of a network across a batch of trials.
    """

    iter_dataset = iter(dataset)
    trial_record_list = []
    for _ in range(steps):
        # Get the data for the current trial
        trial_data = next(iter_dataset)
        trial_data_x = trial_data[0]
        # Save postactivation unit activities
        single_trial_activations = recording_model.predict(trial_data_x)
        single_trial_activations = np.array(single_trial_activations)
        trial_record_list.append(single_trial_activations)

    trial_record_list = np.array(trial_record_list)
    # Reshape the activations to (steps * batch_size, num_time_steps, num_nodes)
    trial_record_list = np.reshape(
        trial_record_list,
        (
            trial_record_list.shape[0] * trial_record_list.shape[1],
            trial_record_list.shape[2],
            trial_record_list.shape[3],
        ),
    )
    return trial_record_list


# Lesioning functions


def lesion_weights(
    weights: list[np.ndarray], weights_to_lesion: np.array
) -> list[np.ndarray]:
    # Assume weights is of shape [(53, 256), (256, 256), (256,)]

    weights = [np.copy(w) for w in weights]
    num_weights = len(weights_to_lesion)

    for weight in weights:
        if weight.shape[0] == num_weights:
            # If the first dimension matches the number of weights to lesion
            weight[weights_to_lesion] = 0

        if len(weight.shape) == 2:
            if weight.shape[1] == num_weights:
                # If the second dimension matches the number of weights to lesion
                weight[:, weights_to_lesion] = 0
    return weights


def lesion_clusters(
    weights: list[np.ndarray], clusters: np.ndarray
) -> list[np.ndarray]:
    """
    Lesions the weights of the model by setting the weights of the specified clusters to zero.
    """
    num_clusters = np.unique(clusters)
    num_clusters = np.sort(num_clusters)

    lesioned_weights = []
    for cluster in num_clusters:
        weights_to_lesion = clusters == cluster
        weights = lesion_weights(weights, weights_to_lesion)
        lesioned_weights.append(weights)

    return lesioned_weights


def evaluate_lesioned_models(
    model: keras.Model,
    lesioned_weights: list[np.ndarray],
    envs_list: dict = modcog_gen.modcog_envs,
    batch_size=64,
    seq_len=50,
    steps=4,
) -> list[np.ndarray]:
    model_weights = model.get_weights()
    lesioned_weights = lesioned_weights.copy()
    lesioned_weights.insert(0, model_weights)
    original_accuracies = []
    accuracy_drops = []

    full_dataset = modcog_gen.ModCogDataset(
        batch_size=batch_size,
        seq_len=seq_len,
        envs=list(envs_list.values()),
        train=False,
    )
    single_env_dataset = modcog_gen.SingleEnvDataset(
        batch_size=batch_size,
        seq_len=seq_len,
        full_envs=list(envs_list.values()),
        optimise_for_speed=True,
    )

    for ix, name in range(len(modcog_gen.original_envs.keys()) + 1):
        if ix == 0:
            name = "Full Dataset"
            dataset_gen = full_dataset.dataset
        else:
            name = list(envs_list.keys())[ix - 1]
            dataset_gen = single_env_dataset.generate_dataset(env_index=ix - 1)

        print("_" * 20)
        print(f"Calculating performance for {name}")

        accuracies = []
        for jx, weights in enumerate(lesioned_weights):
            if jx == 0:
                print("Evaluating original model")
            else:
                print(f"Evaluating cluster {jx}")

            model.set_weights(weights)
            metrics = model.evaluate(dataset_gen, steps=steps)
            accuracies.append(metrics[1])  # Assuming metrics[1] is accuracy

        accuracies = np.array(accuracies)
        if ix == 0:
            original_accuracies = accuracies
        else:
            accuracy_drop = accuracies - original_accuracies
            accuracy_drops.append(accuracy_drop)

    accuracy_drops = np.array(accuracy_drops)
    model.set_weights(model_weights)
    return accuracy_drops


# Calculate activations and variances for ModCog tasks


def calculate_modcog_activations(
    model: models.KerasModel,
    envs_list: dict = modcog_gen.modcog_envs,
    batch_size=64,
    seq_len=50,
    steps=4,
):
    """
    Calculate the variance of the activations for each task.
    """
    # Get the activations for each task

    single_env_dataset = modcog_gen.SingleEnvDataset(
        batch_size=batch_size,
        seq_len=seq_len,
        full_envs=list(envs_list.values()),
        optimise_for_speed=True,
    )
    dataset_gen = single_env_dataset.generate_dataset(env_index=0)

    activations = []
    for ix, name in enumerate(envs_list.keys()):
        print(f"Calculating activations for env {name}")
        dataset_gen = single_env_dataset.generate_dataset(env_index=ix)

        x = next(iter(dataset_gen))[0]
        activation = model.get_rnn_activities(x).numpy()
        activations.append(activation)

    return np.array(activations)


def save_activations(activations: np.ndarray, path: str, reg_strength: float | str):
    """
    Save the activations to a file.
    """
    try:
        strength_str = f"{reg_strength:.5f}"
    except TypeError:
        strength_str = str(reg_strength)

    path = os.path.join("data", path)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f"model_{strength_str}_activations.npy")

    np.save(path, activations)
    print(f"Activations saved to {path}")


def calculate_variances(activations: np.ndarray) -> np.ndarray:
    vars = np.var(activations, axis=1)  # Variance across trials
    vars = np.mean(vars, axis=1)  # Mean variance across time steps

    return vars


def normalize_variances_across_nodes(variances: np.ndarray) -> np.ndarray:
    """
    Normalize the variances across every node (divide by the max variance of the node across all tasks).
    """
    max_variances = np.max(variances, axis=0)
    if np.any(max_variances == 0):
        max_variances[max_variances == 0] = 1
    normalized_variances = variances / max_variances
    return normalized_variances


def normalize_variances_across_tasks(variances: np.ndarray) -> np.ndarray:
    """
    Normalize the variances across every task (divide by the max variance of the task across all nodes).
    """
    max_variances = np.max(variances, axis=1)
    normalized_variances = variances / max_variances[:, np.newaxis]
    return normalized_variances


# Compute variance for 20 cog task models
# Based on variance.py from the multitask repo by @gyyang https://github.com/gyyang/multitask


def _compute_variance_bymodel(
    model,
    hp,
    random_rotation=False,
):
    """Compute variance for all tasks.

    Args:
        model: network.Model instance (tf.keras.Model)
        model_dir: str, directory of the model (added for saving)
        rules: list of rules to compute variance, list of strings
        random_rotation: boolean. If True, rotate the neural activity.
    """
    h_all_byrule = OrderedDict()

    rules = hp["rules"]
    print(f"Computing variance for rules: {rules}")
    n_hidden = hp["n_rnn"]

    if random_rotation:
        from scipy.stats import ortho_group

        random_ortho_matrix = ortho_group.rvs(dim=n_hidden)

    for rule in rules:
        trial = generate_trials(rule, hp, "test", noise_on=False)

        h_tf = model.get_rnn_activities(tf.constant(trial.x, dtype=tf.float32))
        h = h_tf.numpy()

        if random_rotation:
            h = np.dot(h, random_ortho_matrix)

        # Only store variance by rule, skipping epoch-based storage
        start_fix_end = (
            trial.epochs["fix1"][1] if trial.epochs["fix1"][1] is not None else 0
        )
        h_all_byrule[rule] = h[start_fix_end:, :, :]

    # Always process for 'rule' data type
    h_all = h_all_byrule

    h_var_all = np.zeros((n_hidden, len(h_all.keys())))
    for i, val in enumerate(h_all.values()):
        h_var_all[:, i] = val.var(axis=1).mean(axis=0)

    return h_var_all


def compute_variance(model_dir, batch_major=False, regu=None, random_rotation=False):
    model_dir = os.path.join("data", model_dir)
    hp = utils.load_hp(model_dir)
    model = models.KerasModel(
        hp["n_rnn"],
        hp["n_output"],
        activation=hp["activation"],
        loss_type=hp["loss_type"],
        recurrent_regulariser=regu,
        batch_major=batch_major,
    )

    example_trial = generate_trials(
        hp["rules"][0],
        hp,
        "random",
        batch_size=hp["batch_size_train"],
    )

    try:
        _ = model(example_trial.x)
        print("\nModel built successfully with dummy input before restoration.")
    except Exception as e:
        print(
            f"\nError during initial model build for variance calculation: {e}. Model might not be fully initialized for restoration."
        )
        pass

    checkpoint = tf.train.Checkpoint(model=model)
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint).expect_partial()
        print(f"\nModel restored from {latest_checkpoint}")
    else:
        raise FileNotFoundError(
            f"No checkpoint found in {model_dir}. Please check the directory."
        )

    print("\n--- TF2.x Model Weights: State 3 (After Model Restore) ---")
    for var in model.trainable_weights:
        print(f" {var.name}")
        print(f"  {var.shape} (shape)")

    return _compute_variance_bymodel(model, hp, random_rotation)


def filter_inactive_nodes(variances: np.ndarray, threshold: float = 1e-3):
    ind_active = np.where(variances.sum(axis=0) > threshold)[0]
    return variances[:, ind_active]


# Clustering functions


def calculate_optimal_clusters(vars):
    norm_vars_to_calc = vars.T

    silhouette_scores = []
    cluster_list = []
    for k in range(2, 30):
        kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=0, n_init=20)
        kmeans.fit(norm_vars_to_calc)
        score = silhouette_score(norm_vars_to_calc, kmeans.labels_)
        silhouette_scores.append(score)
        cluster_list.append(kmeans.labels_)
    optimal_k = np.argmax(silhouette_scores) + 2
    kmeans = sklearn.cluster.KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(norm_vars_to_calc)
    clusters = kmeans.labels_
    return clusters, silhouette_scores, cluster_list, optimal_k


def sort_clusters(clusters, variances):
    variances = variances.T
    label_prefs = [
        np.argmax(variances[clusters == l].sum(axis=0)) for l in np.unique(clusters)
    ]

    unique_labels_sorted = np.unique(clusters)[np.argsort(label_prefs)]
    mapping = {
        old_label: new_label for new_label, old_label in enumerate(unique_labels_sorted)
    }
    labels = np.array([mapping[label] for label in clusters])

    ind_sort = np.argsort(labels)

    labels = labels[ind_sort]
    return labels, variances[ind_sort, :].T
