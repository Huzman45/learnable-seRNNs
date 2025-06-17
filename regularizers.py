# Partially based on the Capsule for Achterberg & Akarca, et al:
# Spatially-embedded recurrent neural networks reveal
# widespread structural and functional neuroscience findings
# https://codeocean.com/capsule/2879348/tree/v2

import numpy as np
import scipy.spatial
import tensorflow as tf

from tensorflow.python.keras import backend
import keras
from keras import Regularizer
from keras import layers
from keras import callbacks
import torch


class SpacialEmbedding:
    def __init__(
        self,
        coordinates_list: tf.Tensor | tf.Variable = None,
        network_structure: tuple[int] | list[int] = (5, 5, 4),
        distance_power: float = 1.0,  # Changed to float for TensorFlow compatibility
        distance_metric: str = "euclidean",
    ):
        """
        Initializes the SpacialEmbedding class with a TensorFlow-based distance matrix.

        Args:
            coordinates_list (list of arrays or tf.Tensor/tf.Variable): A single TensorFlow Tensor/Variable of shape (neuron_num, num_dimensions).
            network_structure (tuple or list): A tuple or list representing the shape of the network.
            If coordinates_list is None, this will be used to generate a coordinate grid.
            distance_power (float): The power to which the distance is raised.
            distance_metric (str): The distance metric to use. Currently only 'euclidean' is fully
                                   implemented using TensorFlow operations within this class.
        """
        self.version = "v1.3.0"  # Updated version
        self.distance_power = backend.cast_to_floatx(distance_power)
        self.distance_metric = distance_metric

        self.coordinates = None
        self.distance_matrix = None

        if coordinates_list is None and network_structure is None:
            raise ValueError(
                "Either coordinates_list or network_structure must be provided."
            )
        if coordinates_list is None:
            coordinates_list = self.generate_coordinate_grid(
                network_structure=network_structure
            )

        # Calculate the distance matrix
        self.calculate_distance_matrix(coordinates_list=coordinates_list)

    def generate_coordinate_grid(self, network_structure):
        # Set up tensor with distance matrix
        # Set up neurons per dimension
        nx = np.arange(network_structure[0])
        ny = np.arange(network_structure[1])
        nz = np.arange(network_structure[2])

        # Set up coordinate grid
        [x, y, z] = np.meshgrid(nx, ny, nz)
        coordinate_list = [x.ravel(), y.ravel(), z.ravel()]
        return tf.convert_to_tensor(coordinate_list, dtype=tf.float32)

    def calculate_distance_matrix(self, coordinates_list):
        """
        Calculates the distance matrix based on the coordinates provided.
        """
        if not isinstance(
            coordinates_list,
            (tf.Tensor, tf.Variable, keras.Variable, keras.KerasTensor),
        ):
            raise ValueError(
                f"coordinates_list is a {type(coordinates_list)}. It must be a TensorFlow tensor or variable."
            )

        self.coordinates = coordinates_list  # Use the TF tensor directly
        coordinates_tensor = tf.transpose(self.coordinates)
        if coordinates_tensor.shape.rank != 2:
            raise ValueError(
                "If coordinates_list must have rank 2 (num_dimensions, neuron_num)."
            )

        # Calculate the distance matrix using TensorFlow operations
        if self.distance_metric == "euclidean":
            # Expand dimensions for broadcasting: (N, 1, D) and (1, N, D)
            tf.debugging.check_numerics(
                coordinates_tensor, "Coordinates tensor contains NaN or Inf"
            )  # Check for NaN or Inf in the coordinates tensor
            coords_expanded_1 = tf.expand_dims(coordinates_tensor, 1)
            coords_expanded_2 = tf.expand_dims(coordinates_tensor, 0)

            # Compute squared differences: (N, N, D)
            squared_diffs = tf.square(coords_expanded_1 - coords_expanded_2)
            tf.debugging.check_numerics(
                squared_diffs, "Squared differences contain NaN or Inf"
            )  # Check for NaN or Inf in the squared differences

            # Sum squared differences along the dimension axis: (N, N)
            sum_squared_diffs = tf.reduce_sum(squared_diffs, axis=-1)
            sum_squared_diffs = tf.maximum(sum_squared_diffs, 10**-4)
            tf.debugging.check_numerics(
                sum_squared_diffs, "Distance matrix contains NaN or Inf before sqrt."
            )  # Check for NaN or Inf in the sum of squared differences

            # Take the square root for Euclidean distance: (N, N)
            distance_matrix_tf = tf.sqrt(sum_squared_diffs)
            tf.debugging.check_numerics(
                distance_matrix_tf,
                "Distance matrix contains NaN or Inf after sqrt.",
            )

            # Apply the distance power
            self.distance_matrix = tf.pow(distance_matrix_tf, self.distance_power)

        else:
            # For metrics not easily implemented with TF ops, fall back to SciPy
            # Note: This will return a NumPy array, not a TensorFlow tensor initially
            print(
                f"Warning: Distance metric '{self.distance_metric}' is not fully implemented with TensorFlow ops. Using SciPy."
            )
            coords_np = np.array(self.coordinates)
            coords_np = np.transpose(self.coordinates)

            euclidean_vector = scipy.spatial.distance.pdist(
                coords_np, metric=self.distance_metric
            )
            distance_matrix_np = scipy.spatial.distance.squareform(
                euclidean_vector**self.distance_power
            )
            self.distance_matrix = tf.constant(
                distance_matrix_np.astype("float32")
            )  # Convert to TF constant


class DynamicSpacialRegularizer(Regularizer):
    """
    A custom regularizer that applies a spatial regularization term to the weights of a layer.
    This is a placeholder for future implementations.
    """

    def __init__(
        self,
        reg_factor=0.01,
        distance_metric="euclidean",
        distance_power=1.0,
        reduce="mean",
    ):
        self.reg_factor = backend.cast_to_floatx(reg_factor)
        self.distance_metric = distance_metric
        self.reduce = reduce
        if self.reduce not in ["sum", "mean"]:
            raise ValueError(
                f"Invalid reduce method: {self.reduce}. Must be 'sum' or 'mean'."
            )
        self.distance_power = backend.cast_to_floatx(distance_power)

    def __call__(self, x):
        tf.debugging.check_numerics(x, "Input tensor x contains NaN or Inf")
        coordinates_tensor = tf.transpose(x)

        # Expand dimensions for broadcasting: (N, 1, D) and (1, N, D)
        coords_expanded_1 = tf.expand_dims(coordinates_tensor, 1)
        coords_expanded_2 = tf.expand_dims(coordinates_tensor, 0)

        # # Compute squared differences: (N, N, D)
        squared_diffs = tf.square(coords_expanded_1 - coords_expanded_2)

        # Sum squared differences along the dimension axis: (N, N)
        sum_squared_diffs = tf.reduce_sum(squared_diffs, axis=-1)
        sum_squared_diffs = tf.maximum(sum_squared_diffs, 10**-4)

        # Take the square root for Euclidean distance: (N, N)
        distance_matrix = tf.sqrt(sum_squared_diffs)

        # Apply the distance power
        distance_matrix = tf.pow(distance_matrix, self.distance_power)

        if self.reduce == "mean":
            ret = self.reg_factor * tf.reduce_mean(distance_matrix)
        elif self.reduce == "sum":
            ret = self.reg_factor * tf.reduce_sum(distance_matrix)

        return ret


class SpacialRegularizer(Regularizer):
    """
    Spacial regularizer base class.
    This class is used to create a custom regularizer for recurrent layers
    that incorporates spatial information.
    """

    def __init__(self, se: SpacialEmbedding = SpacialEmbedding()):
        self.se = se


class L1(SpacialRegularizer):
    def __init__(
        self,
        reg_factor=0.01,
        reduce="mean",
        se: SpacialEmbedding = SpacialEmbedding(),
    ):
        self._check_penalty_number(reg_factor)
        self.reduce = reduce
        if self.reduce not in ["sum", "mean"]:
            raise ValueError(
                f"Invalid reduce method: {self.reduce}. Must be 'sum' or 'mean'."
            )

        # Transform regularisation strength to TF's standard float format
        self.reg_factor = backend.cast_to_floatx(reg_factor)

        super().__init__(se=se)

    def _calc_l1(self, x, reduce="mean"):
        # Multiply comms weights matrix with distance tensor, scale the mean, and return as loss
        if reduce == "mean":
            return self.reg_factor * tf.math.reduce_mean(x)
        elif reduce == "sum":
            return self.reg_factor * tf.math.reduce_sum(x)

    def __call__(self, x):
        # Add calculation of loss here.
        # L1 for reference: self.l1 * math_ops.reduce_sum(math_ops.abs(x))
        return self._calc_l1(tf.math.abs(x), reduce=self.reduce)

    def _check_penalty_number(self, x):
        """check penalty number availability, raise ValueError if failed."""
        if not isinstance(x, (float, int)):
            raise ValueError(
                (
                    "Value: {} is not a valid regularization penalty number, "
                    "expected an int or float value"
                ).format(x)
            )

    def get_config(self):
        return {"regularization factor": float(self.reg_factor)}


class L1_sWc(L1):
    """
    Version of L1 regulariser which combines the spatial and communicability parts in loss function.
    Additional comms_factor scales the communicability matrix.
    The communicability term used here is unbiased weighted communicability:
    Crofts, J. J., & Higham, D. J. (2009). A weighted communicability measure applied to complex brain networks. Journal of the Royal Society Interface, 6(33), 411-414.
    """

    def __init__(
        self,
        reg_factor=0.01,
        reduce="mean",
        comms_factor=1,
        se: SpacialEmbedding = SpacialEmbedding(),
    ):
        super().__init__(reg_factor, reduce, se)
        self.comms_factor = backend.cast_to_floatx(comms_factor)

    def _calc_comms_matrix(self, x):
        tf.debugging.check_numerics(x, "Weight matrix contains NaN or Inf")

        # Take absolute of weights
        abs_weight_matrix = tf.math.abs(x)

        # Calulcate weighted communicability (see reference in docstring)
        stepI = tf.math.reduce_sum(abs_weight_matrix, axis=1)
        stepI = tf.maximum(stepI, 10**-4)
        stepII = tf.math.pow(stepI, -0.5)
        stepIII = tf.linalg.diag(stepII)
        stepIV = tf.linalg.expm(stepIII @ abs_weight_matrix @ stepIII)
        comms_matrix = tf.linalg.set_diag(stepIV, tf.zeros(stepIV.shape[0:-1]))

        # Multiply absolute weights with communicability weights
        comms_matrix = comms_matrix * self.comms_factor

        return comms_matrix

    def __call__(self, x):
        # Multiply comms weights matrix with distance tensor, scale the mean, and return as loss
        return self._calc_l1(self._calc_comms_matrix(x), self.reduce)


class SE1(L1):
    """A regulariser for sptially embedded RNNs.
    Applies L1 regularisation to recurrent kernel of
    RNN which is weighted by the distance of units
    in predefined 3D space.
    Calculation:
        se1 * sum[distance_matrix o recurrent_kernel]
    Attributes:
        se1: Float; Weighting of SE1 regularisation term.
        connection in weight matrix of network.
    """

    def __call__(self, x):
        # Add calculation of loss here.
        # L1 for reference: self.l1 * math_ops.reduce_sum(math_ops.abs(x))
        abs_weight_matrix = tf.math.abs(x)

        return self._calc_l1(
            tf.math.multiply(abs_weight_matrix, self.se.distance_matrix), self.reduce
        )  # Multiply comms weights matrix with distance tensor, scale the mean, and return as loss


class SE1_sWc(L1_sWc):
    """
    Version of SE1 regulariser which combines the spatial and communicability parts in loss function.
    Additional comms_factor scales the communicability matrix.
    The communicability term used here is unbiased weighted communicability:
    Crofts, J. J., & Higham, D. J. (2009). A weighted communicability measure applied to complex brain networks. Journal of the Royal Society Interface, 6(33), 411-414.
    """

    def __init__(
        self,
        reg_factor=0.01,
        comms_factor=1,
        reduce="mean",
        se: SpacialEmbedding = SpacialEmbedding(),
    ):
        super().__init__(reg_factor, reduce, comms_factor, se)

    def __call__(self, x):
        # Take absolute of weights
        comms_weight_matrix = self._calc_comms_matrix(x)
        penalty_matrix = tf.math.multiply(comms_weight_matrix, self.se.distance_matrix)
        penalty_matrix = tf.clip_by_value(
            penalty_matrix, clip_value_min=-100.0, clip_value_max=100.0
        )  # Cap the penalty matrix to avoid extreme values

        return self._calc_l1(penalty_matrix, reduce=self.reduce)


class SE1_sWc_repulsion(SE1_sWc):
    def __init__(
        self,
        reg_factor=0.01,
        comms_factor=1,
        repulsion_distance=0.5,
        reduce="mean",
        se: SpacialEmbedding = SpacialEmbedding(),
    ):
        super().__init__(reg_factor, comms_factor, reduce, se)
        self.repulsion_distance = backend.cast_to_floatx(repulsion_distance)

    def _calc_repulsion(self, x, a=0.1, b=0.5):
        """
        Calculate the repulsion term based on the distance matrix and the weights.
        The repulsion term is calculated as:
            a/(x-b)
        where x is the distance matrix.
        """

        return tf.math.divide(a, tf.maximum(x - b, 10**-4))

    def __call__(self, x):
        # Take absolute of weights
        comms_weight_matrix = self._calc_comms_matrix(x)
        tf.debugging.check_numerics(
            comms_weight_matrix, "Comms weight matrix contains NaN or Inf"
        )
        penalty_matrix = tf.math.multiply(comms_weight_matrix, self.se.distance_matrix)
        tf.debugging.check_numerics(
            penalty_matrix, "Penalty matrix before clipping contains NaN or Inf"
        )  # Check for NaN or Inf in the penalty matrix
        penalty_matrix = tf.clip_by_value(
            penalty_matrix, clip_value_min=-100.0, clip_value_max=100.0
        )  # Cap the penalty matrix to avoid extreme values

        # Calculate the repulsion term
        repulsion_term = self._calc_repulsion(
            self.se.distance_matrix, b=self.repulsion_distance
        )
        repulsion_term = tf.clip_by_value(
            repulsion_term, clip_value_min=0.0, clip_value_max=100.0
        )  # Cap the repulsion term to avoid extreme values
        tf.debugging.check_numerics(
            repulsion_term, "Repulsion term contains NaN or Inf"
        )  # Check for NaN or Inf in the repulsion term

        # Combine the penalty matrix and the repulsion term
        combined_penalty = penalty_matrix + repulsion_term
        tf.debugging.check_numerics(
            combined_penalty,
            "Combined penalty contains NaN or Inf.",
        )

        loss = self._calc_l1(combined_penalty, reduce=self.reduce)
        tf.debugging.check_numerics(
            loss, "Loss contains NaN or Inf"
        )  # Check for NaN or Inf in the loss
        return loss


class SE1_sWc_repulsion_2(SE1_sWc_repulsion):
    def _calc_repulsion(self, x, a=10, b=0.5):
        """
        Calculate the repulsion term based on the distance matrix and the weights.
        The repulsion term is calculated as:
            ae^(-x/b)
        where x is the distance matrix.
        """
        b = max(b, 10**-4)  # Ensure b is not zero to avoid division by zero
        return a * tf.exp(-x / b)


class SE1_repulsion(SE1_sWc_repulsion_2):
    def _calc_comms_matrix(self, x):
        tf.debugging.check_numerics(x, "Weight matrix contains NaN or Inf")
        return tf.abs(x)


class RNNHistoryI(callbacks.Callback):
    """
    Saves the RNNs weight matrix and coordinate matrix to the training history before
    the start of training and after finishing each epoch.

    The network model needs to be build manually before calling fit() method
    for this callback to work.
    """

    def __init__(self, RNN_layer_number=0):
        super(RNNHistoryI, self).__init__()
        self.RNN_layer_number = RNN_layer_number

    def on_train_begin(self, logs=None):
        # Create key for RNN_Weight_Matrix in history
        self.model.history.history["weight_matrix"] = []
        self.model.history.history["coordinates"] = []
        # print("Created key for RNN_Weight_Matrix in history.")

        # Save matrix before start of training
        for weight in self.model.layers[self.RNN_layer_number].weights:
            if weight.name == "recurrent_kernel":
                self.model.history.history["weight_matrix"].append(weight.numpy())
                break

        self.model.history.history["coordinates"].append(
            np.array(
                self.model.layers[
                    self.RNN_layer_number
                ].recurrent_regularizer.se.coordinates
            )
        )
        # print("Saved RNN_Weight_Matrix to history.")

    def on_epoch_end(self, epoch, logs=None):
        # Save RNN_Weight_Matrix to history
        for weight in self.model.layers[self.RNN_layer_number].weights:
            if weight.name == "recurrent_kernel":
                self.model.history.history["weight_matrix"].append(weight.numpy())
                break

        self.model.history.history["coordinates"].append(
            np.array(
                self.model.layers[
                    self.RNN_layer_number
                ].recurrent_regularizer.se.coordinates
            )
        )
        # print("Saved RNN_Weight_Matrix to history.")


# ----------------------------------------------------
# Torch Regularizers
# ----------------------------------------------------


class PyTorchL1:
    """
    A custom L1 regularizer for PyTorch.
    """

    def __init__(self, reg_factor=0.01, reduce="mean"):
        self.reg_factor = reg_factor
        self.reduce = reduce
        if self.reduce not in ["sum", "mean"]:
            raise ValueError(
                f"Invalid reduce method: {self.reduce}. Must be 'sum' or 'mean'."
            )

    def _calc_l1(self, x, reduce="mean"):
        # Multiply comms weights matrix with distance tensor, scale the mean, and return as loss
        if reduce == "mean":
            return self.reg_factor * torch.mean(x)
        elif reduce == "sum":
            return self.reg_factor * torch.sum(x)

    def __call__(self, weights, distance_matrix):
        # Add calculation of loss here.
        # L1 for reference: self.l1 * math_ops.reduce_sum(math_ops.abs(x))
        return self._calc_l1(torch.abs(weights), reduce=self.reduce)


class PyTorchSE1(PyTorchL1):
    def __call__(self, weights, distance_matrix):
        # Add calculation of loss here.
        # L1 for reference: self.l1 * math_ops.reduce_sum(math_ops.abs(x))
        abs_weight_matrix = torch.abs(weights)

        return self._calc_l1(torch.mul(abs_weight_matrix, distance_matrix), self.reduce)


class PyTorchSE1_repulsion(PyTorchSE1):
    def __init__(self, reg_factor=0.01, reduce="mean", repulsion_distance=0.5):
        super().__init__(reg_factor, reduce)
        self.repulsion_distance = repulsion_distance

    def _calc_repulsion(self, x, a=0.1, b=0.5):
        """
        Calculate the repulsion term based on the distance matrix and the weights.
        The repulsion term is calculated as:
            ae^(-x/b)
        where x is the distance matrix.
        """
        return a * torch.exp(-x / b)

    def __call__(self, weights, distance_matrix):
        # Add calculation of loss here.
        # L1 for reference: self.l1 * math_ops.reduce_sum(math_ops.abs(x))
        abs_weight_matrix = torch.abs(weights)

        penalty_matrix = torch.mul(abs_weight_matrix, distance_matrix)
        penalty_matrix = torch.clamp(penalty_matrix, min=-100.0, max=100.0)

        # Calculate the repulsion term
        repulsion_term = self._calc_repulsion(
            distance_matrix, b=self.repulsion_distance
        )
        repulsion_term = torch.clamp(repulsion_term, min=0.0, max=100.0)

        # Combine the penalty matrix and the repulsion term
        combined_penalty = penalty_matrix + repulsion_term
        return self._calc_l1(combined_penalty, reduce=self.reduce)
