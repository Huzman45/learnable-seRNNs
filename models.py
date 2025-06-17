import keras
import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
import regularizers

from typing import Callable
from regularizers import SpacialRegularizer
from keras import layers
from keras import activations

device = torch.device("mps")


def orthogonal_init_(tensor, gain=1.0):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported.")
    rows = tensor.size(0)
    cols = tensor.size(1)
    flattened = tensor.new(rows, cols).normal_(0, 1)
    if rows < cols:
        flattened.transpose_(0, 1)

    # Calculate the orthonormal matrix
    q, r = torch.linalg.qr(flattened)
    # Make sure the signs match
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q.transpose_(0, 1)
    with torch.no_grad():
        tensor.copy_(q * gain)


# --- 2. Define the PyTorch Model ---
class PyTorchSpaciallyEmbeddedRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        neurons: int,
        act_space: int,
        dynamic_spacial_embedding: bool = False,
        init_coords: None | torch.Tensor = None,
        init_coords_endpoint: float = 6.0,
        stddev=0.05,
    ):
        super(PyTorchSpaciallyEmbeddedRNN, self).__init__()

        self.stddev = stddev

        if init_coords is None:
            init_coords = torch.rand((3, neurons), device=device) * init_coords_endpoint
        self.coordinates = nn.Parameter(
            init_coords, requires_grad=dynamic_spacial_embedding
        )

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=neurons,
            batch_first=True,
            nonlinearity="relu",  # Corresponds to Keras 'activation="relu"'
        )

        # Apply Orthogonal Initialization to recurrent weights
        self._initialize_orthogonal_rnn_weights()
        self.dense = nn.Linear(neurons, act_space)

    def _initialize_orthogonal_rnn_weights(self):
        orthogonal_init_(self.rnn.weight_hh_l0)

    def forward(self, x):
        if self.stddev > 0:
            noise = torch.normal(0, self.stddev, size=x.shape, device=x.device)
            x = x + noise

        rnn_output, _ = self.rnn(
            x
        )  # We only care about rnn_output for return_sequences=True
        dense_output = self.dense(rnn_output)

        return dense_output

    def get_recurrent_weights(self):
        """
        Get the weights of the model.
        """
        return self.rnn.weight_hh_l0

    def get_distance_matrix(self):
        coords = self.coordinates.T
        distance_matrix = torch.cdist(coords, coords, p=2)
        return distance_matrix


class KerasDynamicSpacialRNN(layers.SimpleRNN):
    """
    A SimpleRNN layer with additional trainable 3D coordinates for its units,
    and a regularizer applied to these coordinates.
    The coordinates do NOT influence the RNN's internal computation directly,
    only contribute to the loss via regularization.
    """

    def __init__(
        self,
        units,
        recurrent_regularizer: SpacialRegularizer,
        coord_regularizer: regularizers = None,
        coord_initializer=keras.initializers.RandomUniform(
            minval=-6.0, maxval=6.0, seed=42
        ),
        **kwargs,
    ):
        super().__init__(units, recurrent_regularizer=recurrent_regularizer, **kwargs)

        # Store custom arguments
        self.coord_initializer = keras.initializers.get(coord_initializer)
        if coord_regularizer is not None:
            self.coord_regularizer = keras.regularizers.get(coord_regularizer)
        else:
            self.coord_regularizer = None

    def build(self, input_shape):
        super().build(input_shape)

        self.neuron_coordinates = self.add_weight(
            name="neuron_coordinates",
            shape=(3, self.units),
            initializer=self.coord_initializer,
            regularizer=self.coord_regularizer,
            trainable=True,
            dtype=tf.float32,
        )
        self.recurrent_regularizer.se.calculate_distance_matrix(
            coordinates_list=self.neuron_coordinates
        )
        # self.built is set by super().build()

    def call(self, sequences, initial_state=None, mask=None, training=False):
        output = super().call(
            sequences, mask=mask, training=training, initial_state=initial_state
        )
        if training:
            self.recurrent_regularizer.se.calculate_distance_matrix(
                coordinates_list=self.neuron_coordinates
            )

        return output

    def get_config(self):
        base_config = super().get_config()
        base_config.update(
            {
                "coord_initializer": keras.initializers.serialize(
                    self.coord_initializer
                ),
                "coord_regularizer": keras.regularizers.serialize(
                    self.coord_regularizer
                ),
            }
        )
        return base_config


# Custom model to allow for access to rnn layer activities
# Based on network.py from the multitask repo by @gyyang https://github.com/gyyang/multitask


@keras.saving.register_keras_serializable()
class KerasModel(keras.Model):  # Inherit from tf.keras.Model
    """The model."""

    def __init__(
        self,
        neurons: int,
        output_features: int,
        activation="softplus",
        loss_type="softmax",
        dynamic_spatial=False,
        batch_major=False,
        recurrent_regulariser=None,
    ):
        """
        Initializing the model with information from hp

        Args:
            model_dir: string, directory of the model
            hp: a dictionary or None
            sigma_rec: if not None, overwrite the sigma_rec passed by hp
        """
        super(KerasModel, self).__init__()

        self.neurons = neurons
        self.output_features = output_features
        self.activation = activation
        self.loss_type = loss_type
        self.dynamic_spatial = dynamic_spatial
        self.batch_major = batch_major

        self._define_layers(recurrent_regulariser)

    def _define_layers(self, recurrent_regulariser=None):
        """Define all Keras layers here."""

        self.noise_layer = keras.layers.GaussianNoise(stddev=0.01)
        if self.dynamic_spatial:
            self.rnn_layer = KerasDynamicSpacialRNN(
                self.neurons,
                recurrent_regularizer=recurrent_regulariser,
                coord_initializer=keras.initializers.RandomUniform(
                    minval=-10.0, maxval=10.0, seed=42
                ),
                activation=self.activation,
                return_sequences=True,
                name="rnn",
            )
        else:
            self.rnn_layer = keras.layers.SimpleRNN(
                self.neurons,
                activation=self.activation,
                kernel_initializer="glorot_uniform",
                recurrent_initializer="orthogonal",
                recurrent_regularizer=recurrent_regulariser,
                return_sequences=True,
                name="rnn",
            )
        if self.loss_type == "lsq":
            output_activation = activations.sigmoid
        else:
            output_activation = activations.softmax

        self.output_layer = keras.layers.Dense(
            self.output_features, activation=output_activation, name="output"
        )

    def call(self, x):
        """
        Forward pass of the model.
        """

        if self.batch_major:
            # Input is in batch-major format (Batch, Time, Features)
            x_batch_major = x
        else:
            x_batch_major = tf.transpose(x, perm=[1, 0, 2])

        x_batch_major = self.noise_layer(x_batch_major)
        h_batch_major = self.rnn_layer(x_batch_major)

        if self.batch_major:
            # Output is in batch-major format (Batch, Time, Features)
            h = h_batch_major
        else:
            h = tf.transpose(h_batch_major, perm=[1, 0, 2])

        y_hat = self.output_layer(h)
        return y_hat

    def get_loss(self, y_hat, y_target, c_mask):
        """
        Compute the cost for the model.
        """
        n_output = self.output_features

        if self.loss_type == "lsq":
            y_shaped = tf.reshape(y_target, (-1, n_output))
            y_hat_shaped = tf.reshape(y_hat, (-1, n_output))
            cost = tf.reduce_mean(tf.square((y_shaped - y_hat_shaped) * c_mask))
        else:
            cost = keras.losses.categorical_crossentropy(
                y_target, y_hat, from_logits=False
            )
            cost = tf.reduce_mean(cost)

        return cost

    def get_rnn_activities(self, x):
        if self.batch_major:
            return self.rnn_layer(x)

        x_batch_major = tf.transpose(x, perm=[1, 0, 2])
        h_batch_major = self.rnn_layer(x_batch_major)
        h = tf.transpose(h_batch_major, perm=[1, 0, 2])
        return h


def tf_popvec(y):
    """Population vector read-out in tensorflow."""

    num_units = tf.shape(y)[-1]
    # Convert num_units to a float for division.
    num_units_float = tf.cast(num_units, tf.float32)
    step_size = 2 * np.pi / num_units_float  # Perform division in TF

    # Use tf.range to create the preferences.
    pref = tf.range(0.0, 2 * np.pi, delta=step_size, dtype=tf.float32)

    cos_pref = tf.cos(pref)
    sin_pref = tf.sin(pref)
    temp_sum = tf.reduce_sum(y, axis=-1)

    # Avoid division by zero
    temp_sum = tf.where(tf.equal(temp_sum, 0.0), 1e-10, temp_sum)

    temp_cos = tf.reduce_sum(y * cos_pref, axis=-1) / temp_sum
    temp_sin = tf.reduce_sum(y * sin_pref, axis=-1) / temp_sum
    loc = tf.atan2(temp_sin, temp_cos)
    return tf.math.mod(loc, 2 * np.pi)


def get_perf(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. TensorFlow tensor (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation). TensorFlow tensor (Time, Batch)

    Returns:
      perf: TensorFlow tensor (Batch,)
    """
    if len(y_hat.shape) != 3:
        raise ValueError("y_hat must have shape (Time, Batch, Unit)")

    # Only look at last time points
    y_loc_last = y_loc[-1]
    y_hat_last = y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat_last[..., 0]
    y_hat_loc = tf_popvec(y_hat_last[..., 1:])

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc_last - y_hat_loc
    dist = tf.minimum(tf.abs(original_dist), 2 * np.pi - tf.abs(original_dist))
    corr_loc = dist < 0.2 * np.pi

    # Should fixate?
    should_fix = y_loc_last < 0

    # performance (element-wise multiplication for boolean tensors)
    perf = tf.cast(should_fix, tf.float32) * tf.cast(fixating, tf.float32) + tf.cast(
        (1 - tf.cast(should_fix, tf.float32)), tf.float32
    ) * tf.cast(corr_loc, tf.float32) * tf.cast(
        (1 - tf.cast(fixating, tf.float32)), tf.float32
    )

    return perf
