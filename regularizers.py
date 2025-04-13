import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial
import tensorflow as tf

from tensorflow.python.keras import backend
from keras import Regularizer
from keras import callbacks


class SpacialEmbedding:
    def __init__(self, neuron_num=100, network_structure=(5, 5, 4), coordinates_list=None, distance_power=1, distance_metric='euclidean'):
        self.version = 'v1.1.0'
        self.distance_power = distance_power

        # Set up tensor with distance matrix
        # Set up neurons per dimension
        nx = np.arange(network_structure[0])
        ny = np.arange(network_structure[1])
        nz = np.arange(network_structure[2])

        # Set up coordinate grid
        [x, y, z] = np.meshgrid(nx, ny, nz)
        self.coordinates = [x.ravel(), y.ravel(), z.ravel()]

        # Override coordinate grid if one if provided in init
        if coordinates_list is not None:
            self.coordinates = coordinates_list

        # Check neuron number / number of coordinates
        if (len(self.coordinates[0]) == neuron_num) & (len(self.coordinates[1]) == neuron_num) & (len(self.coordinates[2]) == neuron_num):
            pass
        else:
            raise ValueError(
                'Network / coordinate structure does not match the number of neurons.')

        # Calculate the euclidean distance matrix
        euclidean_vector = scipy.spatial.distance.pdist(
            np.transpose(self.coordinates), metric=distance_metric)
        euclidean = scipy.spatial.distance.squareform(
            euclidean_vector**self.distance_power)
        self.distance_matrix = euclidean.astype('float32')
        self.spatial_cost_matrix = self.distance_matrix

        # Add minimal cost for recurrent self connection (on diagonal)
        # diag_dist = np.diag(np.repeat(0.1,100)).astype('float32')
        # self.distance_matrix = self.distance_matrix + diag_dist

        # Create tensor from distance matrix
        self.distance_tensor = tf.convert_to_tensor(self.distance_matrix)

    def visualise_distance_matrix(self):
        plt.imshow(self.distance_matrix)
        plt.colorbar()
        plt.show()

    def visualise_neuron_structure(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(self.coordinates[0], self.coordinates[1],
                   self.coordinates[2], c='b', marker='.')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


class L1(Regularizer):
    def __init__(self, reg_factor=0.01, se:SpacialEmbedding=SpacialEmbedding()):
        self._check_penalty_number(reg_factor)

        # Transform regularisation strength to TF's standard float format
        self.reg_factor = backend.cast_to_floatx(reg_factor)

        self.se = se

    def __call__(self, x):
        # Add calculation of loss here.
        # L1 for reference: self.l1 * math_ops.reduce_sum(math_ops.abs(x))
        l1_loss = self.reg_factor * tf.math.reduce_sum(tf.math.abs(x))

        return l1_loss

    def _check_penalty_number(self, x):
        """check penalty number availability, raise ValueError if failed."""
        if not isinstance(x, (float, int)):
            raise ValueError(('Value: {} is not a valid regularization penalty number, '
                              'expected an int or float value').format(x))

    def get_config(self):
        return {'regularization factor': float(self.reg_factor)}


class SE1(L1):
    """A regulariser for sptially embedded RNNs.
    Applies L1 regularisation to recurrent kernel of
    RNN which is weighted by the distance of units
    in predefined 3D space.
    Calculation:
        se1 * sum[distance_matrix o recurrent_kernel]
    Attributes:
        se1: Float; Weighting of SE1 regularisation term.
        distance_tensor: TF tensor / matrix with cost per
        connection in weight matrix of network.
    """

    def __call__(self, x):
        # Add calculation of loss here.
        # L1 for reference: self.l1 * math_ops.reduce_sum(math_ops.abs(x))
        abs_weight_matrix = tf.math.abs(x)

        # se1_loss = self.se1 * tf.math.multiply(abs_weight_matrix, self.distance_tensor)
        # se1_loss = tf.math.reduce_sum(abs_weight_matrix)
        se1_loss = self.reg_factor * \
            tf.math.reduce_sum(tf.math.multiply(
                abs_weight_matrix, self.se.distance_tensor))

        return se1_loss


class SE1_sWc(SE1):
    '''
    Version of SE1 regulariser which combines the spatial and communicability parts in loss function.
    Additional comms_factor scales the communicability matrix.
    The communicability term used here is unbiased weighted communicability:
    Crofts, J. J., & Higham, D. J. (2009). A weighted communicability measure applied to complex brain networks. Journal of the Royal Society Interface, 6(33), 411-414.
    '''

    def __init__(self, reg_factor=0.01, comms_factor=1, se:SpacialEmbedding=SpacialEmbedding()):
        SE1.__init__(self, reg_factor, se)
        self.comms_factor = comms_factor

    def __call__(self, x):
        # Take absolute of weights
        abs_weight_matrix = tf.math.abs(x)

        # Calulcate weighted communicability (see reference in docstring)
        stepI = tf.math.reduce_sum(abs_weight_matrix, axis=1)
        stepII = tf.math.pow(stepI, -0.5)
        stepIII = tf.linalg.diag(stepII)
        stepIV = tf.linalg.expm(stepIII@abs_weight_matrix@stepIII)
        comms_matrix = tf.linalg.set_diag(stepIV, tf.zeros(stepIV.shape[0:-1]))

        # Multiply absolute weights with communicability weights
        comms_matrix = comms_matrix**self.comms_factor
        comms_weight_matrix = tf.math.multiply(abs_weight_matrix, comms_matrix)

        # Multiply comms weights matrix with distance tensor, scale the mean, and return as loss
        se1_loss = self.reg_factor * \
            tf.math.reduce_sum(tf.math.multiply(
                comms_weight_matrix, self.se.distance_tensor))

        return se1_loss


class RNNHistoryI(callbacks.Callback):
    '''
    Saves the RNNs weight matrix and coordinate matrix to the training history before
    the start of training and after finishing each epoch.

    The network model needs to be build manually before calling fit() method
    for this callback to work.
    '''

    def __init__(self, RNN_layer_number=0):
        super(RNNHistoryI, self).__init__()
        self.RNN_layer_number = RNN_layer_number

    def on_train_begin(self, logs=None):
        # Create key for RNN_Weight_Matrix in history
        self.model.history.history['weight_matrix'] = []
        self.model.history.history['coordinates'] = []
        # print("Created key for RNN_Weight_Matrix in history.")

        # Save matrix before start of training
        self.model.history.history['weight_matrix'].append(
            self.model.layers[self.RNN_layer_number].get_weights()[1])
        self.model.history.history['coordinates'].append(
            self.model.layers[self.RNN_layer_number].recurrent_regularizer.se.coordinates)
        # print("Saved RNN_Weight_Matrix to history.")

    def on_epoch_end(self, epoch, logs=None):
        # Save RNN_Weight_Matrix to history
        self.model.history.history['weight_matrix'].append(
            self.model.layers[self.RNN_layer_number].get_weights()[1])
        self.model.history.history['coordinates'].append(
            self.model.layers[self.RNN_layer_number].recurrent_regularizer.se.coordinates)
        # print("Saved RNN_Weight_Matrix to history.")
