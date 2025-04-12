import os
import keras
import numpy as np
import tensorflow as tf
import regularizers
import pickle


def train(
    regularizer,
    train_data: tf.data.Dataset,
    test_data: tf.data.Dataset,
    num_reg_strengths=50,
    epochs=10,
    batch_size=32,
    verbose=1,
):
    """
    Train the model with the given regularizer.

    Args:
        regularizer: The regularizer to use.
        data: The training data.
        num_reg_strengths: The number of regularization strengths to use.
        reps_per_strength: The number of repetitions for each regularization strength.
        epochs: The number of epochs to train for.
        batch_size: The batch size to use.
        verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).
    """
    example_data = next(iter(train_data))

    histories = []
    for strength in np.linspace(0, 1, num_reg_strengths):
        keras.backend.clear_session()
        regu = regularizer(strength)

        # Assemble network
        tf_model = keras.models.Sequential(
            [
                keras.layers.GaussianNoise(stddev=0.05),
                keras.layers.SimpleRNN(
                    100,
                    activation="relu",
                    recurrent_initializer="orthogonal",
                    return_sequences=False,
                    recurrent_regularizer=regu,
                ),
                keras.layers.Dense(4, activation="softmax"),
            ]
        )
        tf_model.build(input_shape=example_data[0].shape)

        tf_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train network
        history = tf_model.fit(
            train_data,
            epochs=epochs,
            validation_data=test_data,
            callbacks=regularizers.RNNHistoryI(RNN_layer_number=1),
        )

        histories.append(history.history)

    return histories


def save_histories(histories, network_name):
    """
    Save the training histories to a file.

    Args:
        histories: The training histories to save.
        filename: The filename to save to.
    """
    filename = os.path.join("data", network_name, "history.pickle")
    with open(filename, "wb") as file:
        pickle.dump(histories, file)