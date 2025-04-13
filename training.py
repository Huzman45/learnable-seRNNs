import os
import keras
import tensorflow as tf
import regularizers
import pickle


def train(
    regus: list,
    train_data: tf.data.Dataset,
    test_data: tf.data.Dataset,
    epochs=10,
    batch_size=32,
    verbose=1,
):
    """
    Train the model with the given regularizer.

    Args:
        regularizers: List of regularizers to use.
        train_data: Training data.
        test_data: Testing data.
        epochs: Number of epochs to train for.
        batch_size: Batch size for training.
        verbose: Verbosity mode (0, 1, or 2).
    """
    example_data = next(iter(train_data))

    histories = []
    for regu in regus:
        keras.backend.clear_session()

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


def save_histories(histories, path):
    """
    Save the training histories to a file.

    Args:
        histories: The training histories to save.
        filename: The filename to save to.
    """
    path = os.path.join("data", path)
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, "history.pickle")
    
    with open(filename, "wb") as file:
        pickle.dump(histories, file)