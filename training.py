import os
import keras
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import models
from tqdm import tqdm
import regularizers
import pickle

from keras import Regularizer
from typing import Callable, List, Tuple

import utils


def train_torch(
    model_generator: Callable[[int], models.PyTorchSpaciallyEmbeddedRNN],
    regus: List[regularizers.PyTorchL1],
    reg_strengths: List[float],
    train_data: Callable[
        [], Tuple[np.ndarray, np.ndarray]
    ],  # Changed type hint for Generator
    test_data: Callable[[], Tuple[np.ndarray, np.ndarray]],
    epochs: int = 10,
    steps: int = 40,
    learning_rate: float = 0.001,
    validation_steps: int = 20,
    clip_grad_norm: float = 10000.0,  # Added gradient clipping as a parameter
    device: torch.device = torch.device("mps"),  # Made device a parameter
):
    """
    Train the model with the given regularizer.

    Args:
        model_generator:  A function that creates the model.
        regus:            List of regularization functions.
        reg_strengths:    List of regularization strengths.
        train_data:       Callable that returns training data as a tuple of NumPy arrays (inputs, targets).
        test_data:        Callable that returns testing data, same format as train_data.
        epochs:           Number of epochs to train for.
        steps:            Number of training steps per epoch.
        learning_rate:    Learning rate for the optimizer.
        validation_steps: Number of validation steps per epoch.
        clip_grad_norm:   Gradient clipping value.
        device:           The device to use (e.g., 'cpu', 'cuda', 'mps').
    """
    temp_path = (
        "data/temp"  # Moved path definition outside the function for reusability
    )
    os.makedirs(temp_path, exist_ok=True)

    example_input, example_target = (
        train_data()
    )  # Get a sample to determine input/output shape

    is_one_hot_target = len(example_input.shape) == len(example_target.shape)

    print(f"Using device: {device}")

    histories = []
    for regu, strength in zip(regus, reg_strengths):
        strength_str = utils.strength_to_str(
            strength
        )  # Assuming this utility function exists
        print(f"Regularization strength: {strength_str}")

        # Assemble network
        model = model_generator(example_input.shape[-1]).to(
            device
        )  # Create model and move to device
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss().to(device)  # Move criterion to device

        history_data = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }  # INCOMPLETE

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            model.train()  # Set model to training mode
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            # Use tqdm for the training loop
            for i in tqdm(range(steps), desc="Training", unit="batch"):
                try:
                    inputs_np, targets_np = train_data()
                except tf.errors.OutOfRangeError:
                    print(f"Train dataset exhausted at step {i + 1}/{steps}. Breaking.")
                    break  # Exit the inner loop

                inputs = (
                    torch.from_numpy(inputs_np).float().to(device)
                )  # Use from_numpy and move
                targets = torch.from_numpy(targets_np).to(device)
                if is_one_hot_target:
                    targets = torch.argmax(
                        targets, dim=-1
                    )  # No need to move again, already on device

                optimizer.zero_grad()
                outputs = model(inputs)

                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = targets.view(-1)

                #  targets = targets.long() # No longer needed, CrossEntropyLoss handles both long and one-hot
                main_loss = criterion(outputs_flat, targets_flat)
                weights = model.get_recurrent_weights()
                distance_matrix = model.get_distance_matrix()
                regularization_loss = regu(
                    weights, distance_matrix
                )  # Pass the model to the regularizer
                total_loss = main_loss + regularization_loss  # Sum losses

                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), clip_grad_norm
                )  # Use parameter
                optimizer.step()

                running_loss += total_loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, -1)
                total_samples += targets.size(0) * targets.size(1)
                correct_predictions += (predicted == targets).sum().item()

            epoch_loss = running_loss / total_samples
            epoch_accuracy = correct_predictions / total_samples
            history_data["loss"].append(epoch_loss)
            history_data["accuracy"].append(epoch_accuracy)

            # Validation phase
            model.eval()  # Set model to evaluation mode
            val_running_loss = 0.0
            val_correct_predictions = 0
            val_total_samples = 0

            with torch.no_grad():
                for i in range(validation_steps):
                    try:
                        inputs_np, targets_np = test_data()
                    except tf.errors.OutOfRangeError:
                        print(
                            f"Validation dataset exhausted at step {i + 1}/{validation_steps}. Breaking."
                        )
                        break

                    inputs = torch.from_numpy(inputs_np).float().to(device)
                    targets = torch.from_numpy(targets_np).to(device)
                    if is_one_hot_target:
                        targets = torch.argmax(targets, dim=-1)
                    outputs = model(inputs)

                    outputs_flat = outputs.view(-1, outputs.size(-1))
                    targets_flat = targets.view(-1)

                    val_loss = criterion(outputs_flat, targets_flat)
                    val_running_loss += val_loss.item() * inputs.size(0)

                    _, predicted = torch.max(outputs.data, -1)
                    val_total_samples += targets.size(0) * targets.size(1)
                    val_correct_predictions += (predicted == targets).sum().item()

            val_epoch_loss = val_running_loss / val_total_samples
            val_epoch_accuracy = val_correct_predictions / val_total_samples
            history_data["val_loss"].append(val_epoch_loss)
            history_data["val_accuracy"].append(val_epoch_accuracy)

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f} - "
                f"Val Loss: {val_epoch_loss:.4f} - Val Accuracy: {val_epoch_accuracy:.4f}"
            )

        histories.append(history_data)
        model_path = os.path.join(temp_path, f"model_{strength_str}.pth")
        torch.save(model.state_dict(), model_path)
    return histories


def train_keras(
    model_generator: Callable[[Regularizer], keras.Model],
    regus: list[Regularizer],
    reg_strengths: list[float],
    train_data: tf.data.Dataset,
    test_data: tf.data.Dataset,
    epochs=10,
    steps=40,
    learning_rate=0.001,
):
    """
    Train the model with the given regularizer.

    Args:
        regularizers: List of regularizers to use.
        train_data: Training data.
        test_data: Testing data.
        epochs: Number of epochs to train for.
    """
    temp_path = os.path.join("data", "temp")
    os.makedirs(temp_path, exist_ok=True)

    example_data = next(iter(train_data))
    example_model = model_generator(regus[0])
    example_output = example_model(example_data[0])

    loss_func = (
        "categorical_crossentropy"
        if example_data[1].shape == example_output.shape
        else "sparse_categorical_crossentropy"
    )
    print(f"Loss function: {loss_func}")

    histories = []
    for regu, strength in zip(regus, reg_strengths):
        strength_str = utils.strength_to_str(strength)

        print(f"Regularization strength: {strength_str}")

        keras.backend.clear_session()

        # Assemble network
        tf_model = model_generator(regu)
        tf_model(example_data[0])

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=100.0)
        tf_model.compile(
            optimizer=optimizer,
            loss=loss_func,
            metrics=["accuracy"],
        )

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=tf_model)
        tf_model.checkpoint = checkpoint
        tf_model.checkpoint_prefix = os.path.join(
            temp_path, f"model_{strength_str}.ckpt"
        )

        # Train network
        history = tf_model.fit(
            train_data,
            epochs=epochs,
            steps_per_epoch=steps,
            validation_data=test_data,
            validation_steps=20,
            callbacks=regularizers.RNNHistoryI(RNN_layer_number=1),
        )

        history.history["reg_strength"] = strength

        histories.append(history.history)
        # Save the model
        checkpoint.save(file_prefix=tf_model.checkpoint_prefix)
        model_path = os.path.join(temp_path, f"model_{strength_str}.weights.h5")
        tf_model.save_weights(model_path)

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


def save_models(reg_strengths, path):
    """
    Save the models to a file.

    Args:
        reg_strengths: The regularization strengths to save.
        filename: The filename to save to.
    """
    path = os.path.join("data", path)
    os.makedirs(path, exist_ok=True)

    temp_path = os.path.join("data", "temp")
    for strength in reg_strengths:
        strength_str = utils.strength_to_str(strength)

        tf_model_path = os.path.join(temp_path, f"model_{strength_str}.weights.h5")
        torch_model_path = os.path.join(temp_path, f"model_{strength_str}.pth")
        checkpoint_path = os.path.join(
            temp_path, f"model_{strength_str}.ckpt-1.data-00000-of-00001"
        )
        checkpoint_index_path = os.path.join(
            temp_path, f"model_{strength_str}.ckpt-1.index"
        )
        new_tf_model_path = os.path.join(path, f"model_{strength_str}.weights.h5")
        new_torch_model_path = os.path.join(path, f"model_{strength_str}.pth")
        new_checkpoint_path = os.path.join(
            path, f"model_{strength_str}.ckpt-1.data-00000-of-00001"
        )
        new_checkpoint_index_path = os.path.join(
            path, f"model_{strength_str}.ckpt-1.index"
        )

        if os.path.exists(tf_model_path):
            os.rename(tf_model_path, new_tf_model_path)
        if os.path.exists(torch_model_path):
            os.rename(torch_model_path, new_torch_model_path)
        if os.path.exists(checkpoint_path):
            os.rename(checkpoint_path, new_checkpoint_path)
        if os.path.exists(checkpoint_index_path):
            os.rename(checkpoint_index_path, new_checkpoint_index_path)

    for file in os.listdir(temp_path):
        file_path = os.path.join(temp_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(f"Models saved to {path}")
