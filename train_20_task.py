# Based on train.py from the multitask repo by @gyyang https://github.com/gyyang/multitask

import json
import sys
import time
import os
from collections import defaultdict

import keras
import numpy as np
import tensorflow as tf

import task
from task import generate_trials
from models import KerasModel as Model, get_perf


def get_default_hp(ruleset):
    num_ring = task.get_num_ring(ruleset)
    n_rule = task.get_num_rule(ruleset)

    n_eachring = 16
    n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1
    hp = {
        "batch_size_train": 64,
        "batch_size_test": 512,
        "in_type": "normal",
        "rnn_type": "LeakyRNN",
        "loss_type": "lsq",
        "activation": "softplus",
        "tau": 100,
        "dt": 20,
        "alpha": 0.2,
        "sigma_rec": 0.05,
        "sigma_x": 0.01,
        "w_rec_init": "randortho",
        "p_weight_train": None,
        "target_perf": 1.0,
        "n_eachring": n_eachring,
        "num_ring": num_ring,
        "n_rule": n_rule,
        "rule_start": 1 + num_ring * n_eachring,
        "n_input": n_input,
        "n_output": n_output,
        "n_rnn": 256,
        "ruleset": ruleset,
        "save_name": "test",
        "learning_rate": 0.001,
        "c_intsyn": 0,
        "ksi_intsyn": 0,
    }

    return hp


@tf.function
def do_eval_step(model, trial_x, trial_y, trial_c_mask, trial_y_loc, training_mode):
    """Performs a single evaluation step."""
    y_hat_test = model(trial_x, training=False)
    loss = model.get_loss(
        y_hat_test,
        trial_y,
        trial_c_mask,
    )
    perf_test = tf.reduce_mean(get_perf(y_hat_test, trial_y_loc))
    return loss, perf_test


def evaluate(model, hp, validation_steps=16):
    for rule_test in hp["rules"]:
        batch_size_test_rep = int(hp["batch_size_test"] / validation_steps)
        clsq_tmp = list()
        perf_tmp = list()
        for i_rep in range(validation_steps):
            trial = generate_trials(
                rule_test, hp, "random", batch_size=batch_size_test_rep
            )

            c_lsq, perf_test = do_eval_step(
                model,
                tf.constant(trial.x, dtype=tf.float32),
                tf.constant(trial.y, dtype=tf.float32),
                tf.constant(trial.c_mask, dtype=tf.float32),
                tf.constant(trial.y_loc, dtype=tf.float32),
                False,
            )

            clsq_tmp.append(c_lsq.numpy())
            perf_tmp.append(perf_test.numpy())

        print(
            "{:15s}".format(rule_test)
            + "| cost {:0.6f}".format(np.mean(clsq_tmp))
            + "  | perf {:0.2f}".format(np.mean(perf_tmp))
        )
        sys.stdout.flush()


@tf.function
def train_step_fn(
    model,
    trial_x,
    trial_y,
    trial_c_mask,
    optimizer,
):
    with tf.GradientTape() as tape:
        y_hat = model(trial_x)

        total_loss = model.get_loss(
            y_hat,
            trial_y,
            trial_c_mask,
        )

    gradients = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return total_loss, y_hat


def train_loop(model, hp, optimizer, steps_per_epoch=100):
    for step in range(steps_per_epoch):
        rule_train_now = hp["rng"].choice(hp["rule_trains"], p=hp["rule_probs"])
        trial = generate_trials(
            rule_train_now, hp, "random", batch_size=hp["batch_size_train"]
        )

        total_loss, y_hat_train = train_step_fn(
            model,
            tf.constant(trial.x, dtype=tf.float32),
            tf.constant(trial.y, dtype=tf.float32),
            tf.constant(trial.c_mask, dtype=tf.float32),
            optimizer,
        )


def save_hp(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    hp_copy.pop("rng")  # rng can not be serialized
    with open(os.path.join(model_dir, "hp.json"), "w") as f:
        json.dump(hp_copy, f)


def init_new_model(
    model_dir,
    hp=None,
    seed=42,
):
    model_dir = os.path.join("data", model_dir)
    os.makedirs(model_dir, exist_ok=True)

    default_hp = get_default_hp("all")
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp["seed"] = seed
    hp["rng"] = np.random.RandomState(seed)

    hp["rule_trains"] = task.rules_dict["all"]
    hp["rules"] = hp["rule_trains"]

    rule_prob_map = dict()

    hp["rule_probs"] = None
    if hasattr(hp["rule_trains"], "__iter__"):
        rule_prob = np.array([rule_prob_map.get(r, 1.0) for r in hp["rule_trains"]])
        hp["rule_probs"] = list(rule_prob / np.sum(rule_prob))
    save_hp(hp, model_dir)

    def model_from_regu(regu=None):
        return Model(
            hp["n_rnn"],
            hp["n_output"],
            activation=hp["activation"],
            loss_type=hp["loss_type"],
            recurrent_regulariser=regu,
            dynamic_spatial=True if hp["rnn_type"] == "dynamic" else False,
        )

    for key, val in hp.items():
        print("{:20s} = ".format(key) + str(val))

    return model_from_regu, hp


def train_20Cog(
    model_dir,
    hp=None,
    regu=None,
    epochs=40,
    steps_per_epoch=500,
    validation_steps=15,
    load_dir=None,
    seed=42,
):
    """Train the network.

    Args:
        model_dir: str, training directory
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps
        display_step: int, display steps
        ruleset: the set of rules to train
        rule_trains: list of rules to train, if None then all rules possible
        rule_prob_map: None or dictionary of relative rule probability
        seed: int, random seed to be used

    Returns:
        model is stored at model_dir/model.ckpt
        training configuration is stored at model_dir/hp.json
    """
    model_gen, hp = init_new_model(
        model_dir=model_dir,
        hp=hp,
        regu=regu,
        seed=seed,
    )

    model = model_gen(regu)

    example_trial = generate_trials(
        hp["rules"][0], hp, "random", batch_size=hp["batch_size_train"]
    )
    model(example_trial.x)

    optimizer = keras.optimizers.Adam(learning_rate=hp["learning_rate"])

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    model.checkpoint = checkpoint
    model.checkpoint_prefix = os.path.join(model_dir, "ckpt")

    if load_dir is not None:
        checkpoint_name = tf.train.latest_checkpoint(load_dir)
        if checkpoint_name:  # Check if checkpoint_name is not None
            checkpoint.restore(checkpoint_name).expect_partial()
            print(f"Model {checkpoint_name} restored from {load_dir}")
        else:
            print(f"No checkpoint found in {load_dir}. Starting training from scratch.")
    else:
        pass

    print("\nStarting Custom Training Loop...")
    full_start_time = time.perf_counter()
    for epoch in range(epochs):
        try:
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Training step
            print("Starting training step...")
            start_time = time.perf_counter()
            train_loop(model, hp, optimizer, steps_per_epoch)
            end_time = time.perf_counter()
            print(f"Training time: {end_time - start_time:.2f} seconds")
            print("Training step completed.")

            # Validation step
            print("Starting validation step...")
            start_time = time.perf_counter()
            evaluate(model, hp, validation_steps)
            end_time = time.perf_counter()
            print(f"Validation time: {end_time - start_time:.2f} seconds")
            print("Validation step completed.")

            # Save checkpoint
            checkpoint.save(file_prefix=model.checkpoint_prefix)
        except KeyboardInterrupt:
            print("Optimization interrupted by user")
            break

    full_end_time = time.perf_counter()
    print(f"Total training time: {full_end_time - full_start_time:.2f} seconds")
    print("\nCustom Training Loop Finished.")
