import pickle
import matplotlib.pyplot as plt


# history = ['RNN_Weight_Matrix', 'accuracy', 'loss', 'val_accuracy', 'val_loss']
# history[key] = [(mean, std), ...]
# len(history[key]) = num_epochs
def load_history(file_path) -> dict[str, list[tuple[float, float]]]:
    with open(file_path, "rb") as file:
        history = pickle.load(file)

    return history


def plot_histories(
    histories: list[dict[str, list[tuple[float, float]]]],
    legend: list[str],
    metric: str,
    title: str,
) -> None:
    plt.figure()
    for history, label in zip(histories, legend):
        values = history[metric]
        mean = [value[0] for value in values]
        std = [value[1] for value in values]
        epochs = range(1, len(mean) + 1)
        plt.plot(epochs, mean, label=label)
        plt.fill_between(
            epochs,
            [m - s for m, s in zip(mean, std)],
            [m + s for m, s in zip(mean, std)],
            alpha=0.2,
        )

    plt.xlabel("Epochs")
    plt.ylabel(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    history_path = "data/history.pickle"
    history = load_history(history_path)
    plot_histories([history], ["RNN"], "accuracy", "Accuracy")
    plot_histories([history], ["RNN"], "loss", "Loss")
    plot_histories([history], ["RNN"], "val_accuracy", "Validation Accuracy")
    plot_histories([history], ["RNN"], "val_loss", "Validation Loss")
