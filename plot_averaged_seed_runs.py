
import json
import matplotlib.pyplot as plt
import numpy as np

folder = "results_learningrates_averaged"
noise_types = ["clean", "aggre", "random1", "random2", "random3", "worst"]


for noise in noise_types:
    # Files for each run
    plotName = f"different learning rates {noise} noise"
    runs = {
        "1e-2": f"noise_{noise}_learning_rate_0.01_averaged_history.json",
        "1e-3": f"noise_{noise}_learning_rate_0.001_averaged_history.json",
        "1e-4": f"noise_{noise}_learning_rate_0.0001_averaged_history.json",
        "1e-5": f"noise_{noise}_learning_rate_1e-05_averaged_history.json"
    }

    # Different colors for each run
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    # Different line styles for each metric
    styles = {
        "train": "-",
        "val_noisy": "--",
        "val_clean": ":",
    }

    fig, ax = plt.subplots(3, 1, figsize=(12, 9))

    for (run_name, filename), color in zip(runs.items(), colors):

        with open(f"{folder}/{filename}", "r") as f:
            history = json.load(f)

        epochs = np.arange(1, len(history["mean_train_loss"]) + 1)

        # Loss
        for metric, style in styles.items():
            mean = np.asarray(history[f"mean_{metric}_loss"])
            std = np.asarray(history[f"std_{metric}_loss"])

            ax[0].plot(
                epochs,
                mean,
                color=color,
                linestyle=style,
                label=f"{run_name} - {metric}"
            )

            ax[0].fill_between(
                epochs,
                mean - std,
                mean + std,
                color=color,
                alpha=0.10
            )

        # Accuracy
        for metric, style in styles.items():
            mean = np.asarray(history[f"mean_{metric}_accuracy"])
            std = np.asarray(history[f"std_{metric}_accuracy"])

            ax[1].plot(
                epochs,
                mean,
                color=color,
                linestyle=style,
                label=f"{run_name} - {metric}"
            )

            ax[1].fill_between(
                epochs,
                mean - std,
                mean + std,
                color=color,
                alpha=0.10
            )

        # Overfitting gap
        mean = np.asarray(history["mean_overfitting_gap"])
        std = np.asarray(history["std_overfitting_gap"])

        ax[2].plot(
            epochs,
            mean,
            color=color,
            linestyle="-",
            label=f"{run_name}"
        )

        ax[2].fill_between(
            epochs,
            mean - std,
            mean + std,
            color=color,
            alpha=0.10
        )

    # Loss plot
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Loss")
    ax[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax[0].grid(True, alpha=0.3)

    # Accuracy plot
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Accuracy")
    ax[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax[1].grid(True, alpha=0.3)

    # Overfitting plot
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Overfitting gap")
    ax[2].set_title("Overfitting gap")
    ax[2].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"plots/{plotName}.png", dpi=300, bbox_inches="tight")
    plt.show()
