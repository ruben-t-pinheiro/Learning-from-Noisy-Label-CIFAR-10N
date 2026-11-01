import json
import matplotlib.pyplot as plt
import numpy as np

# Files for each run
plotName = "different learing rates worst noise"
folder = "results"
runs = {
    "1e-2": "noise_worst_learning_rate_0.01_history.json",
    "1e-3": "noise_worst_learning_rate_0.001_history.json",
    "1e-4": "noise_worst_learning_rate_0.0001_history.json",
    "1e-5": "noise_worst_learning_rate_1e-05_history.json"
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

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    for metric, style in styles.items():
        ax[0].plot(
            epochs,
            history[f"{metric}_loss"],
            color=color,
            linestyle=style,
            label=f"{run_name} - {metric}"
        )

    # Accuracy
    for metric, style in styles.items():
        ax[1].plot(
            epochs,
            history[f"{metric}_accuracy"],
            color=color,
            linestyle=style,
            label=f"{run_name} - {metric}"
        )

    # overfitting gap
    overfitting_gap = np.asarray(history[f"train_accuracy"]) - np.asarray(history[f"val_clean_accuracy"])

    ax[2].plot(
        epochs,
        overfitting_gap,
        color=color,
        linestyle="-",
        label=f"{run_name}"
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

# overfitting plot
ax[2].set_xlabel("Epoch")
ax[2].set_ylabel("Overfitting gap")
ax[2].set_title("Overfitting gap (train accuracy - clean val accuracy)")
ax[2].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
ax[2].grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig(f"plots/{plotName}")
plt.show()