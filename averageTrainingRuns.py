import json
import matplotlib.pyplot as plt
import numpy as np

# Files for each run
plotName = "different learning rates no noise"
folder = "results"
runs = {
    "1e-2": "noise_clean_learning_rate_0.01_history.json",
    "1e-3": "noise_clean_learning_rate_0.001_history.json",
    "1e-4": "noise_clean_learning_rate_0.0001_history.json",
    "1e-5": "noise_clean_learning_rate_1e-05_history.json"
}

fig, ax = plt.subplots(3, 1, figsize=(12, 9))

for run_name, filename in runs.items():

    with open(f"{folder}/{filename}", "r") as f:
        history = json.load(f)

    # Loss
    history[f"{metric}_loss"],

    # Accuracy
    history[f"{metric}_accuracy"]

    # overfitting gap
    overfitting_gap = np.asarray(history[f"train_accuracy"]) - np.asarray(history[f"val_clean_accuracy"])
