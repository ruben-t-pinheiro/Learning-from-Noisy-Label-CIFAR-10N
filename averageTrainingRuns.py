
import json
import numpy as np

from engine import load_data

# Files for each run
plotName = "different learning rates averaged"
folder = "results_learningrates"
output_folder = "results_learningrates_averaged"

SEEDS = [42, 43, 100, 200, 300]
LEARNING_RATES = [0.01, 0.001, 0.0001, 0.00001]
label_sets = load_data()

for noise in label_sets:
    for learning_rate in LEARNING_RATES:
        histories = []

        for seed in SEEDS:
            file_name = f"noise_{noise}_learning_rate_{learning_rate}_seed{seed}_epochs_100_history.json"
            file_path = f"{folder}/{file_name}"

            with open(file_path) as f:
                history = json.load(f)
                histories.append(history)

        averaged_history = {
            "noise": noise,
            "learning_rate": learning_rate,
            "seeds": SEEDS,
        }

        for metric in ["train_loss", "val_noisy_loss", "val_clean_loss", "train_accuracy", "val_noisy_accuracy", "val_clean_accuracy"]:
            values = np.array([h[metric] for h in histories])

            averaged_history[f"mean_{metric}"] = np.mean(values, axis=0).tolist()
            averaged_history[f"std_{metric}"] = np.std(values, axis=0).tolist()


            gaps = np.array([np.asarray(h["train_accuracy"]) - np.asarray(h["val_clean_accuracy"]) for h in histories])

            averaged_history["mean_overfitting_gap"] = np.mean(gaps, axis=0).tolist()
            averaged_history["std_overfitting_gap"] = np.std(gaps, axis=0).tolist()

        output_name = f"noise_{noise}_learning_rate_{learning_rate}_averaged_history.json"
        output_path = f"{output_folder}/{output_name}"

        with open(output_path, "w") as f:
            json.dump(averaged_history, f, indent=4)

        print(f"Saved: {output_path}")
