import json
import matplotlib.pyplot as plt
import numpy as np

def plot_label_Accuracy(plotName, label_sets, learning_rates, seeds, max_epochs):
    experiments = len(label_sets)* len(learning_rates)* len(seeds)
    experiment = 0

    # Different colors for each run
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    if experiments > len(colors):
            print("to manny runs, add more collers or reduse label_sets or learning_rates")
            return
    
    fig, ax = plt.subplots(3, 1, figsize=(12, 9))

    for noise in label_sets:
        for learning_rate in learning_rates: 
            for seed in seeds:
                experiment_name = f"noise_{noise}_learning_rate_{learning_rate}_seed{seed}_epochs_{max_epochs}"
                experiment_file = f"results/{experiment_name}_history.json"

                with open(experiment_file, "r") as f:
                    history = json.load(f)

                epochs = range(1, max_epochs+1)

                ax[0].plot(
                    epochs,
                    history["correctly_labeled_accuracy"],
                    color=colors[experiment],
                    linestyle="-",
                    label=f"{noise}, lr={learning_rate}"
                )

                ax[1].plot(
                    epochs,
                    history["incorrectly_labeled_accuracy"],
                    color=colors[experiment],
                    linestyle="-",
                    label=f"{noise}, lr={learning_rate}"
                )

                ax[2].plot(
                    epochs,
                    history["incorrectly_labeled_clean_accuracy"],
                    color=colors[experiment],
                    linestyle="-",
                    label=f"{noise}, lr={learning_rate}"
                )

                experiment += 1

    # Loss plot
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Correctly labeled")
    ax[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax[0].grid(True, alpha=0.3)

    # Accuracy plot
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Incorrectly labeled")
    ax[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax[1].grid(True, alpha=0.3)

    # overfitting plot
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Accuracy")
    ax[2].set_title("Incorrectly labeled with clean lables")
    ax[2].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"plots/{plotName}")
    plt.show()

def plot_label_Accuracy_one_plot(plotName, label_sets, learning_rates, seeds, max_epochs):
    experiments = len(label_sets)* len(learning_rates)* len(seeds)
    experiment = 0

    # Different colors for each run
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    styles = {
    "correctly_labeled": "-",
    "incorrectly_labeled": "--",
    "incorrectly_labeled_clean": ":",
    }

    if experiments > len(colors):
            print("to manny runs, add more collers or reduse label_sets or learning_rates")
            return
    
    fig, ax = plt.subplots(figsize=(12, 6))

    for noise in label_sets:
        for learning_rate in learning_rates: 
            for seed in seeds:
                experiment_name = f"noise_{noise}_learning_rate_{learning_rate}_seed{seed}_epochs_{max_epochs}"
                experiment_file = f"results/{experiment_name}_history.json"

                with open(experiment_file, "r") as f:
                    history = json.load(f)

                epochs = range(1, max_epochs+1)

                for metric, style in styles.items():
                    ax.plot(
                        epochs,
                        history[f"{metric}_accuracy"],
                        color=colors[experiment],
                        linestyle=style,
                        label=f"{metric}, {noise}, lr={learning_rate}"
                    )

                experiment += 1

    # Loss plot
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy depending on label correctness")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"plots/{plotName}_one_plot")
    plt.show()


if __name__ == "__main__":
    plotName = "Accuracy based on if the label is correct"
    #label_sets = ["clean", "aggre", "random1", "random2", "random3", "worst"]
    label_sets = ["random2"]
    learning_rates = [0.001]
    seeds = [42, 100, 200, 300]
    epochs = 30
    plot_label_Accuracy(plotName, label_sets, learning_rates, seeds, epochs)
    plot_label_Accuracy_one_plot(plotName, label_sets, learning_rates, seeds, epochs)