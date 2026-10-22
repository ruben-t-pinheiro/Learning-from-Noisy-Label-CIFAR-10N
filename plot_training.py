import json
import matplotlib.pyplot as plt

with open("experiment1_history.json", "r") as f:
    history = json.load(f)
    plt.plot(history["train_loss"])
    plt.plot(history["val_noisy_loss"])
    plt.plot(history["val_clean_loss"])
    plt.plot(history["train_accuracy"])
    plt.plot(history["val_noisy_accuracy"])
    plt.plot(history["val_clean_accuracy"])
    plt.legend(["Train Loss", "Noisy Validation Loss", "Clean Validation Loss", "Train Accuracy", "Noisy Validation Accuracy", "Clean Validation Accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.show()