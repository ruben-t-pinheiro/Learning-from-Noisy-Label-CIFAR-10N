import json
import matplotlib.pyplot as plt

with open("demo_aggre_history.json", "r") as f:
    history = json.load(f)
    plt.plot(history["train_loss"])
    plt.plot(history["val_noisy_loss"])
    plt.plot(history["val_clean_loss"])
    plt.plot(history["train_accuracy"])
    plt.plot(history["val_noisy_accuracy"])
    plt.plot(history["val_clean_accuracy"])
    plt.legend(["Train", "Noisy Validation Loss", "Clean Validation Loss", "Train Accuracy", "Noisy Validation Accuracy", "Clean Validation Accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.show()

    # plot confusion matrices for the last epoch
    val_noisy_cm = history["val_noisy_cm"]
    val_clean_cm = history["val_clean_cm"]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(val_noisy_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Noisy Validation Confusion Matrix")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(val_clean_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Clean Validation Confusion Matrix")
    plt.colorbar()
    plt.show()