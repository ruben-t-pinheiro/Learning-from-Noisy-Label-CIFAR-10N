import json
import matplotlib.pyplot as plt

with open("experiment1_history.json", "r") as f:
    history = json.load(f)

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Loss plot
ax[0].plot(history["train_loss"], label="Train Loss")
ax[0].plot(history["val_noisy_loss"], label="Noisy Validation Loss")
ax[0].plot(history["val_clean_loss"], label="Clean Validation Loss")

ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].set_title("Training and Validation Loss")
ax[0].legend()
ax[0].grid(True)

# Accuracy plot
ax[1].plot(history["train_accuracy"], label="Train Accuracy")
ax[1].plot(history["val_noisy_accuracy"], label="Noisy Validation Accuracy")
ax[1].plot(history["val_clean_accuracy"], label="Clean Validation Accuracy")

ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].set_title("Training and Validation Accuracy")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()