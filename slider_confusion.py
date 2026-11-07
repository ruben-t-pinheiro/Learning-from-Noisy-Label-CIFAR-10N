import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

noise = "random2"
learning_rate = 0.001
seed = 42
EPOCHS = 100
experiment_name =  f"noise_{noise}_learning_rate_{learning_rate}_seed{seed}_epochs_{EPOCHS}"
FOLDER = "results_learningrates"
PATH = f"{FOLDER}/{experiment_name}_history.json"


with open(PATH, "r") as f:
    cms = np.array(json.load(f)["val_clean_cm"], dtype=float)   

print(f"Shape: {cms.shape}")  # [epochs, 10, 10]

# Row-normalize each epoch's matrix: P(predicted | true)
cms_norm = cms / cms.sum(axis=2, keepdims=True)
n_epochs = len(cms_norm)

# --- Figure with room at the bottom for the slider ---
fig, ax = plt.subplots(figsize=(6.5, 6))
fig.subplots_adjust(bottom=0.18)

im = ax.imshow(cms_norm[0], vmin=0, vmax=1, cmap="viridis")
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
title = ax.set_title(f"Epoch 1 / {n_epochs} - Validation Confusion Matrix")
fig.colorbar(im, ax=ax, label="Normalized count")

# --- Slider (native matplotlib widget, works in any window) ---
slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.04])
slider = Slider(slider_ax, "Epoch", 1, n_epochs, valinit=1, valstep=1)


def update(val):
    epoch = int(slider.val) - 1
    im.set_data(cms_norm[epoch])
    title.set_text(f"Epoch {epoch + 1} / {n_epochs} - Validation Confusion Matrix")
    fig.canvas.draw_idle()


slider.on_changed(update)


# Bonus: left/right arrow keys also scrub
def on_key(event):
    if event.key == "right":
        slider.set_val(min(slider.val + 1, n_epochs))
    elif event.key == "left":
        slider.set_val(max(slider.val - 1, 1))


fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()