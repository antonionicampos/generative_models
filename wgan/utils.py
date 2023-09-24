import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import os
import tensorflow as tf


class CustomHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.history = {"batch": [], "critic_loss": []}

    def on_batch_end(self, batch, logs={}):
        self.history["batch"].append(batch)
        self.history["critic_loss"].append(logs.get("critic_loss"))


def plot_images(images):
    fig, axs = plt.subplots(3, 3, figsize=(4, 4))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i, :, :, 0], cmap="gray")
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    plt.show()


def save_images(images, label, folder):
    os.mkdir(f"{folder}\\{label}")
    for sample in np.arange(images.shape[0]):
        matplotlib.image.imsave(
            f"{folder}\\{label}\\{sample+1:04d}.png",
            images[sample, :, :, 0].numpy(),
            cmap="gray",
        )
