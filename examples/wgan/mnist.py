import os
import sys

sys.path.append(os.path.join("..", ".."))

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from wgan import WGANGP, critic_loss, generator_loss
from wgan.critics import DeepConvCritic
from wgan.generators import DeepConvGenerator
from wgan.utils import plot_images, CustomHistory


print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

filepath = os.path.join(os.getcwd(), "datasets", "mnist.npz")

# Download and save MNIST data
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=filepath)
# x_train, y_train, x_test, y_test = (
#     (x_train[..., np.newaxis] / 255.0).astype("float32"),
#     y_train,
#     (x_test[..., np.newaxis] / 255.0).astype("float32"),
#     y_test,
# )

# Load MNIST data from .npz file
data = np.load(filepath)
x_train, y_train, x_test, y_test = (
    (data["x_train"][..., np.newaxis] / 255.0).astype("float32"),
    data["y_train"],
    (data["x_test"][..., np.newaxis] / 255.0).astype("float32"),
    data["y_test"],
)

# Plot MNIST sampled images
# sample = np.random.choice(x_train.shape[0], size=9)
# plot_images(x_train[sample, :, :, :])

# Plot generated images before WGAN training
# generator = DeepConvGenerator()
# noise = tf.random.normal([9, 128])
# generated_images = generator(noise, training=False)
# plot_images(generated_images)

# Training WGAN
latent_dim = 128
lambda_ = 10.0
n_critic = 5
alpha = 0.0001
beta_1, beta_2 = 0.5, 0.9
batch_size = 50
epochs = 1
specialized_class = "all"  # 'all' or 0 to 9

options = ["all"] + [i for i in range(10)]
assert_msg = f"parâmetro specialized_class aceita somente as opções {options}"
assert specialized_class in options, assert_msg

if specialized_class == "all":
    dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)
else:
    class_indexes = np.argwhere(y_train == specialized_class).flatten()
    x_train_specialized = x_train[class_indexes, :, :, :]
    dataset = tf.data.Dataset.from_tensor_slices(x_train_specialized).batch(batch_size)

critic = DeepConvCritic()
generator = DeepConvGenerator()

critic_optimizer = tf.keras.optimizers.Adam(
    learning_rate=alpha,
    beta_1=beta_1,
    beta_2=beta_2,
)
generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=alpha,
    beta_1=beta_1,
    beta_2=beta_2,
)

wgan_gp = WGANGP(
    critic=critic,
    generator=generator,
    latent_dim=latent_dim,
    n_critic=n_critic,
    lambda_=lambda_,
)

wgan_gp.compile(
    critic_optimizer=critic_optimizer,
    generator_optimizer=generator_optimizer,
    critic_loss_fn=critic_loss,
    generator_loss_fn=generator_loss,
)

callbacks = [
    CustomHistory(),
    tf.keras.callbacks.EarlyStopping(monitor="critic_loss", patience=15, mode="max"),
]

wgan_gp.fit(dataset, epochs=epochs, callbacks=callbacks)

# critic.save(
#     f".\\models\\wgangp_critic_model_mnist_epochs={epochs:03d}_class={specialized_class}"
# )
# generator.save(
#     f".\\models\\wgangp_generator_model_mnist_epochs={epochs:03d}_class={specialized_class}"
# )

neg_critic_loss = -np.array(callbacks[0].history["critic_loss"])
batches = neg_critic_loss.shape[0]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(np.arange(batches), neg_critic_loss, alpha=0.4)
ax.plot(np.arange(batches), pd.Series(neg_critic_loss).rolling(100).mean())
ax.set_xlabel("batch iteration")
ax.set_ylabel("negative critic loss")
plt.show()

# # Generate images
# noise = tf.random.normal([9, latent_dim])
# generated_images = generator(noise, training=False)

# plot_images(generated_images)
