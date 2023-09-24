import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from wgan import MLPCritic, MLPGenerator, WGAN, critic_loss, generator_loss

plt.style.use("seaborn-v0_8")

# Load dataset
df = pd.read_csv(".\\datasets\\8gaussians.csv").astype("float32")
print(df.shape)
df.head()

input_dim = 2
latent_dim = 2
lambda_ = 0.1
n_critic = 5
alpha = 1e-4
beta_1, beta_2 = 0.5, 0.9
batch_size = 256
epochs = 5

dataset = tf.data.Dataset.from_tensor_slices(df).batch(batch_size)

critic = MLPCritic(hidden_units=[512, 512, 512])
generator = MLPGenerator(output_dim=2, hidden_units=[512, 512, 512])

critic_optimizer = tf.keras.optimizers.Adam(
    learning_rate=alpha, beta_1=beta_1, beta_2=beta_2
)
generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=alpha, beta_1=beta_1, beta_2=beta_2
)

random_latent_vectors = tf.random.normal((5000, latent_dim))


def generate_and_save_plot(model, batch):
    synthetic_samples = model.generator(random_latent_vectors).numpy()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(synthetic_samples[:, 0], synthetic_samples[:, 1], alpha=0.5)
    ax.scatter(df["feature1"], df["feature2"], alpha=0.05)
    ax.set_xlabel(f"feature1")
    ax.set_ylabel(f"feature2")
    ax.axis("scaled")
    ax.grid(True)

    fig.tight_layout()
    plt.savefig(".\\images\\8gaussians\\data_at_batch_{:04d}.png".format(batch))
    plt.close("all")


class CustomHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.history = {"batch": [], "critic_loss": []}

    def on_batch_end(self, batch, logs={}):
        self.history["batch"].append(batch)
        self.history["critic_loss"].append(logs.get("critic_loss"))

        # if batch % 5 == 0:
        #     generate_and_save_plot(self.model, batch)


# Get the Wasserstein GAN model
wgan = WGAN(
    critic=critic,
    generator=generator,
    latent_dim=latent_dim,
    n_critic=n_critic,
    lambda_=lambda_,
)

# Compile the Wasserstein GAN model
wgan.compile(
    critic_optimizer=critic_optimizer,
    generator_optimizer=generator_optimizer,
    critic_loss_fn=critic_loss,
    generator_loss_fn=generator_loss,
)

callbacks = [CustomHistory()]

wgan.fit(dataset, epochs=epochs, callbacks=callbacks)

critic.save(f".\\models\\critic_8gaussians_{epochs:04d}epochs.keras")
generator.save(f".\\models\\generator_8gaussians_{epochs:04d}epochs.keras")

neg_critic_loss = -np.array(callbacks[0].history["critic_loss"])
batches = neg_critic_loss.shape[0]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(np.arange(batches), neg_critic_loss, alpha=0.4)
ax.plot(np.arange(batches), pd.Series(neg_critic_loss).rolling(100).mean())
ax.set_xlabel("batch iteration")
ax.set_ylabel("negative critic loss")
plt.show()

random_latent_vectors = tf.random.normal((10000, latent_dim))
synthetic_samples = wgan.generator(random_latent_vectors).numpy()

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(synthetic_samples[:, 0], synthetic_samples[:, 1], alpha=0.5, s=7)
ax.scatter(df["feature1"], df["feature2"], alpha=0.05, s=7)
ax.set_xlabel(f"feature1")
ax.set_ylabel(f"feature2")
ax.axis("scaled")
ax.grid(True)

fig.tight_layout()
plt.show()
