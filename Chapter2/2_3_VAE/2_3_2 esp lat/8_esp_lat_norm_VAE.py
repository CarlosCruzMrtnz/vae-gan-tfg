import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import (
    layers,
    models,
    datasets,
    callbacks,
    losses,
    optimizers,
    metrics,
)
from scipy.stats import norm
import os  # Manejo de carpetas

os.makedirs("./VAE/models", exist_ok=True)
os.makedirs("./VAE/logs", exist_ok=True)

IMAGE_SIZE = 32
BATCH_SIZE = 100
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 2
EPOCHS = 3
ALPHA = 500

def display(images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None):
    plt.figure(figsize=size)
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        ax.imshow(images[i].astype(as_type), cmap=cmap)
        ax.axis("off")
    if save_to:
        plt.savefig(save_to, dpi=300)
        print(f"\nSaved to {save_to}")
    plt.show(block=False)

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

def preprocess(imgs):
    imgs = imgs.astype("float32") / 127.5 - 1.0  # Normalizar a [-1, 1] para tanh
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=-1.0)  # padding negro
    imgs = np.expand_dims(imgs, -1)
    return imgs

x_train = preprocess(x_train)
x_test = preprocess(x_test)

display((x_train + 1) / 2.0)  # Mostrar en escala [0,1]

# ========================
# Sampling layer
# ========================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ========================
# Encoder
# ========================
encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name="encoder_input")
x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
x = layers.Conv2D(256, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(512, (3, 3), strides=2, activation="relu", padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]

x = layers.Flatten()(x)
z_mean = layers.Dense(EMBEDDING_DIM, name="z_mean")(x)
z_log_var = layers.Dense(EMBEDDING_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# ========================
# Decoder
# ========================
decoder_input = layers.Input(shape=(EMBEDDING_DIM,), name="decoder_input")
x = layers.Dense(int(np.prod(shape_before_flattening)))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(512, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(256, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
decoder_output = layers.Conv2D(1, (3, 3), strides=1, activation="tanh", padding="same")(x)

decoder = models.Model(decoder_input, decoder_output)
decoder.summary()

# ========================
# VAE model
# ========================
class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data)
            reconstruction_loss = tf.reduce_mean(
                ALPHA * tf.reduce_mean(tf.square(data - reconstruction), axis=(1, 2, 3))
            )
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var, reconstruction = self(data)
        reconstruction_loss = tf.reduce_mean(
            ALPHA * tf.reduce_mean(tf.square(data - reconstruction), axis=(1, 2, 3))
        )
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1)
        )
        total_loss = reconstruction_loss + kl_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

vae = VAE(encoder, decoder)
optimizer = optimizers.Adam(learning_rate=0.0005)
vae.compile(optimizer=optimizer)

# ========================
# Callbacks
# ========================
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath="./VAE/checkpoint_vae_weights.weights.h5",
    save_weights_only=True,
    save_freq="epoch",
    monitor="total_loss",
    mode="min",
    save_best_only=True,
    verbose=1,
)

tensorboard_callback = callbacks.TensorBoard(log_dir="./VAE/logs")

# ========================
# Training
# ========================
vae.fit(
    x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=[model_checkpoint_callback, tensorboard_callback],
)

# ========================
# Saving models
# ========================
vae.save_weights("./VAE/models/vae_weights_final.weights.h5")
encoder.save("./VAE/models/encoder.keras")
decoder.save("./VAE/models/decoder.keras")

# ========================
n_to_predict = 5000
example_images = x_test[:n_to_predict]
example_labels = y_test[:n_to_predict]

z_mean, z_log_var, reconstructions = vae.predict(example_images)
print("Example real clothing items")
display((example_images + 1)/2.0)
print("Reconstructions")
display((reconstructions + 1)/2.0)

z_mean, z_var, z = encoder.predict(example_images)
print(z[:10])

# ========================
figsize = 8
plt.figure(figsize=(figsize, figsize))
plt.scatter(z[:, 0], z[:, 1], c="black", alpha=0.5, s=3)
plt.show(block=False)

# ========================
grid_width, grid_height = (6, 3)
z_sample = np.random.normal(size=(grid_width * grid_height, 2))
reconstructions = decoder.predict(z_sample)
p = norm.cdf(z)
p_sample = norm.cdf(z_sample)

plt.figure(figsize=(figsize, figsize))
plt.scatter(z[:, 0], z[:, 1], c="black", alpha=0.5, s=2)
plt.scatter(z_sample[:, 0], z_sample[:, 1], c="#00B0F0", alpha=1, s=40)
plt.show(block=False)

# ========================
fig = plt.figure(figsize=(figsize, grid_height * 2))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(grid_width * grid_height):
    ax = fig.add_subplot(grid_height, grid_width, i + 1)
    ax.axis("off")
    ax.text(0.5, -0.35, str(np.round(z_sample[i, :], 1)), fontsize=10, ha="center", transform=ax.transAxes)
    ax.imshow((reconstructions[i, :, :] + 1)/2.0, cmap="Greys")
plt.show(block=False)

# ========================
figsize = 8
fig = plt.figure(figsize=(figsize * 2, figsize))
ax = fig.add_subplot(1, 2, 1)
plot_1 = ax.scatter(z[:, 0], z[:, 1], cmap="rainbow", c=example_labels, alpha=0.8, s=3)
plt.colorbar(plot_1)
ax = fig.add_subplot(1, 2, 2)
plot_2 = ax.scatter(p[:, 0], p[:, 1], cmap="rainbow", c=example_labels, alpha=0.8, s=3)
plt.show(block=False)

# ========================
figsize = 12
grid_size = 15
plt.figure(figsize=(figsize, figsize))
plt.scatter(p[:, 0], p[:, 1], cmap="rainbow", c=example_labels, alpha=0.8, s=300)
plt.colorbar()
plt.show(block=False)

x = norm.ppf(np.linspace(0, 1, grid_size))
y = norm.ppf(np.linspace(1, 0, grid_size))
xv, yv = np.meshgrid(x, y)
xv = xv.flatten()
yv = yv.flatten()
grid = np.array(list(zip(xv, yv)))
reconstructions = decoder.predict(grid)

fig = plt.figure(figsize=(figsize, figsize))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(grid_size**2):
    ax = fig.add_subplot(grid_size, grid_size, i + 1)
    ax.axis("off")
    ax.imshow((reconstructions[i, :, :] + 1)/2.0, cmap="Greys")
plt.show(block=True)