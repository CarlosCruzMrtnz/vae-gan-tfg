import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models, datasets, callbacks, losses, optimizers, metrics
from scipy.stats import norm
import os

os.makedirs("./VAE/models", exist_ok=True)
os.makedirs("./VAE/logs", exist_ok=True)

IMAGE_SIZE = 32
BATCH_SIZE = 100
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 2
EPOCHS = 100
ALPHA = 500  
BETA = 4     

def display(images, n=10, size=(20,3), cmap="gray_r", as_type="float32", save_to=None):
    plt.figure(figsize=size)
    for i in range(n):
        ax = plt.subplot(1, n, i+1)
        ax.imshow(images[i].astype(as_type), cmap=cmap)
        ax.axis("off")
    if save_to:
        plt.savefig(save_to, dpi=300)
        print(f"\nSaved to {save_to}")
    plt.show(block=False)

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

def preprocess(imgs):
    imgs = imgs.astype("float32") / 127.5 - 1.0  # Normalizar [-1,1]
    imgs = np.pad(imgs, ((0,0),(2,2),(2,2)), constant_values=-1.0)
    imgs = np.expand_dims(imgs, -1)
    return imgs

x_train = preprocess(x_train)
x_test = preprocess(x_test)

display((x_train + 1)/2.0)  # Mostrar ejemplos originales

# -------------------------
# Sampling layer
# -------------------------
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# -------------------------
# Encoder
# -------------------------
encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
x = layers.Conv2D(128, (3,3), strides=2, activation="relu", padding="same")(encoder_input)
x = layers.Conv2D(256, (3,3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(512, (3,3), strides=2, activation="relu", padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]

x = layers.Flatten()(x)
z_mean = layers.Dense(EMBEDDING_DIM, name="z_mean")(x)
z_log_var = layers.Dense(EMBEDDING_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# -------------------------
# Decoder
# -------------------------
decoder_input = layers.Input(shape=(EMBEDDING_DIM,))
x = layers.Dense(int(np.prod(shape_before_flattening)))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(512, (3,3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(256, (3,3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(128, (3,3), strides=2, activation="relu", padding="same")(x)
decoder_output = layers.Conv2D(1, (3,3), strides=1, activation="tanh", padding="same")(x)

decoder = models.Model(decoder_input, decoder_output)
decoder.summary()

# -------------------------
# VAE model
# -------------------------
class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data)
            reconstruction_loss = tf.reduce_mean(
                ALPHA * tf.reduce_mean(tf.square(data - reconstruction), axis=(1,2,3))
            )
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(-0.5*(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1)
            )
            total_loss = reconstruction_loss + BETA * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if isinstance(data, tuple): data = data[0]
        z_mean, z_log_var, reconstruction = self(data)
        reconstruction_loss = tf.reduce_mean(
            ALPHA * tf.reduce_mean(tf.square(data - reconstruction), axis=(1,2,3))
        )
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(-0.5*(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1)
        )
        total_loss = reconstruction_loss + BETA * kl_loss
        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}

vae = VAE(encoder, decoder)
vae.compile(optimizer=optimizers.Adam(0.0005))

history = vae.fit(
    x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(x_test, x_test),
)

# -------------------------
# Saving models
# -------------------------
vae.save_weights("./VAE/models/vae_weights_final.weights.h5")
encoder.save("./VAE/models/encoder.keras")
decoder.save("./VAE/models/decoder.keras")

# -------------------------
n_to_predict = 5000
example_images = x_test[:n_to_predict]
example_labels = y_test[:n_to_predict]

z_mean, z_log_var, reconstructions = vae.predict(example_images)
display((example_images + 1)/2.0)  # Originales
display((reconstructions + 1)/2.0)  # Reconstrucciones

# -------------------------
z_mean, z_log_var, z = encoder.predict(example_images)
plt.figure(figsize=(8,8))
plt.scatter(z[:,0], z[:,1], c="black", alpha=0.5, s=3)
plt.show(block=False)

# -------------------------
grid_width, grid_height = (6,3)
z_sample = np.random.normal(size=(grid_width*grid_height, 2))
reconstructions = decoder.predict(z_sample)
plt.figure(figsize=(grid_width*3, grid_height*3))
for i in range(grid_width*grid_height):
    ax = plt.subplot(grid_height, grid_width, i+1)
    ax.imshow((reconstructions[i]+1)/2.0, cmap="Greys")
    ax.axis("off")
plt.show(block=False)

# -------------------------
plt.figure(figsize=(16,8))
plt.scatter(z[:,0], z[:,1], c=example_labels, cmap="rainbow", alpha=0.8, s=3)
plt.colorbar()
plt.show(block=False)

# -------------------------
grid_size = 15
x = norm.ppf(np.linspace(0,1,grid_size))
y = norm.ppf(np.linspace(1,0,grid_size))
xv, yv = np.meshgrid(x, y)
grid = np.column_stack([xv.flatten(), yv.flatten()])
reconstructions = decoder.predict(grid)
plt.figure(figsize=(12,12))
for i in range(grid_size**2):
    ax = plt.subplot(grid_size, grid_size, i+1)
    ax.imshow((reconstructions[i]+1)/2.0, cmap="Greys")
    ax.axis("off")
plt.show(block=False)

# -------------------------
# Loss vs Epochs: overfitting
# -------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['total_loss'], label='Loss Entrenamiento')
plt.plot(history.history['val_loss'], label='Loss Validación')
plt.plot(history.history['reconstruction_loss'], label='Reconstrucción Entrenamiento', linestyle='--')
plt.plot(history.history['val_reconstruction_loss'], label='Reconstrucción Validación', linestyle='--')
plt.plot(history.history['kl_loss'], label='KL Entrenamiento', linestyle=':')
plt.plot(history.history['val_kl_loss'], label='KL Validación', linestyle=':')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Overfitting')
plt.legend()
plt.show(block=True)