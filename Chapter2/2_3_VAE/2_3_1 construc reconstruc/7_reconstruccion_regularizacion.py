import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, optimizers

IMAGE_SIZE = 32
BATCH_SIZE = 100
LATENT_DIM = 2
EPOCHS = 3 
ALPHA = 500

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

def preprocess(imgs):
    imgs = imgs.astype("float32") / 127.5 - 1.0  # normalizamos a [-1,1] para tanh
    imgs = np.pad(imgs, ((0,0),(2,2),(2,2)), constant_values=-1.0)
    imgs = np.expand_dims(imgs, -1)
    return imgs

x_train = preprocess(x_train)
x_test = preprocess(x_test)

# ========================
# Sampling layer
# ========================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ========================
# VAE model
# ========================
def build_vae(use_recon=True, use_kl=True):
    # ---- Encoder ----
    encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(encoder_input)
    x = layers.Conv2D(256, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(512, 3, strides=2, activation="relu", padding="same")(x)
    shape_before_flatten = tf.keras.backend.int_shape(x)[1:]  # (H,W,C)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(LATENT_DIM)(x)
    z_log_var = layers.Dense(LATENT_DIM)(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(encoder_input, [z_mean, z_log_var, z])

    # ---- Decoder ----
    decoder_input = layers.Input(shape=(LATENT_DIM,))
    x = layers.Dense(int(np.prod(shape_before_flatten)), activation="relu")(decoder_input)
    x = layers.Reshape(shape_before_flatten)(x)
    x = layers.Conv2DTranspose(512, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(256, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)
    decoder_output = layers.Conv2D(1, 3, activation="tanh", padding="same")(x)
    decoder = models.Model(decoder_input, decoder_output)

    # ----  VAE  ----
    class VAE(models.Model):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.mse_loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)

                recon_loss = tf.reduce_mean(
                    ALPHA * tf.reduce_mean(tf.square(data - reconstruction), axis=(1,2,3))
                ) if use_recon else 0

                kl_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                        axis=1
                    )
                ) if use_kl else 0

                total_loss = recon_loss + kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            return {"loss": total_loss}

    vae = VAE(encoder, decoder)
    return vae, encoder

experiments = [
    ("MSE only", True, False),
    ("KL divergence only", False, True),
    ("MSE + KL divergence", True, True)
]

latent_spaces = []
titles = []

x_train_full = x_train

for name, use_recon, use_kl in experiments:
    vae, encoder = build_vae(use_recon, use_kl)
    vae.compile(optimizer=optimizers.Adam(learning_rate=0.0005))
    vae.fit(x_train_full, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    z_mean, _, _ = encoder.predict(x_test, batch_size=128)
    latent_spaces.append(z_mean)
    titles.append(name)

for i in range(len(latent_spaces)):
    plt.figure(figsize=(6,6))
    scatter = plt.scatter(latent_spaces[i][:,0],
                          latent_spaces[i][:,1],
                          c=y_test,
                          cmap="rainbow",
                          alpha=0.8,
                          s=3)
    plt.title(titles[i])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(f"latent_space_mse_{i+1}.png", dpi=300)
    plt.show()