import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')            
tf.config.experimental.set_memory_growth(gpus[0], True)  

import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models, metrics, optimizers, losses, callbacks
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import glob
import random


DATA_PATH   = r"CHANGE_ME\celeba\img_align_celeba" 

IMAGE_SIZE    = 64      
IMG_CHANNELS  = 3       
BATCH_SIZE    = 64      
EMBEDDING_DIM = 256      
EPOCHS_VAE    = 50
EPOCHS_GAN    = 200
ALPHA         = 500     
BETA          = 4       

VAL_SPLIT     = 0.1     
MAX_IMAGES    = 150000   

Z_DIM_GAN     = 256     
LEARNING_RATE = 0.0002
ADAM_BETA_1   = 0.5

SAVE_INTERVAL_VAE   = 10
SAMPLE_INTERVAL_VAE = 5
MOSAIC_SIZE         = 4
VISUALIZE_GAN_EVERY = 50

# =========================================================
# Folders
# =========================================================
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Directorio de trabajo: {os.getcwd()}")

os.makedirs("./VAE_GAN/models",      exist_ok=True)
os.makedirs("./VAE_GAN/figures",     exist_ok=True)
os.makedirs("./VAE_GAN/samples",     exist_ok=True)
os.makedirs("./VAE_GAN/mosaics",     exist_ok=True)
os.makedirs("./VAE_GAN/checkpoints", exist_ok=True)
os.makedirs("./VAE_GAN/originals",   exist_ok=True)

# =========================================================
# Aux fuctions
# =========================================================
def save_fig(path, dpi=150, tight=True):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Guardado en {path}")


def display_and_save(images, path, n=10, size=(20, 4)):
    """Guarda una fila de n imágenes RGB en disco."""
    fig = plt.figure(figsize=size)
    for i in range(min(n, len(images))):
        ax = plt.subplot(1, n, i + 1)
        img = images[i]
        if img.shape[-1] == 1:
            ax.imshow(img.squeeze(), cmap="gray")
        else:
            ax.imshow(np.clip(img, 0, 1))
        ax.axis("off")
    save_fig(path)


def save_image_grid(images, path, n=10):
    """Guarda una fila de n imágenes en disco."""
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    for i in range(n):
        img = images[i]
        if img.shape[-1] == 1:
            axes[i].imshow(img.squeeze(), cmap="gray")
        else:
            axes[i].imshow(np.clip(img, 0, 1))
        axes[i].axis("off")
    save_fig(path)


def save_mosaic(decoder, epoch, n=MOSAIC_SIZE, latent_dim=EMBEDDING_DIM):
    """Genera y guarda un mosaico n×n de caras generadas desde el espacio latente."""
    z_sample = np.random.normal(size=(n * n, latent_dim)).astype("float32")
    recons   = decoder.predict(z_sample, verbose=0)

    fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            img = (recons[idx] + 1) / 2.0
            axes[i, j].imshow(np.clip(img, 0, 1))
            axes[i, j].axis("off")
    path = f"./VAE_GAN/mosaics/mosaic_epoch_{epoch:04d}.png"
    save_fig(path)


def save_reconstruction_comparison(vae, images, epoch, n=8):
    """Guarda comparación entre originales y reconstrucciones."""
    _, _, recons = vae.predict(images[:n], verbose=0)
    orig  = (images[:n] + 1) / 2.0
    recon = (recons     + 1) / 2.0

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 5))
    for i in range(n):
        axes[0, i].imshow(np.clip(orig[i],  0, 1)); axes[0, i].axis("off")
        axes[1, i].imshow(np.clip(recon[i], 0, 1)); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Recon",    fontsize=9)
    path = f"./VAE_GAN/samples/reconstruction_epoch_{epoch:04d}.png"
    save_fig(path)


def scatter_latente(ax, z, title):
    """Scatter simple del espacio latente (sin etiquetas de clase en CelebA)."""
    ax.scatter(z[:, 0], z[:, 1], c="steelblue", alpha=0.3, s=2)
    ax.set_title(title)
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")


# =========================================================
# 1. Loading CelebA
# =========================================================
print("\nCargando imágenes de CelebA...")

all_paths = glob.glob(os.path.join(DATA_PATH, "*.jpg"))
random.shuffle(all_paths)

if MAX_IMAGES is not None:
    all_paths = all_paths[:MAX_IMAGES]

print(f"  → Imágenes encontradas: {len(all_paths)}")

n_val   = int(len(all_paths) * VAL_SPLIT)
n_train = len(all_paths) - n_val
train_paths = all_paths[:n_train]
val_paths   = all_paths[n_train:]

print(f"  → Train: {n_train} | Val: {n_val}")


def load_and_preprocess(path):
    """Carga una imagen, hace center crop y normaliza a [-1, 1]."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=IMG_CHANNELS)
    # CelebA tiene caras centradas: crop central antes de resize
    img = tf.image.resize_with_crop_or_pad(img, 178, 178)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.cast(img, tf.float32) / 127.5 - 1.0  # → [-1, 1]
    return img


def make_dataset(paths, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(len(paths))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_dataset = make_dataset(train_paths, shuffle=True)
val_dataset   = make_dataset(val_paths,   shuffle=False)

# Guardar muestra de originales
sample_batch = next(iter(val_dataset)).numpy()
save_image_grid((sample_batch[:10] + 1) / 2.0, "./VAE_GAN/originals/originals.png")
display_and_save((sample_batch[:10] + 1) / 2.0, "./VAE_GAN/figures/00_originales.png")

# =========================================================
# 2. β-VAE: 64×64 RGB
# =========================================================

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim   = tf.shape(z_mean)[1]
        eps   = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

# --- Encoder ---
# 64 → 32 → 16 → 8 → 4  (4 capas de stride 2)
encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, IMG_CHANNELS))
x = layers.Conv2D(64,  3, strides=2, activation="relu", padding="same")(encoder_input)  # 32×32
x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)              # 16×16
x = layers.Conv2D(256, 3, strides=2, activation="relu", padding="same")(x)              # 8×8
x = layers.Conv2D(512, 3, strides=2, activation="relu", padding="same")(x)              # 4×4
shape_before_flat = K.int_shape(x)[1:]
x         = layers.Flatten()(x)
z_mean    = layers.Dense(EMBEDDING_DIM, name="z_mean")(x)
z_log_var = layers.Dense(EMBEDDING_DIM, name="z_log_var")(x)
z         = Sampling()([z_mean, z_log_var])
encoder   = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# --- Decoder ---
decoder_input = layers.Input(shape=(EMBEDDING_DIM,))
x = layers.Dense(int(np.prod(shape_before_flat)))(decoder_input)
x = layers.Reshape(shape_before_flat)(x)
x = layers.Conv2DTranspose(512, 3, strides=2, activation="relu", padding="same")(x)    # 8×8
x = layers.Conv2DTranspose(256, 3, strides=2, activation="relu", padding="same")(x)    # 16×16
x = layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)    # 32×32
x = layers.Conv2DTranspose(64,  3, strides=2, activation="relu", padding="same")(x)    # 64×64
decoder_output = layers.Conv2D(IMG_CHANNELS, 3, strides=1, activation="tanh", padding="same")(x)
decoder = models.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()

# --- VAE Model ---
class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker          = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker             = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return z_mean, z_log_var, self.decoder(z)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data)
            rec_loss = tf.reduce_mean(
                ALPHA * tf.reduce_mean(tf.square(data - reconstruction), axis=(1,2,3))
            )
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(-0.5*(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1)
            )
            total_loss = rec_loss + BETA * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if isinstance(data, tuple): data = data[0]
        z_mean, z_log_var, reconstruction = self(data)
        rec_loss = tf.reduce_mean(
            ALPHA * tf.reduce_mean(tf.square(data - reconstruction), axis=(1,2,3))
        )
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(-0.5*(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1)
        )
        return {"loss": rec_loss + BETA*kl_loss,
                "reconstruction_loss": rec_loss,
                "kl_loss": kl_loss}

# --- Callback VAE ---
class VAECallback(callbacks.Callback):
    def __init__(self, vae, decoder, x_sample):
        super().__init__()
        self.vae      = vae
        self.decoder  = decoder
        self.x_sample = x_sample

    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1
        if ep % SAMPLE_INTERVAL_VAE == 0 or ep == EPOCHS_VAE:
            save_reconstruction_comparison(self.vae, self.x_sample, ep)
            save_mosaic(self.decoder, ep)
        if ep % SAVE_INTERVAL_VAE == 0 or ep == EPOCHS_VAE:
            path = f"./VAE_GAN/checkpoints/vae_weights_epoch_{ep:04d}.weights.h5"
            self.vae.save_weights(path)
            print(f"  Checkpoint VAE guardado en {path}")

# --- Entrenar β-VAE ---
print("\n" + "="*60)
print("  FASE 1: Entrenando β-VAE sobre CelebA")
print("="*60)

vae = VAE(encoder, decoder)
vae.compile(optimizer=optimizers.Adam(0.0005))
history_vae = vae.fit(
    train_dataset,
    epochs=EPOCHS_VAE,
    validation_data=val_dataset,
    callbacks=[VAECallback(vae, decoder, sample_batch[:8])],
)

vae.save_weights("./VAE_GAN/models/vae_weights_final.weights.h5")
encoder.save("./VAE_GAN/models/encoder.keras")
decoder.save("./VAE_GAN/models/decoder.keras")
print("Modelos VAE guardados en ./VAE_GAN/models/")

# =========================================================
# 3. EXTRAER VECTORES LATENTES DEL β-VAE
#    CelebA no tiene etiquetas de clase, usamos z muestreado.
# =========================================================
print("\n" + "="*60)
print("  Extrayendo vectores latentes z (muestreados) del β-VAE...")
print("="*60)

# Extraemos z de los primeros MAX_LATENT batches para no quedarnos sin RAM
MAX_LATENT_BATCHES = 500   # ajusta si tienes poca RAM
z_list = []
for batch in train_dataset.take(MAX_LATENT_BATCHES):
    _, _, z_batch = encoder.predict(batch, verbose=0)
    z_list.append(z_batch)
z_train = np.concatenate(z_list, axis=0)

z_list_val = []
for batch in val_dataset.take(50):
    _, _, z_batch = encoder.predict(batch, verbose=0)
    z_list_val.append(z_batch)
z_test = np.concatenate(z_list_val, axis=0)

print(f"  → z entrenamiento: {z_train.shape}")
print(f"  → z validación:    {z_test.shape}")

# Scatter del espacio latente (solo primeras 2 dimensiones, sin etiquetas de clase)
fig, ax = plt.subplots(figsize=(8, 8))
scatter_latente(ax, z_train[:5000, :2], "Espacio latente β-VAE — CelebA (dim 0 vs dim 1)")
save_fig("./VAE_GAN/figures/01_espacio_latente_vae.png")

# Dataset TensorFlow de vectores latentes
latent_dataset = (
    tf.data.Dataset.from_tensor_slices(z_train.astype("float32"))
    .shuffle(len(z_train))
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

# =========================================================
# 4. GAN EN EL ESPACIO LATENTE
# =========================================================

# --- Discriminador ---
disc_input = layers.Input(shape=(EMBEDDING_DIM,))
x = layers.Dense(256)(disc_input)
x = layers.LeakyReLU(0.2)(x)
#x = layers.Dropout(0.3)(x)
x = layers.Dense(512)(x)
x = layers.LeakyReLU(0.2)(x)
#x = layers.Dropout(0.3)(x)
x = layers.Dense(256)(x)
x = layers.LeakyReLU(0.2)(x)
disc_output = layers.Dense(1, activation="sigmoid")(x)
discriminator_latent = models.Model(disc_input, disc_output, name="discriminador_latente")
discriminator_latent.summary()

# --- Generador ---
gen_input = layers.Input(shape=(Z_DIM_GAN,))
x = layers.Dense(128)(gen_input)
x = layers.LeakyReLU(0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128)(x)
x = layers.LeakyReLU(0.2)(x)
gen_output = layers.Dense(EMBEDDING_DIM)(x)   # Sin activación: z ∈ ℝ
generator_latent = models.Model(gen_input, gen_output, name="generador_latente")
generator_latent.summary()

# --- Latent GAN ---
class LatentGAN(models.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator     = generator
        self.latent_dim    = latent_dim
        self.d_loss_metric = metrics.Mean(name="d_loss")
        self.g_loss_metric = metrics.Mean(name="g_loss")

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.loss_fn     = losses.BinaryCrossentropy()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_z):
        batch_size = tf.shape(real_z)[0]
        noise      = tf.random.normal((batch_size, self.latent_dim))

        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            fake_z   = self.generator(noise, training=True)
            real_out = self.discriminator(real_z, training=True)
            fake_out = self.discriminator(fake_z, training=True)

            real_labels = tf.ones_like(real_out)  * 0.9
            fake_labels = tf.zeros_like(fake_out) + 0.1

            d_loss = (self.loss_fn(real_labels, real_out) +
                      self.loss_fn(fake_labels, fake_out)) / 2
            g_loss = self.loss_fn(tf.ones_like(fake_out), fake_out)

        grads_d = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        grads_g = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads_d, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(grads_g, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {m.name: m.result() for m in self.metrics}

# --- Callback GAN: scatter (sin etiquetas) + grid de caras generadas ---
class LatentVisualizer(callbacks.Callback):
    def __init__(self, real_z, every=VISUALIZE_GAN_EVERY):
        self.real_z = real_z
        self.every  = every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every != 0:
            return
        noise  = tf.random.normal((len(self.real_z), Z_DIM_GAN))
        fake_z = self.model.generator(noise).numpy()

        # Scatter: z reales vs z generados (primeras 2 dimensiones)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        scatter_latente(axes[0], self.real_z[:, :2],
                        f"z reales (β-VAE) — época {epoch+1}")
        axes[1].scatter(fake_z[:, 0], fake_z[:, 1], c="black", alpha=0.4, s=3)
        axes[1].set_title(f"z generados (GAN) — época {epoch+1}")
        axes[1].set_xlabel("z₁"); axes[1].set_ylabel("z₂")
        path = f"./VAE_GAN/figures/latent_gan_epoch_{epoch+1:04d}.png"
        save_fig(path, dpi=120)

        # Grid de caras generadas
        n = 4
        noise_grid = tf.random.normal((n*n, Z_DIM_GAN))
        z_grid     = self.model.generator(noise_grid).numpy()
        imgs_grid  = decoder.predict(z_grid, verbose=0)
        fig, axes  = plt.subplots(n, n, figsize=(n*2, n*2))
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                axes[i, j].imshow(np.clip((imgs_grid[idx] + 1) / 2.0, 0, 1))
                axes[i, j].axis("off")
        path_grid = f"./VAE_GAN/figures/caras_gan_epoch_{epoch+1:04d}.png"
        save_fig(path_grid, dpi=120)

# --- Entrenar Latent GAN ---
print("\n" + "="*60)
print("  FASE 2: Entrenando GAN en el espacio latente")
print("="*60)

latent_gan = LatentGAN(discriminator_latent, generator_latent, Z_DIM_GAN)
latent_gan.compile(
    d_optimizer=optimizers.Adam(LEARNING_RATE, beta_1=ADAM_BETA_1),
    g_optimizer=optimizers.Adam(LEARNING_RATE, beta_1=ADAM_BETA_1),
)

history_gan = latent_gan.fit(
    latent_dataset,
    epochs=EPOCHS_GAN,
    callbacks=[LatentVisualizer(z_test[:2000])],
)

latent_gan.generator.save("./VAE_GAN/models/generator_latente.keras")
latent_gan.discriminator.save("./VAE_GAN/models/discriminator_latente.keras")
print("Modelos GAN guardados en ./VAE_GAN/models/")

# =========================================================
# 5. GENERACIÓN FINAL: ruido → GAN → z sintético → Decoder → cara
# =========================================================
print("\n" + "="*60)
print("  FASE 3: Generación de caras vía pipeline completo")
print("="*60)

n_gen         = 16
noise         = np.random.normal(size=(n_gen, Z_DIM_GAN)).astype("float32")
z_sintetico   = generator_latent.predict(noise)
imgs_hibridas = decoder.predict(z_sintetico)

display_and_save(
    (imgs_hibridas + 1) / 2.0,
    "./VAE_GAN/figures/02_caras_hibridas.png",
    n=n_gen, size=(n_gen * 2, 3)
)

# Grid 8×4
grid_w, grid_h = 8, 4
noise_grid = np.random.normal(size=(grid_w*grid_h, Z_DIM_GAN)).astype("float32")
z_grid     = generator_latent.predict(noise_grid)
imgs_grid  = decoder.predict(z_grid)

fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w*2, grid_h*2))
for i in range(grid_h):
    for j in range(grid_w):
        idx = i * grid_w + j
        axes[i, j].imshow(np.clip((imgs_grid[idx] + 1) / 2.0, 0, 1))
        axes[i, j].axis("off")
plt.suptitle("Caras generadas: ruido → Latent GAN → Decoder β-VAE", fontsize=13)
save_fig("./VAE_GAN/figures/03_grid_caras_hibridas.png")

# =========================================================
# 6. COMPARACIÓN: β-VAE puro vs Híbrido β-VAE+GAN
# =========================================================
print("\n  Comparando muestreo desde N(0,I) vs GAN latente...")

n_cmp = 8

z_prior    = np.random.normal(size=(n_cmp, EMBEDDING_DIM)).astype("float32")
imgs_prior = decoder.predict(z_prior)

noise_cmp  = np.random.normal(size=(n_cmp, Z_DIM_GAN)).astype("float32")
z_gan_cmp  = generator_latent.predict(noise_cmp)
imgs_gan   = decoder.predict(z_gan_cmp)

fig, axes = plt.subplots(2, n_cmp, figsize=(n_cmp*2, 5))
for i in range(n_cmp):
    axes[0, i].imshow(np.clip((imgs_prior[i]+1)/2.0, 0, 1)); axes[0, i].axis("off")
    axes[1, i].imshow(np.clip((imgs_gan[i]  +1)/2.0, 0, 1)); axes[1, i].axis("off")
axes[0, 0].set_ylabel("β-VAE\n(prior N(0,I))", fontsize=10)
axes[1, 0].set_ylabel("Híbrido\n(GAN latente)", fontsize=10)
plt.suptitle("Comparación: muestreo desde prior vs. GAN latente", fontsize=13)
save_fig("./VAE_GAN/figures/04_comparacion_prior_vs_gan.png")

# =========================================================
# 7. SCATTER FINAL: z reales vs z GAN (primeras 2 dimensiones)
# =========================================================
noise_scatter  = np.random.normal(size=(len(z_test), Z_DIM_GAN)).astype("float32")
z_fake_scatter = generator_latent.predict(noise_scatter)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
scatter_latente(axes[0], z_test[:, :2], "z reales muestreados (β-VAE, validación)")
axes[1].scatter(z_fake_scatter[:, 0], z_fake_scatter[:, 1],
                c="steelblue", alpha=0.3, s=4)
axes[1].set_title("z generados (GAN latente)")
axes[1].set_xlabel("z₁"); axes[1].set_ylabel("z₂")
save_fig("./VAE_GAN/figures/05_scatter_real_vs_gan.png")

# =========================================================
# 8. CURVAS DE PÉRDIDA
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(history_vae.history["total_loss"],          label="Total (train)")
axes[0].plot(history_vae.history["val_loss"],            label="Total (val)")
axes[0].plot(history_vae.history["reconstruction_loss"], label="Rec (train)", ls="--")
axes[0].plot(history_vae.history["kl_loss"],             label="KL (train)",  ls=":")
axes[0].set_title("β-VAE — Pérdidas")
axes[0].legend(); axes[0].set_xlabel("Época"); axes[0].grid(True)

axes[1].plot(history_gan.history["d_loss"], label="Discriminador")
axes[1].plot(history_gan.history["g_loss"], label="Generador")
axes[1].set_title("GAN Latente — Pérdidas")
axes[1].legend(); axes[1].set_xlabel("Época"); axes[1].grid(True)

save_fig("./VAE_GAN/figures/06_curvas_perdida.png")

print("\n✓ Experimento completo. Figuras en ./VAE_GAN/figures/")