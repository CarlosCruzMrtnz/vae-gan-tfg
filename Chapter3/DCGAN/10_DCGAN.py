import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import (
    layers,
    models,
    callbacks,
    losses,
    metrics,
    optimizers,
    datasets,
)

IMAGE_SIZE = 28
CHANNELS = 1
BATCH_SIZE = 128
Z_DIM = 100
EPOCHS = 250
LOAD_MODEL = False

ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.999
LEARNING_RATE = 0.0002
NOISE_PARAM = 0.1

# -----------------------------
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data() # Dataset Fashion-MNIST

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# normalización a [-1, 1]
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

# añadir canal
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

train = tf.data.Dataset.from_tensor_slices(x_train)
train = train.shuffle(60000).batch(BATCH_SIZE, drop_remainder=True).prefetch(2)

def sample_batch(dataset):
    return next(iter(dataset)).numpy()


def display(images, n=10, size=(20, 3), cmap="gray_r"):
    images = (images + 1) / 2  # [-1,1] → [0,1]

    plt.figure(figsize=size)
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i].squeeze(), cmap=cmap)
        plt.axis("off")
    plt.show(block=False)


train_sample = sample_batch(train)

# =============================
# DISCRIMINATOR
# =============================
discriminator_input = layers.Input(shape=(28, 28, 1))

x = layers.Conv2D(32, 4, strides=2, padding="same", use_bias=False)(discriminator_input)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)

x = layers.Conv2D(64, 4, strides=2, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)

x = layers.Conv2D(128, 3, strides=2, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(0.2)(x)

x = layers.Flatten()(x)
discriminator_output = layers.Dense(1, activation="sigmoid")(x)

discriminator = models.Model(discriminator_input, discriminator_output)
discriminator.summary()

# =============================
# GENERATOR
# =============================
generator_input = layers.Input(shape=(Z_DIM,))

x = layers.Dense(7 * 7 * 128, use_bias=False)(generator_input)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(0.2)(x)

x = layers.Reshape((7, 7, 128))(x)

x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(0.2)(x)

x = layers.Conv2DTranspose(32, 4, strides=2, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(0.2)(x)

generator_output = layers.Conv2DTranspose(
    1, 3, activation="tanh", padding="same"
)(x)

generator = models.Model(generator_input, generator_output)
generator.summary()

# =============================
# DCGAN
# =============================
class DCGAN(models.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.loss_fn = losses.BinaryCrossentropy()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_metric = metrics.Mean(name="d_loss")
        self.g_loss_metric = metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent = tf.random.normal((batch_size, self.latent_dim))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(random_latent, training=True)

            real_out = self.discriminator(real_images, training=True)
            fake_out = self.discriminator(fake_images, training=True)

            real_labels = tf.ones_like(real_out)
            fake_labels = tf.zeros_like(fake_out)

            d_loss_real = self.loss_fn(real_labels, real_out)
            d_loss_fake = self.loss_fn(fake_labels, fake_out)
            d_loss = (d_loss_real + d_loss_fake) / 2

            g_loss = self.loss_fn(tf.ones_like(fake_out), fake_out)

        grads_d = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        grads_g = gen_tape.gradient(g_loss, self.generator.trainable_variables)

        self.d_optimizer.apply_gradients(
            zip(grads_d, self.discriminator.trainable_variables)
        )
        self.g_optimizer.apply_gradients(
            zip(grads_g, self.generator.trainable_variables)
        )

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}

dcgan = DCGAN(discriminator, generator, Z_DIM)

dcgan.compile(
    d_optimizer=optimizers.Adam(LEARNING_RATE, beta_1=ADAM_BETA_1),
    g_optimizer=optimizers.Adam(LEARNING_RATE, beta_1=ADAM_BETA_1),
)

# =============================
# Callback 
# =============================
class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img=10, latent_dim=Z_DIM):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        z = tf.random.normal((self.num_img, self.latent_dim))
        imgs = self.model.generator(z)
        display(imgs.numpy(), n=self.num_img)

# =============================
# Training
# =============================
dcgan.fit(
    train,
    epochs=EPOCHS,
    callbacks=[ImageGenerator()]
)

# =============================
grid_w, grid_h = (10, 3)
z = np.random.normal(size=(grid_w * grid_h, Z_DIM))

gen_imgs = generator.predict(z)

plt.figure(figsize=(18, 5))
for i in range(grid_w * grid_h):
    plt.subplot(grid_h, grid_w, i + 1)
    plt.imshow(gen_imgs[i].squeeze(), cmap="gray_r")
    plt.axis("off")

plt.show(block=False)

# =============================
def compare_images(img1, img2):
    return np.mean(np.abs(img1 - img2))


all_data = x_train

r, c = 3, 5
gen = generator.predict(np.random.normal(size=(r * c, Z_DIM)))

fig, axs = plt.subplots(r, c, figsize=(10, 6))
cnt = 0

for i in range(r):
    for j in range(c):
        axs[i, j].imshow(gen[cnt].squeeze(), cmap="gray_r")
        axs[i, j].axis("off")
        cnt += 1

plt.suptitle("Generated images")
plt.show(block=False)

fig, axs = plt.subplots(r, c, figsize=(10, 6))
cnt = 0

for i in range(r):
    for j in range(c):
        best = None
        best_d = 1e9

        for k in all_data:
            d = compare_images(gen[cnt], k)
            if d < best_d:
                best_d = d
                best = k

        axs[i, j].imshow(best.squeeze(), cmap="gray_r")
        axs[i, j].axis("off")
        cnt += 1

plt.suptitle("Closest images in dataset")
plt.show(block=True)