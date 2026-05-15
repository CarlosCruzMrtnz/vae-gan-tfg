import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, datasets, callbacks
import os

IMAGE_SIZE = 28  
BATCH_SIZE = 100
EMBEDDING_DIM = 2
EPOCHS = 3

def display(images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None):
    """
    Muestra n imágenes de un array de imágenes.
    """
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    plt.figure(figsize=size)
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].astype(as_type), cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show(block=False)  

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

def preprocess_dense(imgs):
    imgs = imgs.astype("float32") / 255.0
    imgs = imgs.reshape((imgs.shape[0], -1))
    return imgs

x_train_flat = preprocess_dense(x_train)
x_test_flat  = preprocess_dense(x_test)

print("Forma de x_train_flat:", x_train_flat.shape)

# -------------------------
# Encoder fully connected
# -------------------------
encoder_input = layers.Input(shape=(IMAGE_SIZE*IMAGE_SIZE,), name="encoder_input")
x = layers.Dense(512, activation="relu")(encoder_input)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
encoder_output = layers.Dense(EMBEDDING_DIM, name="encoder_output")(x)
encoder = models.Model(encoder_input, encoder_output)
encoder.summary()

# -------------------------
# Decoder fully connected
# -------------------------
decoder_input = layers.Input(shape=(EMBEDDING_DIM,), name="decoder_input")
x = layers.Dense(128, activation="relu")(decoder_input)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(512, activation="relu")(x)
decoder_output = layers.Dense(IMAGE_SIZE*IMAGE_SIZE, activation="relu", name="decoder_output")(x)
decoder = models.Model(decoder_input, decoder_output)
decoder.summary()

# -------------------------
# Autoencoder
# -------------------------
autoencoder_input = encoder_input
autoencoder_output = decoder(encoder_output)
autoencoder = models.Model(autoencoder_input, autoencoder_output)
#autoencoder.summary()

autoencoder.compile(optimizer="adam", loss="mse")

# -------------------------
# Callbacks
# -------------------------
os.makedirs("./relu/checkpoint", exist_ok=True)
os.makedirs("./relu/logs", exist_ok=True)
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath="./relu/checkpoint/checkpoint.keras",
    save_weights_only=False,
    save_freq="epoch",
    monitor="loss",
    mode="min",
    save_best_only=True,
    verbose=0,
)
tensorboard_callback = callbacks.TensorBoard(log_dir="./relu/logs")

# -------------------------
# Training
# -------------------------
autoencoder.fit(
    x_train_flat,
    x_train_flat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(x_test_flat, x_test_flat),
    callbacks=[model_checkpoint_callback, tensorboard_callback],
)

# -------------------------
# Saving models
# -------------------------
os.makedirs("./relu/models", exist_ok=True)
autoencoder.save("./relu/models/autoencoder_dense.keras")
encoder.save("./relu/models/encoder_dense.keras")
decoder.save("./relu/models/decoder_dense.keras")

# -------------------------
# Predictions
# -------------------------
n_to_predict = 5000
example_images = x_test_flat[:n_to_predict]
example_labels = y_test[:n_to_predict]
predictions = autoencoder.predict(example_images)

display(x_test[:n_to_predict], n=10, size=(20,3))  # Originales
display(predictions.reshape((-1, IMAGE_SIZE, IMAGE_SIZE)), n=10, size=(20,3))  # Reconstrucciones

# -------------------------
# Encode
# -------------------------
embeddings = encoder.predict(example_images)

# -------------------------
plt.figure(figsize=(8,8))
plt.scatter(embeddings[:,0], embeddings[:,1], c="black", alpha=0.5, s=3)
plt.title("Embeddings en 2D")
plt.show(block=False)

# -------------------------
plt.figure(figsize=(8,8))
plt.scatter(embeddings[:,0], embeddings[:,1], c=example_labels, cmap="rainbow", alpha=0.8, s=3)
plt.colorbar(label="Clase de prenda")
plt.title("Embeddings coloreados por etiqueta")
plt.show(block=False)

# -------------------------
mins, maxs = np.min(embeddings, axis=0), np.max(embeddings, axis=0)
grid_width, grid_height = (6, 3)
sample = np.random.uniform(mins, maxs, size=(grid_width*grid_height, EMBEDDING_DIM))
reconstructions = decoder.predict(sample).reshape((-1, IMAGE_SIZE, IMAGE_SIZE))

plt.figure(figsize=(8,8))
plt.scatter(embeddings[:,0], embeddings[:,1], c="black", alpha=0.5, s=2)
plt.scatter(sample[:,0], sample[:,1], c="#00B0F0", alpha=1, s=40)
plt.title("Embeddings y puntos muestreados")
plt.show(block=False)

# -------------------------
fig = plt.figure(figsize=(12,6))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(grid_width*grid_height):
    ax = fig.add_subplot(grid_height, grid_width, i+1)
    ax.axis("off")
    ax.text(0.5, -0.35, str(np.round(sample[i,:],1)), fontsize=10, ha="center", transform=ax.transAxes)
    ax.imshow(reconstructions[i], cmap="Greys")
plt.suptitle("Reconstrucciones de puntos muestreados")
plt.show(block=False)

# -------------------------
grid_size = 15
plt.figure(figsize=(12,12))
plt.scatter(embeddings[:,0], embeddings[:,1], c=example_labels, cmap="rainbow", alpha=0.8, s=300)
plt.colorbar(label="Clase de prenda")
plt.title("Embeddings coloreados con grid")
plt.show(block=False)

# -------------------------
grid = np.stack(np.meshgrid(np.linspace(mins[0], maxs[0], grid_size),
                            np.linspace(maxs[1], mins[1], grid_size)), axis=-1).reshape(-1,2)
reconstructions = decoder.predict(grid).reshape((-1, IMAGE_SIZE, IMAGE_SIZE))

fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(grid_size**2):
    ax = fig.add_subplot(grid_size, grid_size, i+1)
    ax.axis("off")
    ax.imshow(reconstructions[i], cmap="Greys")
plt.suptitle("Grid de imágenes decodificadas grandes")
plt.show()  # <-- Mantiene todas las figuras abiertas al final