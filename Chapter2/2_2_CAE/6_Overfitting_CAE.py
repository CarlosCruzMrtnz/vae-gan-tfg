import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, datasets, callbacks
import tensorflow.keras.backend as K

os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

IMAGE_SIZE = 32
CHANNELS = 1
BATCH_SIZE = 100
EMBEDDING_DIM = 2
EPOCHS = 100

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

def preprocess(imgs):
    imgs = imgs.astype("float32") / 255.0
    imgs = np.pad(imgs, ((0,0),(2,2),(2,2)), constant_values=0.0)
    imgs = np.expand_dims(imgs, -1)
    return imgs

x_train = preprocess(x_train)
x_test = preprocess(x_test)

# -------------------------
# Encoder conv
# -------------------------
encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name="encoder_input")
x = layers.Conv2D(128, (3,3), strides=2, activation="relu", padding="same")(encoder_input)
x = layers.Conv2D(256, (3,3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(512, (3,3), strides=2, activation="relu", padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]

x = layers.Flatten()(x)
encoder_output = layers.Dense(EMBEDDING_DIM, name="encoder_output")(x)

encoder = models.Model(encoder_input, encoder_output)
encoder.summary()

# -------------------------
# Decoder conv
# -------------------------
decoder_input = layers.Input(shape=(EMBEDDING_DIM,), name="decoder_input")
x = layers.Dense(int(np.prod(shape_before_flattening)))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(512, (3,3), strides=2, activation="relu", padding="same")(x)  
x = layers.Conv2DTranspose(256, (3,3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(128, (3,3), strides=2, activation="relu", padding="same")(x)
decoder_output = layers.Conv2D(CHANNELS, (3,3), strides=1, activation="tanh", padding="same", name="decoder_output")(x)

decoder = models.Model(decoder_input, decoder_output)
decoder.summary()

# -------------------------
# Autoencoder
# -------------------------
autoencoder = models.Model(encoder_input, decoder(encoder_output))
autoencoder.summary()

autoencoder.compile(optimizer="adam", loss="mse")

# -------------------------
# Callbacks
# -------------------------
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath="./models/checkpoint.keras",
    save_weights_only=False,
    save_freq="epoch",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1,
)
tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")

# -------------------------
# Training
# -------------------------
history = autoencoder.fit(
    x_train,
    x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=[model_checkpoint_callback, tensorboard_callback],
)

# -------------------------
# Saving models
# -------------------------
autoencoder.save("./models/autoencoder.keras")
encoder.save("./models/encoder.keras")
decoder.save("./models/decoder.keras")

# -------------------------
n_to_predict = 10
example_images = x_test[:n_to_predict]
predictions = autoencoder.predict(example_images)

print("Ejemplos reales:")
display(example_images)
print("Reconstrucciones:")
display(predictions)

# -------------------------
# Loss vs Epochs: overfitting
# -------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Loss Entrenamiento')
plt.plot(history.history['val_loss'], label='Loss Validación')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Overfitting')
plt.legend()
plt.show()