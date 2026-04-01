import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMAGE_DIR = "data/processed/images"
LABEL_FILE = "data/processed/labels.csv"
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 20
 

df = pd.read_csv(LABEL_FILE)

image_paths = df["image_name"].apply(lambda x: os.path.join(IMAGE_DIR, x)).values
labels = df["divergence_score"].values.astype(np.float32)


X_train, X_val, y_train, y_val = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32)  
    return image, label


train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.map(load_image).shuffle(100).batch(BATCH_SIZE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.map(load_image).batch(BATCH_SIZE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),

    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1)  
])


model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


model.summary()


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
model.save("Artery.keras")