import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# -------- Config --------
MODEL_PATH = "D:/VS CODE FILES/TENSORFLOW/Artery Divergence/Artery.keras"
IMAGE_DIR = "data/processed/images"
IMAGE_SIZE = 128


# -------- Load trained model --------
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded:", MODEL_PATH)


# -------- Image loader --------
def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32)  # already 0–1
    image = tf.expand_dims(image, axis=0)  # add batch dimension
    return image


# -------- Pick an image --------
image_name = sorted(os.listdir(IMAGE_DIR))[0]
image_path = os.path.join(IMAGE_DIR, image_name)

image = load_image(image_path)


# -------- Predict --------
prediction = model.predict(image)[0][0]

print(f"Image: {image_name}")
print(f"Predicted divergence score: {prediction:.4f}")


# -------- Visualize --------
plt.imshow(image[0, :, :, 0], cmap="twilight_r")
plt.title(f"Predicted score: {prediction:.3f}")
plt.axis("off")
plt.show()
