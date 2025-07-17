import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from utils import load_dataset, load_images
from features import extract_features_from_twin

IMAGE_REDUCED_SIZE = 64
IMAGE_DIR = "images"
FOLDERS = ['letters', 'numbers', 'flags', 'animals']

filenames, labels, categories = load_dataset(FOLDERS)
images = load_images(filenames)
images = images.astype(np.float32) / 255.0

assert images.shape[1:] == (64, 64, 3), "Images must be (N, 64, 64, 3)"

with open("users/user_001.json") as f:
    twin = json.load(f)
X, y, _ = extract_features_from_twin(twin, source="history")

assert len(X) > 0 and len(y) > 0, "No features or labels found!"

def build_autoencoder(latent_dim=9):
    input_img = keras.Input(shape=(64, 64, 3))
    x = layers.Flatten()(input_img)
    x = layers.Dense(256, activation='relu')(x)
    encoded = layers.Dense(latent_dim, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(encoded)
    x = layers.Dense(64*64*3, activation='sigmoid')(x)
    decoded = layers.Reshape((64, 64, 3))(x)
    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

autoencoder = build_autoencoder()
autoencoder.fit(images, images, epochs=50, batch_size=16, validation_split=0.1)
autoencoder.save("autoencoder.h5")

def build_mlp_classifier(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

mlp = build_mlp_classifier(X.shape[1])
mlp.fit(X, y, epochs=30, batch_size=8, validation_split=0.1)
mlp.save("classifier.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
tflite_autoencoder = converter.convert()
with open("autoencoder.tflite", "wb") as f:
    f.write(tflite_autoencoder)

converter = tf.lite.TFLiteConverter.from_keras_model(mlp)
tflite_classifier = converter.convert()
with open("classifier.tflite", "wb") as f:
    f.write(tflite_classifier)