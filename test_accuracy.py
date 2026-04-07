import tensorflow as tf
import numpy as np

# Try keras 3
from tensorflow import keras

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)

try:
    model = keras.models.load_model('models/deepfake_detector.h5', compile=False)
    print("Successfully loaded model from .h5")
except Exception as e:
    print("Failed to load .h5:", e)

try:
    model = keras.models.load_model('models/deepfake_detector.keras', compile=False)
    print("Successfully loaded model from .keras")
except Exception as e:
    print("Failed to load .keras:", e)

# Test predict
z1 = np.zeros((1, 224, 224, 3))
z2 = np.ones((1, 224, 224, 3))
print("Prediction on zeros:", model.predict(z1, verbose=0))
print("Prediction on ones:", model.predict(z2, verbose=0))
