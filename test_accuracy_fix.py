import tensorflow as tf
from tensorflow import keras
import sys

print("Loading h5 now")
try:
    model = keras.models.load_model('models/deepfake_detector.h5', compile=False)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
