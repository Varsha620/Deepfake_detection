import traceback
import tensorflow as tf
from src.model.model_utils import build_model
try:
    model = tf.keras.models.load_model('models/deepfake_detector.keras')
    print('SUCCESS loading .keras')
except Exception as e:
    print('FAILED loading .keras:')
    traceback.print_exc()

try:
    model = tf.keras.models.load_model('models/deepfake_detector.h5')
    print('SUCCESS loading .h5')
except Exception as e:
    print('FAILED loading .h5:')
    traceback.print_exc()
