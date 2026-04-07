import tensorflow as tf
from src.model.model_utils import build_model

keras_path = 'models/deepfake_detector.keras'
h5_path = 'models/deepfake_detector.h5'

model = build_model()

try:
    print("Trying load_weights WITHOUT by_name (h5)...")
    model.load_weights(h5_path)
    print("SUCCESS full topological match!")
    # Test predict
    import numpy as np
    dummy_input = np.zeros((1, 224, 224, 3))
    out = model.predict(dummy_input)
    print("Test predict output:", out)
except Exception as e:
    import traceback
    traceback.print_exc()

print("="*40)
try:
    print("Trying load_weights WITHOUT by_name (keras)...")
    model.load_weights(keras_path)
    print("SUCCESS full topological match!")
    out = model.predict(dummy_input)
    print("Test predict output:", out)
except Exception as e:
    import traceback
    traceback.print_exc()
