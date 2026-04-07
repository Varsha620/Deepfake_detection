import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras import layers, models

inputs = layers.Input(shape=(224, 224, 3))
base_model = EfficientNetV2B0(weights=None, include_top=False, input_tensor=inputs)
base_model._name = 'efficientnetb0' 
base_model.trainable = False 
x = base_model(inputs, training=False)
if isinstance(x, (list, tuple)): x = x[0]
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs=inputs, outputs=outputs)
try:
    model.load_weights('models/deepfake_detector.h5')
    print("SUCCESS V2B0!")
except Exception as e:
    print("FAILED V2B0:", e)
