import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

def build_model(input_shape=(224, 224, 3), num_classes=1):
    """
    Builds a Deepfake Detection model using EfficientNetB0 as the backbone.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Revert to vanilla backbone to ensure 1:1 weight matching
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model._name = 'efficientnetb0' 
    base_model.trainable = False 

    x = base_model(inputs, training=False)
    
    # Defensive unwrapping for Keras 3 tuple bug
    if isinstance(x, (list, tuple)):
        x = x[0]
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_image(img_path, target_size=(224, 224)):
    """
    Loads and preprocesses an image for model prediction.
    """
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    return preprocess_input(img_array)
