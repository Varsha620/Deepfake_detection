import os
import tensorflow as tf
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from model_utils import build_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5  # Reduced for demonstration; set to 20+ for production
LOCAL_DATASET_PATH = 'data/cifake'

def plot_metrics(history, test_generator, model):
    """
    Generates and saves plots for Training History, Confusion Matrix, and ROC Curve.
    """
    os.makedirs('reports', exist_ok=True)
    
    # 1. Training History
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig('reports/training_history.png')
    
    # 2. Confusion Matrix & ROC
    print("\nEvaluating on test set...")
    Y_pred = model.predict(test_generator)
    y_pred = (Y_pred > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('reports/confusion_matrix.png')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, Y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('reports/roc_curve.png')
    
    print("\n" + "="*30)
    print("CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
    print(f"Final AUC Score: {roc_auc:.4f}")
    print("="*30)
    print("Reports saved in /reports folder.")

def train():
    # 1. Setup Data Path
    if os.path.exists(LOCAL_DATASET_PATH):
        dataset_path = LOCAL_DATASET_PATH
    else:
        print("Dataset not found locally. Downloading via kagglehub...")
        dataset_path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")

    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    
    # 2. Data Generators
    # Added augmentation for better generalization
    datagen = ImageDataGenerator(
        rescale=1./255, 
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    
    train_gen = datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', subset='training')
    val_gen = datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', subset='validation')
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

    # 3. Model Build
    model = build_model()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/deepfake_detector_checkpoint.keras', 
            save_best_only=True, 
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    ]

    # Phase 1: Training the Head (Frozen Base)
    print("\n--- Phase 1: Training the Classification Head ---")
    # STEPS_PER_EPOCH = min(100, train_gen.samples // BATCH_SIZE) 
    # VAL_STEPS = min(20, val_gen.samples // BATCH_SIZE)
    
    history_p1 = model.fit(
        train_gen, 
        # steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_gen, 
        # validation_steps=VAL_STEPS,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Phase 2: Fine-Tuning (Unfreeze Top Layers)
    print("\n--- Phase 2: Fine-Tuning Top Layers ---")
    
    # Properly find the base model layer in Functional API
    base_model = None
    for layer in model.layers:
        if 'efficientnet' in layer.name.lower():
            base_model = layer
            break
            
    if base_model is None:
        print("Warning: Could not find base model layer for fine-tuning.")
        # Fallback to indexing if search fails
        base_model = model.layers[1] 

    base_model.trainable = True
    
    # Freeze all layers except the last 20
    fine_tune_at = 20 # Unfreeze the last 20 layers of EfficientNet
    for layer in base_model.layers[:-fine_tune_at]:
        layer.trainable = False

    # Recompile with a MUCH smaller learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    history_p2 = model.fit(
        train_gen,
        # steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_gen,
        # validation_steps=VAL_STEPS,
        epochs=EPOCHS, # Run for 5 more epochs
        callbacks=callbacks
    )

    # 4. Evaluation & Visualization
    plot_metrics(history_p2, test_gen, model)

    # 5. Save Model
    os.makedirs('models', exist_ok=True)
    model.save('models/deepfake_detector.keras')
    # Maintain .h5 for legacy app support if needed
    model.save('models/deepfake_detector.h5')
    print("Optimized model saved in .keras and .h5 formats.")

if __name__ == "__main__":
    train()
