import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ CUDA GPU Detected: {gpus[0].name}")
else:
    print("⚠️ CUDA GPU not detected. Training on CPU.")

# Constants
IMG_WIDTH = 220
IMG_HEIGHT = 155

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0  # Normalize
    return img.reshape((IMG_HEIGHT, IMG_WIDTH, 1))

def load_dataset(org_folder, forg_folder):
    X, y = [], []

    # Originals (label = 1)
    for filename in tqdm(os.listdir(org_folder), desc="Originals"):
        if filename.endswith((".png", ".tif")):
            img = preprocess_image(os.path.join(org_folder, filename))
            X.append(img)
            y.append(1)

    # Forgeries (label = 0)
    for filename in tqdm(os.listdir(forg_folder), desc="Forgeries"):
        if filename.endswith((".png", ".tif")):
            img = preprocess_image(os.path.join(forg_folder, filename))
            X.append(img)
            y.append(0)

    return np.array(X), np.array(y)

# Paths
org_path = "D:/Signet/archive/signatures/full_org"
forg_path = "D:/Signet/archive/signatures/full_forg"

# Load data
X, y = load_dataset(org_path, forg_path)

# Shuffle and split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)

# Model Architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('signature_verifier.keras', monitor='val_accuracy', save_best_only=True)

# Train model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# Plot Accuracy and Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# Final evaluation
y_pred = model.predict(X_test).round()
print(classification_report(y_test, y_pred))
