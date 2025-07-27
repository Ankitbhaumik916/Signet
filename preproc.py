import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
        if filename.endswith(".png") or filename.endswith(".tif"):
            path = os.path.join(org_folder, filename)
            img = preprocess_image(path)
            X.append(img)
            y.append(1)

    # Forgeries (label = 0)
    for filename in tqdm(os.listdir(forg_folder), desc="Forgeries"):
        if filename.endswith(".png") or filename.endswith(".tif"):
            path = os.path.join(forg_folder, filename)
            img = preprocess_image(path)
            X.append(img)
            y.append(0)

    return np.array(X), np.array(y)

# Paths
org_path = "D:/Signet/archive/signatures/full_forg"
forg_path = "D:/Signet/archive/signatures/full_org"

# Load data
X, y = load_dataset(org_path, forg_path)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(155, 220, 1)),
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('signature_verifier.h5', monitor='val_accuracy', save_best_only=True)

# Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    callbacks=[early_stop, checkpoint]
)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Accuracy Curve")
plt.show()
from sklearn.metrics import classification_report

y_pred = model.predict(X_test).round()
print(classification_report(y_test, y_pred))
