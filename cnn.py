import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ✅ GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPUs detected: {[gpu.name for gpu in gpus]}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ No GPU detected. Training will run on CPU.")

# ✅ Set Paths
BASE_DIR = r'C:\Users\hp5cd\OneDrive\Desktop\signet\dataset\signatures\processed_data'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
IMG_SIZE = (155, 220)
BATCH_SIZE = 32

# ✅ Sanity check on folders
required_dirs = [
    os.path.join(TRAIN_DIR, 'genuine'),
    os.path.join(TRAIN_DIR, 'forged'),
    os.path.join(TEST_DIR, 'genuine'),
    os.path.join(TEST_DIR, 'forged')
]
for path in required_dirs:
    if not os.path.exists(path) or len(os.listdir(path)) == 0:
        raise FileNotFoundError(f"❌ Missing or empty: {path}")

# ✅ Data Augmentation & Generator
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_data = datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ✅ Dataset Check
if train_data.samples == 0 or test_data.samples == 0:
    raise ValueError("🚨 The dataset is empty. Check folder structure and image files.")

# ✅ CNN Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ✅ Train Model
EPOCHS = 15
history = model.fit(train_data, validation_data=test_data, epochs=EPOCHS)

# ✅ Save Model
model.save('signet_model.h5')
print("✅ Model saved as signet_model.h5")

# ✅ Plot Training Curve
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
