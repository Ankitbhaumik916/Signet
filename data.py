import os
import shutil
import random

# ✅ Paths
GENUINE_DIR = r'C:\Users\hp5cd\OneDrive\Desktop\signet\dataset\signatures\full_org'
FORGED_DIR = r'C:\Users\hp5cd\OneDrive\Desktop\signet\dataset\signatures\full_forg'
OUTPUT_DIR = r'C:\Users\hp5cd\OneDrive\Desktop\signet\dataset\signatures\processed_data'

TRAIN_SPLIT = 0.8

def organize_signet_dataset():
    # ✅ Create necessary directories
    for split in ['train', 'test']:
        for label in ['genuine', 'forged']:
            os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)

    # ✅ Process genuine signatures
    genuine_files = [f for f in os.listdir(GENUINE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(genuine_files)
    split_index = int(TRAIN_SPLIT * len(genuine_files))

    for i, file in enumerate(genuine_files):
        dest = 'train' if i < split_index else 'test'
        shutil.copy(
            os.path.join(GENUINE_DIR, file),
            os.path.join(OUTPUT_DIR, dest, 'genuine', file)
        )

    # ✅ Process forged signatures (also assuming they're .png like you said)
    forged_files = [f for f in os.listdir(FORGED_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(forged_files)
    split_index = int(TRAIN_SPLIT * len(forged_files))

    for i, file in enumerate(forged_files):
        dest = 'train' if i < split_index else 'test'
        shutil.copy(
            os.path.join(FORGED_DIR, file),
            os.path.join(OUTPUT_DIR, dest, 'forged', file)
        )

    print("✅ Dataset has been organized into train/test sets.")

organize_signet_dataset()
