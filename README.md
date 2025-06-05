# Signet: Signature Verification using CNN

Signet is a forensic deep learning project that verifies whether a given signature is **genuine** or **forged** using a Convolutional Neural Network (CNN). This project uses the CEDAR Signature dataset and prepares it for training/testing, builds a CNN classifier, and saves the final model for deployment.

---

## 📁 Project Structure

```
signet/
├── dataset/
│   └── signatures/
│       ├── full_org/          # Raw genuine signatures (.png)
│       ├── full_forg/         # Raw forged signatures (.png)
│       └── processed_data/    # Organized train/test data
│           ├── train/
│           │   ├── genuine/
│           │   └── forged/
│           └── test/
│               ├── genuine/
│               └── forged/
├── data.py                    # Script to organize dataset
├── cnn_model.py               # CNN training & saving code
├── signet_model.h5            # Final trained model
└── README.md                  # Project documentation
```

---

## 🧠 CNN Model Overview

* **Input**: Preprocessed grayscale signature image (resized, normalized)
* **Architecture**:

  * Convolutional layers
  * Max pooling
  * Dropout for regularization
  * Fully connected (dense) layers
  * Sigmoid activation for binary classification

---

## 🛠 How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/signet.git
cd signet
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Make sure you have the CEDAR signature dataset placed in:

```
dataset/signatures/full_org
```

```
dataset/signatures/full_forg
```

Then run:

```bash
python data.py
```

### 4. Train the Model

```bash
python cnn_model.py
```

The trained model will be saved as `signet_model.h5`

---

## 📱 Future Plans

* Create a **desktop or mobile app** for signature verification
* Add user interface for uploading and predicting signatures
* Possibly integrate with forensic or legal software for real-world use

---

## 👨‍💻 Author

**Ankit Bhaumik**
B.Tech in Artificial Intelligence, 2nd Year

---
