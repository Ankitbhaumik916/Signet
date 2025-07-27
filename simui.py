import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("signature_verifier.h5")

# Image preprocessing function
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (155, 220))
    img = img.astype('float32') / 255.0
    return img.reshape(220, 155, 1)

# UI class
class SignatureVerifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signature Verifier")

        self.ref_path = None
        self.test_path = None

        tk.Button(root, text="Upload Reference Signature", command=self.load_reference).pack(pady=5)
        self.ref_label = tk.Label(root)
        self.ref_label.pack()

        tk.Button(root, text="Upload Test Signature", command=self.load_test).pack(pady=5)
        self.test_label = tk.Label(root)
        self.test_label.pack()

        tk.Button(root, text="Verify Signature", command=self.verify_signature, bg="green", fg="white").pack(pady=10)
        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack()

    def load_reference(self):
        path = filedialog.askopenfilename()
        if path:
            self.ref_path = path
            self.display_image(path, self.ref_label)

    def load_test(self):
        path = filedialog.askopenfilename()
        if path:
            self.test_path = path
            self.display_image(path, self.test_label)

    def display_image(self, path, label):
        img = Image.open(path).resize((155, 110))
        img = ImageTk.PhotoImage(img)
        label.config(image=img)
        label.image = img

    def verify_signature(self):
        if not self.ref_path or not self.test_path:
            messagebox.showwarning("Missing Image", "Please upload both reference and test signatures.")
            return

        ref_img = preprocess_image(self.ref_path)
        test_img = preprocess_image(self.test_path)

        # Your model expects [ref, test] as two inputs
        pred = model.predict([np.expand_dims(ref_img, 0), np.expand_dims(test_img, 0)])
        similarity = pred[0][0]

        if similarity > 0.5:
            self.result_label.config(text=f"Genuine Signature ✅\n(Similarity: {similarity:.2f})", fg="green")
        else:
            self.result_label.config(text=f"Forged Signature ❌\n(Similarity: {similarity:.2f})", fg="red")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = SignatureVerifierApp(root)
    root.mainloop()
