import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

# Constants
IMG_SIZE = (48, 48)  # Resize all images to 48x48
EMOTIONS = ["anger", "fear", "happy", "neutral", "sad", "surprise"]  # Emotion categories

# Paths to model and scaler
PROJECT_ROOT = r"C:\emotion_recognition"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "svm_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
TEST_IMAGES_PATH = os.path.join(PROJECT_ROOT, "datasets", "JAFFE-[70,30]", "test")

# Load the trained model and scaler
print("Loading model and scaler.")
svm = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Function to preprocess a single image
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image at {img_path} could not be loaded.")
    img_resized = cv2.resize(img, IMG_SIZE)
    fd, _ = hog(img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return img_resized, fd

# Function to visualize predictions
def visualize_predictions(test_images_path, num_images=6):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    axes = axes.flatten()

    images_processed = 0
    for emotion in os.listdir(test_images_path):
        emotion_folder = os.path.join(test_images_path, emotion)
        if not os.path.isdir(emotion_folder):  # Skip non-folders
            continue
        
        for file in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, file)
            if not os.path.isfile(img_path):  # Skip invalid files
                continue

            try:
                print(f"Processing file: {img_path}")  # Debugging statement
                img_resized, fd = preprocess_image(img_path)
                fd_scaled = scaler.transform([fd])
                pred_label = svm.predict(fd_scaled)[0]
                pred_emotion = EMOTIONS[pred_label]
                axes[images_processed].imshow(img_resized, cmap='gray')
                axes[images_processed].set_title(pred_emotion)
                axes[images_processed].axis('off')
                images_processed += 1
                if images_processed >= num_images:
                    break
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        if images_processed >= num_images:
            break

    plt.tight_layout()
    plt.show()

# Call the visualization function
print("Starting visualization.")
visualize_predictions(TEST_IMAGES_PATH)
