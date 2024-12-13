import os
import cv2
import numpy as np
from skimage.feature import hog
import random

# Constants
IMG_SIZE = (48, 48)  # Resize all images to 48x48
EMOTIONS = ["anger", "fear", "happy", "neutral", "sad", "surprise"]  # Emotion categories

# Paths to datasets
PROJECT_ROOT = r"C:\emotion_recognition"
CK_PATH = os.path.join(PROJECT_ROOT, "datasets", "CK_dataset")
JAFFE_PATH = os.path.join(PROJECT_ROOT, "datasets", "JAFFE-[70,30]")

# Function to load and preprocess data
def load_data(dataset_path, augment=False):
    data = []
    labels = []
    for label_idx, emotion in enumerate(EMOTIONS):
        emotion_folder = os.path.join(dataset_path, emotion)
        for file in os.listdir(emotion_folder):
            try:
                # Load the image
                img_path = os.path.join(emotion_folder, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # Resize the image
                img_resized = cv2.resize(img, IMG_SIZE)

                # Optionally augment the image
                if augment:
                    augmented_images = augment_image(img_resized)
                    data.extend(augmented_images)
                    labels.extend([label_idx] * len(augmented_images))
                else:
                    data.append(img_resized)
                    labels.append(label_idx)

            except Exception as e:
                print(f"Error processing file {file}: {e}")

    return np.array(data), np.array(labels)

# Function to augment images
def augment_image(img):
    augmented_images = []

    # Flip the image horizontally
    flipped_img = cv2.flip(img, 1)
    augmented_images.append(flipped_img)

    # Rotate the image by a random angle
    rotated_img = random_rotation(img)
    augmented_images.append(rotated_img)

    # Apply random translation
    translated_img = random_translation(img)
    augmented_images.append(translated_img)

    # Add original image as well
    augmented_images.append(img)

    return augmented_images

# Function to rotate an image by a random angle
def random_rotation(img):
    angle = random.randint(-30, 30)  # Rotate by a random angle between -30 and 30 degrees
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
    return rotated_img

# Function to translate an image randomly
def random_translation(img):
    rows, cols = img.shape
    tx = random.randint(-5, 5)  # Translate in x direction by a random amount
    ty = random.randint(-5, 5)  # Translate in y direction by a random amount
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_img = cv2.warpAffine(img, translation_matrix, (cols, rows))
    return translated_img

# Load and process training data from CK dataset with augmentation
print("Loading and processing CK training data with augmentation.")
ck_train_data, ck_train_labels = load_data(os.path.join(CK_PATH, "train"), augment=True)

# Load and process training data from JAFFE dataset with augmentation
print("Loading and processing JAFFE training data with augmentation.")
jaffe_train_data, jaffe_train_labels = load_data(os.path.join(JAFFE_PATH, "train"), augment=True)

# Function to extract HOG features
def extract_hog_features(images):
    hog_features = []
    for img in images:
        fd, _ = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_features.append(fd)
    return np.array(hog_features)

# Extract HOG features from training data
print("Extracting HOG features from augmented CK training data.")
ck_train_hog = extract_hog_features(ck_train_data)
print("Extracting HOG features from augmented JAFFE training data.")
jaffe_train_hog = extract_hog_features(jaffe_train_data)

# Save processed training data to .npy files in the `datasets` folder
print("Saving processed augmented training data.")
np.save(os.path.join(PROJECT_ROOT, "datasets", "ck_train_labels.npy"), ck_train_labels)
np.save(os.path.join(PROJECT_ROOT, "datasets", "jaffe_train_labels.npy"), jaffe_train_labels)
np.save(os.path.join(PROJECT_ROOT, "datasets", "ck_train_hog.npy"), ck_train_hog)
np.save(os.path.join(PROJECT_ROOT, "datasets", "jaffe_train_hog.npy"), jaffe_train_hog)

print("Training data preprocessing with augmentation complete!")
