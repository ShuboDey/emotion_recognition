import numpy as np

# Paths to the saved files
ck_train_hog_path = r"C:\emotion_recognition\datasets\ck_train_hog.npy"
ck_train_labels_path = r"C:\emotion_recognition\datasets\ck_train_labels.npy"
jaffe_train_hog_path = r"C:\emotion_recognition\datasets\jaffe_train_hog.npy"
jaffe_train_labels_path = r"C:\emotion_recognition\datasets\jaffe_train_labels.npy"

# Load the data
ck_train_hog = np.load(ck_train_hog_path)
ck_train_labels = np.load(ck_train_labels_path)
jaffe_train_hog = np.load(jaffe_train_hog_path)
jaffe_train_labels = np.load(jaffe_train_labels_path)

# Display data shapes
print("CK HOG features shape:", ck_train_hog.shape)
print("CK Labels shape:", ck_train_labels.shape)
print("JAFFE HOG features shape:", jaffe_train_hog.shape)
print("JAFFE Labels shape:", jaffe_train_labels.shape)

# Inspect a small sample
print("Sample CK HOG features:", ck_train_hog[:2])  # First 2 HOG feature sets
print("Sample CK Labels:", ck_train_labels[:2])    # Corresponding labels
print("Sample JAFFE HOG features:", jaffe_train_hog[:2])
print("Sample JAFFE Labels:", jaffe_train_labels[:2])
