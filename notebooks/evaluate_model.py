import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import hog
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
IMG_SIZE = (48, 48)  # Resize all images to 48x48
EMOTIONS = ["anger", "fear", "happy", "neutral", "sad", "surprise"]  # Emotion categories

# Paths to datasets
PROJECT_ROOT = r"C:\emotion_recognition"
CK_PATH = os.path.join(PROJECT_ROOT, "datasets", "CK_dataset")
JAFFE_PATH = os.path.join(PROJECT_ROOT, "datasets", "JAFFE-[70,30]")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "svm_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")

# Function to load and preprocess test data
def load_data(dataset_path):
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
                data.append(img_resized)
                labels.append(label_idx)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    return np.array(data), np.array(labels)

# Function to extract HOG features
def extract_hog_features(images):
    hog_features = []
    for img in images:
        fd, _ = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_features.append(fd)
    return np.array(hog_features)

# Load and preprocess test data from CK and JAFFE datasets
print("Loading and processing CK test data.")
ck_test_data, ck_test_labels = load_data(os.path.join(CK_PATH, "test"))

print("Loading and processing JAFFE test data.")
jaffe_test_data, jaffe_test_labels = load_data(os.path.join(JAFFE_PATH, "test"))

# Combine test data from both datasets
test_data = np.concatenate((ck_test_data, jaffe_test_data), axis=0)
test_labels = np.concatenate((ck_test_labels, jaffe_test_labels), axis=0)

# Extract HOG features from test data
print("Extracting HOG features from test data.")
test_hog = extract_hog_features(test_data)

# Load the trained SVM model and scaler
print("Loading trained SVM model and scaler.")
svm_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Standardize the test data using the scaler
test_hog_scaled = scaler.transform(test_hog)

# Make predictions on the test data
print("Making predictions on the test data.")
predictions = svm_model.predict(test_hog_scaled)

# Evaluate the model
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print("\nClassification Report:")
print(classification_report(test_labels, predictions, target_names=EMOTIONS))

# Confusion matrix
cm = confusion_matrix(test_labels, predictions)
print("\nConfusion Matrix:")
print(cm)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=EMOTIONS, yticklabels=EMOTIONS)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig(os.path.join(PROJECT_ROOT, "results", "confusion_matrix.png"))
plt.show()

# Save evaluation metrics to a text file
with open(os.path.join(PROJECT_ROOT, "results", "SVM_evaluation_metrics.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(test_labels, predictions, target_names=EMOTIONS))
    f.write("\nConfusion Matrix:\n")            
    f.write(str(cm))

print("Model evaluation complete!")
