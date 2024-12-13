import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier  # Import the Decision Tree classifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Paths to the saved feature files
ck_hog_path = r"C:\emotion_recognition\datasets\ck_train_hog.npy"
ck_labels_path = r"C:\emotion_recognition\datasets\ck_train_labels.npy"
jaffe_hog_path = r"C:\emotion_recognition\datasets\jaffe_train_hog.npy"
jaffe_labels_path = r"C:\emotion_recognition\datasets\jaffe_train_labels.npy"

# Load the data
ck_hog = np.load(ck_hog_path)
ck_labels = np.load(ck_labels_path)
jaffe_hog = np.load(jaffe_hog_path)
jaffe_labels = np.load(jaffe_labels_path)

# Combine HOG features and labels from both datasets
combined_hog = np.concatenate((ck_hog, jaffe_hog), axis=0)
combined_labels = np.concatenate((ck_labels, jaffe_labels), axis=0)

# Split the combined data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    combined_hog, 
    combined_labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=combined_labels
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save the scaler for future use
scaler_path = r"C:\emotion_recognition\models\scaler.pkl"
joblib.dump(scaler, scaler_path)

# Initialize the Decision Tree model
dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')  # Decision Tree with class balancing

# Hyperparameters for "epoch" simulation (optional)
EPOCHS = 10
BATCH_SIZE = len(X_train_scaled) // EPOCHS  # Split into EPOCHS batches
train_accuracies = []
val_accuracies = []

# Simulate training over "epochs"
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    # Split the data into batches
    batch_start = epoch * BATCH_SIZE
    batch_end = (epoch + 1) * BATCH_SIZE if (epoch + 1) * BATCH_SIZE <= len(X_train_scaled) else len(X_train_scaled)

    # Train on current batch
    X_batch = X_train_scaled[batch_start:batch_end]
    y_batch = y_train[batch_start:batch_end]
    dt.fit(X_batch, y_batch)

    # Calculate accuracy on the full training and validation sets
    train_accuracy = dt.score(X_train_scaled, y_train)
    val_accuracy = dt.score(X_val_scaled, y_val)

    # Store accuracies for plotting
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

# Save the trained model
model_path = r"C:\emotion_recognition\models\decision_tree_model.pkl"
joblib.dump(dt, model_path)

# Plot accuracy graphs
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Training Accuracy', marker='o')
plt.plot(range(1, EPOCHS + 1), val_accuracies, label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(r"C:\emotion_recognition\results\accuracy_graph.png")
plt.show()

# Evaluate the model on the validation set
y_val_pred = dt.predict(X_val_scaled)

# Classification report and confusion matrix
classification_rep = classification_report(y_val, y_val_pred)
conf_matrix = confusion_matrix(y_val, y_val_pred)

# Calculate the accuracy percentage
accuracy_percentage = accuracy_score(y_val, y_val_pred) * 100

# Save the evaluation metrics to a text file
evaluation_metrics_path = r"C:\emotion_recognition\results\evaluation_metrics.txt"
with open(evaluation_metrics_path, 'w') as file:
    file.write("Evaluation Metrics for Decision Tree Classifier\n")
    file.write("--------------------------------------------------\n")
    file.write(f"Accuracy: {accuracy_percentage:.2f}%\n")  # Accuracy percentage
    file.write("Classification Report:\n")
    file.write(classification_rep)
    file.write("\nConfusion Matrix:\n")
    file.write(np.array2string(conf_matrix))

# Print the results to the console
print(f"Validation Accuracy: {accuracy_percentage:.2f}%")
print("Validation Classification Report:")
print(classification_rep)

print("Validation Confusion Matrix:")
print(conf_matrix)
