import joblib

# Paths to the model and scaler
model_path = r"C:\emotion_recognition\models\svm_model.pkl"
scaler_path = r"C:\emotion_recognition\models\scaler.pkl"

# Load the SVM model
svm_model = joblib.load(model_path)
print("SVM Model Summary:")
print(svm_model)

# Load the scaler
scaler = joblib.load(scaler_path)
print("\nScaler Summary:")
print("Mean:", scaler.mean_)
print("Scale:", scaler.scale_)
