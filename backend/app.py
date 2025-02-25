from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import librosa
import torch
import os
from werkzeug.utils import secure_filename
import random

#Initialize flask app
app = Flask(__name__)
CORS(app) #Enabling CORS to allow request from front end

#Load the trained scaler, PCA and model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("music_classifier.pkl", "rb") as f:
    state_dict = pickle.load(f)

# Load the trained LabelEncoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print ("label_encoder classes: ", le.classes_)
#Define the class model and make this was the original class used for training
class MusicClassifier(torch.nn.Module):

    def __init__(self, input_size, num_classes):
        super(MusicClassifier, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, 256)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.3)

        self.fc2 = torch.nn.Linear(256, 128)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.3)

        self.fc3 = torch.nn.Linear(128, 64)
        self.relu3 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(0.3)

        self.fc4 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return x

#Initialize the model and load the weights
input_size = pca.n_components_
num_classes = 10
model = MusicClassifier(input_size, num_classes)
model.load_state_dict(state_dict)
model.eval() #Set model to evaluation mode

# Function to extract features from an audio file (10 to 25 seconds)
def extract_audio_features(audio_path, min_duration=10, max_duration=25):
    duration = random.uniform(min_duration, max_duration)
    y, sr = librosa.load(audio_path, sr=None, duration=duration)

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    harmony, perceptr = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features = {
        "length": len(y),
        "chroma_stft_mean": np.mean(chroma_stft),
        "chroma_stft_var": np.var(chroma_stft),
        "rms_mean": np.mean(rms),
        "rms_var": np.var(rms),
        "spectral_centroid_mean": np.mean(spectral_centroid),
        "spectral_centroid_var": np.var(spectral_centroid),
        "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
        "spectral_bandwidth_var": np.var(spectral_bandwidth),
        "rolloff_mean": np.mean(rolloff),
        "rolloff_var": np.var(rolloff),
        "zero_crossing_rate_mean": np.mean(zero_crossing_rate),
        "zero_crossing_rate_var": np.var(zero_crossing_rate),
        "harmony_mean": np.mean(harmony),
        "harmony_var": np.var(harmony),
        "perceptr_mean": np.mean(perceptr),
        "perceptr_var": np.var(perceptr),
        "tempo": tempo
    }

    # Extract MFCCs
    for i in range(20):
        features[f"mfcc{i+1}_mean"] = np.mean(mfccs[i])
        features[f"mfcc{i+1}_var"] = np.var(mfccs[i])

    return pd.DataFrame([features])

# Function to predict genre
def predict_genre(audio_path):
    features = extract_audio_features(audio_path)

    # Apply scaling and PCA
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    # Convert to tensor for PyTorch model
    input_tensor = torch.tensor(features_pca, dtype=torch.float32)

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_label_index = predicted.item()
    print("LabelEncoder classes:", le.classes_)
    print ("predicted label index: ", predicted_label_index)
    print ("Label classes ", le.classes_[predicted_label_index])
    predicted_genre = le.classes_[predicted_label_index]

    return predicted_genre

# Route for file upload and genre prediction
@app.route("/predict", methods=["POST"])
def upload_file():
    print ("Request Received")
    print ("Request file: ", request.files)
    
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    print("Received file:", file.filename)
    
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join("uploads", filename)
    
    # Save the file temporarily
    os.makedirs("uploads", exist_ok=True)  # Ensure upload folder exists
    file.save(file_path)
    
    try:
        # Predict genre
        genre_prediction = predict_genre(file_path)
        print("Predicted Genre:", genre_prediction)
        return jsonify({"predicted_genre": genre_prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(file_path)  # Remove file after processing

if __name__ == '__main__':
    app.run(debug=True)