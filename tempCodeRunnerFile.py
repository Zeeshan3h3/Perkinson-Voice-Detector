# app.py
import os
from flask import Flask, request, jsonify, render_template
import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import traceback

from flask_cors import CORS



app = Flask(__name__)
CORS(app)
# from pydub import AudioSegment
# AudioSegment.from_file("input.m4a").export("output.wav", format="wav")

# Assuming your original script is in 'your_script.py'
from mainprogram import extract_features, make_melspectrogram, build_cnn_model, SR, N_MFCC, RF_MODEL_PATH, CNN_MODEL_PATH, TF_AVAILABLE

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max upload size

# --- Load Models Once ---
rf_model = None
rf_scaler = None
rf_features_list = None
cnn_model = None

try:
    rf_data = joblib.load(RF_MODEL_PATH)
    rf_model = rf_data['model']
    rf_scaler = rf_data['scaler']
    rf_features_list = rf_data['features']
    print(f"Random Forest model loaded from {RF_MODEL_PATH}")
except Exception as e:
    print(f"Error loading RF model: {e}")
    print("Please ensure you've run your_script.py to train and save the model.")

if TF_AVAILABLE:
    try:
        # For CNN, we need a dummy input shape to rebuild the model structure
        # A more robust way would be to save/load the model with architecture
        # or have a known input shape if it's fixed.
        # Assuming your generate_spectrogram_dataset outputs (128, 128, 1)
        # If your CNN takes a different shape, adjust this
        dummy_input_shape = (128, 128, 1)
        cnn_model = build_cnn_model(dummy_input_shape) # Rebuild architecture
        cnn_model.load_weights(CNN_MODEL_PATH) # Load weights
        print(f"CNN model loaded from {CNN_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading CNN model: {e}")
        print("Please ensure you've run your_script.py with run_cnn=True to train and save the CNN model.")
else:
    print("TensorFlow not available, CNN inference will be skipped.")

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        result = {}
        # --- Random Forest Prediction ---
        if rf_model and rf_scaler and rf_features_list:
            try:
                features = extract_features(filepath, sr=SR, n_mfcc=N_MFCC)
                if features is None:
                    raise ValueError("Could not extract features from audio.")

                # Ensure features are in the same order as trained model
                feature_df = pd.DataFrame([features])
                # Reindex to match the order of features used during training
                feature_df = feature_df.reindex(columns=rf_features_list, fill_value=0)

                scaled_features = rf_scaler.transform(feature_df)
                rf_prediction = rf_model.predict(scaled_features)[0]
                rf_probability = rf_model.predict_proba(scaled_features)[0][1] # Probability of class 1
                result['rf_prediction'] = "Parkinson's" if rf_prediction == 1 else "Healthy"
                result['rf_probability'] = round(float(rf_probability), 4)
            except Exception as e:
                result['rf_error'] = f"RF Prediction failed: {e}"
                print("RF error:")
                traceback.print_exc()
                
        else:
            result['rf_error'] = "Random Forest model not loaded."

        # --- CNN Prediction ---
        if cnn_model and TF_AVAILABLE:
            try:
                y_audio, _ = librosa.load(filepath, sr=SR)
                S_db = make_melspectrogram(y_audio, sr=SR, n_mels=128)
                # Ensure consistent size for CNN input
                out_size_cnn = (128, 128) # Must match how your CNN was trained
                if S_db.shape[1] < out_size_cnn[1]:
                    pad_width = out_size_cnn[1] - S_db.shape[1]
                    S_db = np.pad(S_db, ((0,0),(0,pad_width)), mode='constant')
                else:
                    S_db = S_db[:, :out_size_cnn[1]]
                S_norm = (S_db - np.min(S_db)) / (np.max(S_db) - np.min(S_db) + 1e-8)
                cnn_input = S_norm[np.newaxis, ..., np.newaxis].astype('float32') # Add batch and channel dimensions

                cnn_probability = cnn_model.predict(cnn_input)[0][0]
                cnn_prediction = 1 if cnn_probability > 0.5 else 0
                result['cnn_prediction'] = "Parkinson's" if cnn_prediction == 1 else "Healthy"
                result['cnn_probability'] = round(float(cnn_probability), 4)
            except Exception as e:
                result['cnn_error'] = f"CNN Prediction failed: {e}"
                print("CNN error:")
                traceback.print_exc()
        else:
            result['cnn_error'] = "CNN model not loaded or TensorFlow not available."

        # Clean up the uploaded file
        os.remove(filepath)
        return jsonify(result)

    return jsonify({"error": "Unexpected error"}), 500


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True) # Set debug=False for production