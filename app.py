import os
import platform
import traceback
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
import subprocess, shlex

# ----------------------------
# Import your helper functions and constants
# ----------------------------
from mainprogram import (
    extract_features, make_melspectrogram,
    build_cnn_model, SR, N_MFCC,
    RF_MODEL_PATH, CNN_MODEL_PATH, TF_AVAILABLE
)

# ----------------------------
# Flask app setup
# ----------------------------
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ----------------------------
# FFmpeg setup for pydub
# ----------------------------
ffmpeg_folder = os.path.join(os.getcwd(), "ffmpeg")
ffmpeg_exe = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
ffmpeg_path = os.path.join(ffmpeg_folder, ffmpeg_exe)
if os.path.exists(ffmpeg_path):
    AudioSegment.converter = ffmpeg_path
    print(f"✅ Using ffmpeg from: {ffmpeg_path}")
else:
    print("⚠️ ffmpeg not found in ./ffmpeg — audio conversion may fail!")


def ensure_wav(input_path):
    """Convert any audio to WAV mono 22050Hz using ffmpeg."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return input_path

    if not os.path.exists(ffmpeg_path):
        raise FileNotFoundError(f"ffmpeg not found at {ffmpeg_path}")

    output_path = input_path + "_converted.wav"
    cmd = f'"{ffmpeg_path}" -y -i "{input_path}" -ac 1 -ar 22050 "{output_path}"'
    subprocess.run(shlex.split(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(output_path):
        raise RuntimeError("FFmpeg conversion failed")
    os.remove(input_path)
    return output_path


# ----------------------------
# Load models
# ----------------------------
rf_model = None
rf_scaler = None
rf_features_list = None
cnn_model = None

try:
    rf_data = joblib.load(RF_MODEL_PATH)
    rf_model = rf_data['model']
    rf_scaler = rf_data['scaler']
    rf_features_list = rf_data['features']
    print(f"✅ Random Forest model loaded from {RF_MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading RF model: {e}")

if TF_AVAILABLE:
    try:
        dummy_shape = (128, 128, 1)
        cnn_model = build_cnn_model(dummy_shape)
        cnn_model.load_weights(CNN_MODEL_PATH)
        print(f"✅ CNN model loaded from {CNN_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ CNN model not loaded: {e}")
else:
    print("⚠️ TensorFlow not available — CNN skipped.")


# ----------------------------
# Routes
# ----------------------------
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

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    print(f"📁 Saved upload: {filepath}")

    # Convert to WAV if needed
    try:
        filepath = ensure_wav(filepath)
        print(f"✅ File converted and ready: {filepath}")
    except Exception as e:
        print("❌ Conversion failed:", e)
        return jsonify({
            "RF_Error": "Audio conversion failed",
            "CNN1_Error": "Audio conversion failed",
            "CNN2_Error": "Audio conversion failed"
        }), 500

    result = {}

    # ----------------------------
    # Random Forest Prediction
    # ----------------------------
    if rf_model and rf_scaler and rf_features_list:
        try:
            features = extract_features(filepath, sr=SR, n_mfcc=N_MFCC)
            if features is None:
                raise ValueError("Could not extract features from audio.")

            df = pd.DataFrame([features]).reindex(columns=rf_features_list, fill_value=0)
            scaled = rf_scaler.transform(df)
            y_pred = int(rf_model.predict(scaled)[0])
            y_prob = float(rf_model.predict_proba(scaled)[0][1])
            result["RF_Pred"] = y_pred
            result["RF_Prob"] = y_prob
        except Exception as e:
            print("❌ RF Prediction error:", e)
            traceback.print_exc()
            result["RF_Error"] = "Analysis Failed"
    else:
        result["RF_Error"] = "Random Forest model not loaded."

    # ----------------------------
    # CNN Prediction
    # ----------------------------
    if cnn_model and TF_AVAILABLE:
        try:
            y, _ = librosa.load(filepath, sr=SR)
            S = make_melspectrogram(y, sr=SR, n_mels=128)
            S = np.pad(S, ((0, 0), (0, max(0, 128 - S.shape[1]))), mode='constant')
            S = S[:, :128]
            S = (S - np.min(S)) / (np.max(S) - np.min(S) + 1e-8)
            X = S[np.newaxis, ..., np.newaxis].astype('float32')
            p = float(cnn_model.predict(X)[0][0])
            result["CNN1_Pred"] = 1 if p > 0.5 else 0
            result["CNN1_Prob"] = p
        except Exception as e:
            print("❌ CNN1 Prediction error:", e)
            traceback.print_exc()
            result["CNN1_Error"] = "Analysis Failed"
    else:
        result["CNN1_Error"] = "CNN not loaded or TensorFlow unavailable."

    # ----------------------------
    # CNN2 (Extra) - duplicate CNN1 if no second model
    # ----------------------------
    if "CNN1_Pred" in result and "CNN1_Prob" in result:
        result["CNN2_Pred"] = result["CNN1_Pred"]
        result["CNN2_Prob"] = result["CNN1_Prob"]
    else:
        result["CNN2_Error"] = "Analysis Failed"

    # Optional: clean up uploaded file
    try:
        os.remove(filepath)
    except Exception:
        pass

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
