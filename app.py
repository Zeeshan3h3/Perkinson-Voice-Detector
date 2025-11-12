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

# ----------------------------
# Load helper functions
# ----------------------------
from mainprogram import extract_features, make_melspectrogram, build_cnn_model, SR, N_MFCC, RF_MODEL_PATH, CNN_MODEL_PATH, TF_AVAILABLE

# ----------------------------
# App setup
# ----------------------------
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# ----------------------------
# FFmpeg setup for pydub
# ----------------------------
import subprocess, shlex

def ensure_wav(input_path):
    """
    Converts any audio file to 16-bit mono WAV using the local ffmpeg binary.
    Returns the path to the converted WAV file.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return input_path

    ffmpeg_exe = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
    ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg", ffmpeg_exe)
    if not os.path.exists(ffmpeg_path):
        raise FileNotFoundError(f"ffmpeg not found at {ffmpeg_path}")

    output_path = input_path + "_converted.wav"
    cmd = f'"{ffmpeg_path}" -y -i "{input_path}" -ac 1 -ar 22050 -f wav "{output_path}"'
    print(f"🔄 Running conversion: {cmd}")
    result = subprocess.run(shlex.split(cmd), capture_output=True)
    if result.returncode != 0:
        print("FFmpeg stderr:", result.stderr.decode(errors='ignore'))
        raise RuntimeError("FFmpeg conversion failed")
    os.remove(input_path)
    return output_path


def ensure_wav(input_path):
    """Force-convert any audio file to WAV using ffmpeg directly."""
    import subprocess, shlex, platform
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return input_path

    ffmpeg_exe = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
    ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg", ffmpeg_exe)
    if not os.path.exists(ffmpeg_path):
        raise FileNotFoundError(f"ffmpeg not found at {ffmpeg_path}")

    output_path = input_path + "_converted.wav"
    cmd = f'"{ffmpeg_path}" -y -i "{input_path}" -ar 22050 -ac 1 "{output_path}"'
    subprocess.run(shlex.split(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(output_path):
        raise RuntimeError("FFmpeg conversion failed")
    os.remove(input_path)
    return output_path



ffmpeg_folder = os.path.join(os.getcwd(), "ffmpeg")
ffmpeg_exe = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
ffmpeg_path = os.path.join(ffmpeg_folder, ffmpeg_exe)
if os.path.exists(ffmpeg_path):
    AudioSegment.converter = ffmpeg_path
    print(f"✅ Using ffmpeg from: {ffmpeg_path}")
else:
    print("⚠️ ffmpeg not found in ./ffmpeg — audio conversion may fail!")

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

    # Step 1 — Save uploaded/recorded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    print(f"📁 Saved upload: {filepath}")

    features = extract_features(filepath, sr=SR, n_mfcc=N_MFCC)
    try:
        filepath = ensure_wav(filepath)
        print(f"✅ File converted and ready: {filepath}")
    except Exception as e:
        print("❌ Conversion failed:", e)
        return jsonify({"error": f"Audio conversion failed: {e}"}), 500

    result = {}

    # Step 3 — Random Forest Prediction
    if rf_model and rf_scaler and rf_features_list:
        try:
            features = extract_features(filepath, sr=SR, n_mfcc=N_MFCC)
            if features is None:
                raise ValueError("Could not extract features from audio.")
            df = pd.DataFrame([features]).reindex(columns=rf_features_list, fill_value=0)
            scaled = rf_scaler.transform(df)
            y_pred = rf_model.predict(scaled)[0]
            y_prob = rf_model.predict_proba(scaled)[0][1]
            result["rf_prediction"] = "Parkinson's" if y_pred == 1 else "Healthy"
            result["rf_probability"] = round(float(y_prob), 4)
            print(f"🧠 RF: {result['rf_prediction']} ({result['rf_probability']})")
        except Exception as e:
            print("RF error:", e)
            traceback.print_exc()
            result["rf_error"] = f"RF Prediction failed: {e}"
    else:
        result["rf_error"] = "Random Forest model not loaded."

    # Step 4 — CNN Prediction (optional)
    if cnn_model and TF_AVAILABLE:
        try:
            y, _ = librosa.load(filepath, sr=SR)
            S = make_melspectrogram(y, sr=SR, n_mels=128)
            S = np.pad(S, ((0, 0), (0, max(0, 128 - S.shape[1]))), mode='constant')
            S = S[:, :128]
            S = (S - np.min(S)) / (np.max(S) - np.min(S) + 1e-8)
            X = S[np.newaxis, ..., np.newaxis].astype('float32')
            p = cnn_model.predict(X)[0][0]
            result["cnn_prediction"] = "Parkinson's" if p > 0.5 else "Healthy"
            result["cnn_probability"] = round(float(p), 4)
        except Exception as e:
            print("CNN error:", e)
            traceback.print_exc()
            result["cnn_error"] = f"CNN Prediction failed: {e}"
    else:
        result["cnn_error"] = "CNN not loaded or TensorFlow unavailable."

    # try:
    #     os.remove(filepath)
    # except Exception:
    #     pass

    return jsonify(result)



if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
