"""
Prototype: Parkinson's Disease detection from voice (single-file)

Usage:
  - Put your audio files into a folder and create a CSV `metadata.csv` with columns:
      filename,label
    where `label` is 1 for Parkinson's, 0 for healthy control (or use text labels).
  - Install dependencies (recommended in virtualenv):
      pip install -r requirements.txt

This prototype implements:
  - audio loading
  - feature extraction (MFCCs, pitch, jitter, shimmer approximation, spectral features)
  - a classical ML baseline (RandomForest) with evaluation
  - a simple CNN that trains on mel-spectrograms (small model for demonstration)
  - saving trained models

Notes:
  - This is a prototype for experimentation and education. For clinical use, rigorous
    data collection, validation, IRB approval and regulatory compliance are required.
  - Paths and hyperparameters are intentionally conservative for small datasets.

requirements (example) -> requirements.txt:
  numpy
  pandas
  librosa
  scikit-learn
  matplotlib
  soundfile
  tensorflow
  joblib

"""

import os
import math
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier





# Optional: TensorFlow for a small CNN
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


SR = 22050  # sampling rate
N_MFCC = 13
AUDIO_DIR = "./audio"  # folder containing wav files
METADATA_CSV = "metadata.csv"  # must contain `filename,label` columns
RF_MODEL_PATH = "rf_parkinson_model.joblib"
CNN_MODEL_PATH = "cnn_parkinson_model.h5"



def load_audio(path, sr=SR, duration=None, offset=0.0):
    y, _ = librosa.load(path, sr=sr, duration=duration, offset=offset)
    return y


def compute_pitch(y, sr=SR):
    # Use librosa.yin for robust F0 estimation
    try:
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        # remove unvoiced frames (yin returns nan for unvoiced)
        f0 = f0[~np.isnan(f0)]
        if len(f0) == 0:
            return 0.0, 0.0
        return float(np.mean(f0)), float(np.std(f0))
    except Exception:
        return 0.0, 0.0


def compute_jitter(f0_vals):
    # Simple jitter approximation: mean absolute difference between consecutive F0s normalized
    if len(f0_vals) < 2:
        return 0.0
    diffs = np.abs(np.diff(f0_vals))
    if np.mean(f0_vals) == 0:
        return 0.0
    return float(np.mean(diffs) / (np.mean(f0_vals) + 1e-8))


def compute_shimmer(amplitude_envelope):
    # Simple shimmer approximation: mean absolute difference between consecutive amplitudes normalized
    if len(amplitude_envelope) < 2:
        return 0.0
    diffs = np.abs(np.diff(amplitude_envelope))
    mean_amp = np.mean(amplitude_envelope) + 1e-8
    return float(np.mean(diffs) / mean_amp)


def extract_features(path, sr=SR, n_mfcc=N_MFCC):
    y = load_audio(path, sr=sr)
    if len(y) == 0:
        return None

    # Basic features
    features = {}

    # MFCCs (mean + std)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features.update({f"mfcc_{i}_mean": float(np.mean(mfcc[i])) for i in range(n_mfcc)})
    features.update({f"mfcc_{i}_std": float(np.std(mfcc[i])) for i in range(n_mfcc)})

    # Delta MFCCs
    delta = librosa.feature.delta(mfcc)
    features.update({f"d_mfcc_{i}_mean": float(np.mean(delta[i])) for i in range(n_mfcc)})

    # Spectral features
    features["spectral_centroid_mean"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features["spectral_rolloff_mean"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features["zcr_mean"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # Pitch / F0
    try:
        f0_full = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        f0_voiced = f0_full[~np.isnan(f0_full)]
    except Exception:
        f0_voiced = np.array([])

    if len(f0_voiced) > 0:
        features["f0_mean"] = float(np.mean(f0_voiced))
        features["f0_std"] = float(np.std(f0_voiced))
        features["jitter"] = compute_jitter(f0_voiced)
    else:
        features["f0_mean"] = 0.0
        features["f0_std"] = 0.0
        features["jitter"] = 0.0

    # Amplitude envelope -> shimmer
    hop_length = 512
    frame_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size > 0:
        features["rms_mean"] = float(np.mean(rms))
        features["shimmer"] = compute_shimmer(rms)
    else:
        features["rms_mean"] = 0.0
        features["shimmer"] = 0.0

    # Harmonics-to-noise ratio proxy using harmonic-percussive separation
    try:
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        h_energy = np.sum(np.abs(y_harmonic)) + 1e-8
        p_energy = np.sum(np.abs(y_percussive)) + 1e-8
        features["hnr_proxy"] = float(10 * math.log10(h_energy / p_energy))
    except Exception:
        features["hnr_proxy"] = 0.0

    return features

# -----------------------------
# Dataset utilities
# -----------------------------

def build_feature_dataframe(metadata_csv=METADATA_CSV, audio_dir=AUDIO_DIR):
    df_meta = pd.read_csv(metadata_csv)
    rows = []
    

    for idx, row in df_meta.iterrows():
        fname = str(row['filename'])
        label = row['label']
        path = os.path.join(audio_dir, fname)
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue
        feats = extract_features(path)
        if feats is None:
            print(f"Could not extract from: {path}")
            continue
        feats['filename'] = fname
        feats['label'] = label
        rows.append(feats)
    df = pd.DataFrame(rows)
    return df

# -----------------------------
# Classical ML baseline
# -----------------------------

def train_random_forest(df, target_col='label'):
    X = df.drop(columns=['filename', target_col])
    y = df[target_col].astype(int)

    # Handle missing / infinite
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(Xs, y, stratify=y, test_size=0.5, random_state=42)


    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None

    print("Classification report (RandomForest):")
    print(classification_report(y_test, y_pred))
    if y_proba is not None:
        try:
            print("ROC AUC:", roc_auc_score(y_test, y_proba))
        except Exception:
            pass

    # Save model + scaler
    joblib.dump({'model': clf, 'scaler': scaler, 'features': list(X.columns)}, RF_MODEL_PATH)
    print(f"Saved RandomForest model to {RF_MODEL_PATH}")
    return clf, scaler

# -----------------------------
# Simple CNN on Mel-spectrograms
# -----------------------------

def make_melspectrogram(y, sr=SR, n_mels=128, hop_length=512, n_fft=2048):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def generate_spectrogram_dataset(metadata_csv=METADATA_CSV, audio_dir=AUDIO_DIR, out_size=(128,128)):
    df_meta = pd.read_csv(metadata_csv)
    X = []
    y = []
    for idx, row in df_meta.iterrows():
        fname = str(row['filename'])
        label = int(row['label'])
        path = os.path.join(audio_dir, fname)
        if not os.path.exists(path):
            continue
        y_audio = load_audio(path, sr=SR)
        S_db = make_melspectrogram(y_audio, sr=SR, n_mels=out_size[0])
        # Resize/crop or pad to out_size[1]
        if S_db.shape[1] < out_size[1]:
            pad_width = out_size[1] - S_db.shape[1]
            S_db = np.pad(S_db, ((0,0),(0,pad_width)), mode='constant')
        else:
            S_db = S_db[:, :out_size[1]]
        # Normalize
        S_norm = (S_db - np.min(S_db)) / (np.max(S_db) - np.min(S_db) + 1e-8)
        X.append(S_norm)
        y.append(label)
    X = np.array(X)[..., np.newaxis].astype('float32')
    y = np.array(y).astype('int')
    return X, y


def build_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(32, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------
# Main: run end-to-end prototype
# -----------------------------

def main(run_cnn=False):
    if not os.path.exists(METADATA_CSV):
        print(f"Metadata CSV not found at {METADATA_CSV}. Create a CSV with columns: filename,label")
        return

    print("Extracting features for classical ML baseline...")
    df = build_feature_dataframe(METADATA_CSV, AUDIO_DIR)
    if df.shape[0] < 10:
        print("Warning: small dataset. Results may be unreliable. Collected rows:", df.shape[0])

    clf, scaler = train_random_forest(df, target_col='label')

    if run_cnn:
        if not TF_AVAILABLE:
            print("TensorFlow not available - skipping CNN stage.")
            return
        print("Preparing spectrogram dataset for CNN...")
        X, y = generate_spectrogram_dataset(METADATA_CSV, AUDIO_DIR, out_size=(128,128))
        if X.shape[0] < 8:
            print("Too few spectrograms to train a CNN. Need more audio files.")
            return
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        model = build_cnn_model(input_shape=X_train.shape[1:])
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        model.fit(X_train, y_train, epochs=30, batch_size=8, validation_split=0.15, callbacks=callbacks)
        loss, acc = model.evaluate(X_test, y_test)
        print(f"CNN test loss: {loss:.4f}, accuracy: {acc:.4f}")
        model.save(CNN_MODEL_PATH)
        print(f"Saved CNN model to {CNN_MODEL_PATH}")
        
def train_from_csv(csv_path='pd_speech_features.csv'):
    """
    Loads features directly from the pd_speech_features CSV, scales them, 
    and trains a Random Forest model.
    """
def train_from_csv(csv_path='pd_speech_features.csv'):
        # ... (File loading and initial setup)
    df = pd.read_csv(csv_path)

    # ... (Dropping ID column)
    if df.columns[0].lower() in ['id', 'sl_no']:
        df = df.drop(df.columns[0], axis=1, errors='ignore')

    # FIX: Filter out any classes with a single sample (assuming they are errors)
    # The PD dataset should only have '0' (Healthy) and '1' (Parkinson's)
    target_col = df.columns[-1]
    
    # 1. Calculate the size of each class
    class_sizes = df[target_col].value_counts()
    
    # 2. Identify classes to remove (size < 2)
    single_member_classes = class_sizes[class_sizes < 2].index
    
    # 3. Filter the DataFrame
    df = df[~df[target_col].isin(single_member_classes)]
    
    if len(single_member_classes) > 0:
        print(f"Removed {len(single_member_classes)} rows belonging to class(es) with only one member.")
        print(f"Remaining classes: {df[target_col].value_counts()}")
    # END FIX
    
    # Features (X) are all columns except the last one (the 'class' label).
    X = df.iloc[:, :-1]
    y = df[target_col]

    # ... (Rest of the function: train_test_split, scaling, training)
    # 1. Split Data (using a 75/25 split, 80/20 is also common)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # 2. Scale Features (Essential for many ML models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train Model (Random Forest)
    print("\nTraining Random Forest Classifier...")
    # 'class_weight'='balanced' helps with potentially imbalanced PD/Healthy groups
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)

    # 4. Evaluate Model
    y_pred = clf.predict(X_test_scaled)
    print("\n--- Model Performance on Test Set ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, clf.predict_proba(X_test_scaled)[:, 1]):.4f}")

    # 5. Save the trained model and scaler
    joblib.dump(clf, 'rf_pd_model_csv.pkl')
    joblib.dump(scaler, 'rf_pd_scaler_csv.pkl')
    print("Model and Scaler saved as 'rf_pd_model_csv.pkl' and 'rf_pd_scaler_csv.pkl'")

    return clf, scaler

# --------------------------------------------------------------------------------
# EXAMPLE EXECUTION (In place of your existing main function call)
# --------------------------------------------------------------------------------

if __name__ == '__main__':
    # Use the new function to train directly from the CSV
    trained_model, data_scaler = train_from_csv(csv_path='pd_speech_features.csv')
    
    if trained_model:
        print("\nTraining complete using pre-extracted features.")

    # You can comment out the original `main()` call or remove it if you are 
    # no longer using raw audio files.
    # main(run_cnn=False)

if __name__ == '__main__':
    # By default run classical baseline. To also run CNN set run_cnn=True
    main(run_cnn=False)
