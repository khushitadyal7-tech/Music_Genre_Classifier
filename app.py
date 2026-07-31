import streamlit as st
import joblib
import librosa
import numpy as np


# Load model
model = joblib.load("knn_model.pkl")


# App UI
st.title("🎵 Music Genre Classification")
st.write("Upload an audio file and predict its genre")


audio_file = st.file_uploader(
    "Choose a music file",
    type=["wav", "mp3"]
)


# Extract 57 Features
def extract_features(file):

    audio, sr = librosa.load(file, duration=30)

    features = []


    # Chroma STFT (2)
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr
    )

    features.append(np.mean(chroma))
    features.append(np.var(chroma))


    # RMS (2)
    rms = librosa.feature.rms(y=audio)

    features.append(np.mean(rms))
    features.append(np.var(rms))


    # Spectral Centroid (2)
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    features.append(np.mean(centroid))
    features.append(np.var(centroid))


    # Spectral Bandwidth (2)
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )

    features.append(np.mean(bandwidth))
    features.append(np.var(bandwidth))


    # Roll Off (2)
    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )

    features.append(np.mean(rolloff))
    features.append(np.var(rolloff))


    # Zero Crossing Rate (2)
    zcr = librosa.feature.zero_crossing_rate(audio)

    features.append(np.mean(zcr))
    features.append(np.var(zcr))


    # Harmony (2)
    harmonic, percussive = librosa.effects.hpss(audio)

    features.append(np.mean(harmonic))
    features.append(np.var(harmonic))


    # Percussion (2)
    features.append(np.mean(percussive))
    features.append(np.var(percussive))


    # Tempo (1)
    tempo, _ = librosa.beat.beat_track(
        y=audio,
        sr=sr
    )

    features.append(float(tempo[0]))


    # MFCC (40)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=20
    )

    for i in range(20):
        features.append(np.mean(mfcc[i]))
        features.append(np.var(mfcc[i]))


    return np.array(features).reshape(1, -1)



# Prediction
if audio_file is not None:

    st.audio(audio_file)

    st.success("File uploaded successfully ✅")


    if st.button("Predict Genre"):

        features = extract_features(audio_file)

        st.write("Feature shape:", features.shape)

        prediction = model.predict(features)

        st.success(
            f"🎵 Predicted Genre: {prediction[0]}"
        )