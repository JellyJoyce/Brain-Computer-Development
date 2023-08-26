import pickle
from scipy.io import wavfile
from test import extract_feature, extract_wav_data
import numpy as np

def zero_crossing_rate(frame):
    count = 0
    for i in range(1, len(frame)):
        if (frame[i - 1] >= 0 and frame[i] < 0) or (frame[i - 1] < 0 and frame[i] >= 0):
            count += 1
    return count

# Short-Time Zero Crossing Rate, STZCR
def detect_action_start_times(signal, sample_rate):
    frame_size = int(sample_rate) 
    threshold = 10

    zc_rates = []
    for i in range(0, len(signal) - frame_size, frame_size):
        frame = signal[i:i + frame_size]
        zcr = zero_crossing_rate(frame)
        zc_rates.append(zcr)
    print(4)
    print(zc_rates)

    action_start_times = []
    for i in range(1, len(zc_rates)):
        if zc_rates[i] - zc_rates[i - 1] > threshold:
            action_start_time = i
            action_start_times.append(action_start_time)

    return action_start_times

def predict_labels(X_new, model):
    y_pred = model.predict(X_new)
    return y_pred

def extract_features_new_wavefile(action_start_times, signal, sample_rate):
    features_matrix = []
    for i in range(len(action_start_times) - 1):
        start_point_index = int(action_start_times[i] * sample_rate)
        end_point_index = int(action_start_times[i + 1] * sample_rate)
        signal_segment = signal[start_point_index:end_point_index]
        features_matrix.append((np.max(signal_segment), np.min(signal_segment), np.std(signal_segment)))

    X_new = np.array(features_matrix)
    return X_new

def predict_labels(X_new, model):
    y_pred = model.predict(X_new)
    return y_pred

# Preprocess the new audio file and extract features
new_audio_file = "data/BYB_Recording_2023-04-02_12.41.17.wav"
signal, sample_rate = extract_wav_data(new_audio_file)
time = len(signal) / sample_rate
print(1)
print(time)
print(2)
print(signal, sample_rate)

action_start_times = detect_action_start_times(signal, sample_rate)
print(3)
print(action_start_times)

X_new = extract_features_new_wavefile(action_start_times, signal, sample_rate)
print(X_new)
labels_placeholder =  len(action_start_times) * ['Unknown']
labels_placeholder_np = np.array(labels_placeholder)

#load model
with open("rf_classifier.pkl", "rb") as f:
    rf_model = pickle.load(f)

predicted_labels = predict_labels(X_new, rf_model)

print("Predicted Labels:", predicted_labels)
