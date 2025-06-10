import os
import numpy as np
import pandas as pd
import pickle
import pretty_midi
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# === Load saved model, scaler, label encoder ===
model = load_model("my_genre_classifier_model.h5")
with open("my_genre_classifier_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("my_genre_classifier_labels.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# === Load df_filtered that you trained on ===
df = pd.read_csv("df_filtered.csv")  # You MUST save df_filtered to CSV earlier

# === Feature extraction (must match training!) ===
def extract_features(midi_file):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
    except Exception as e:
        print(f"Skipping {midi_file}: {e}")
        return None

    features = np.zeros(20)
    total_time = midi_data.get_end_time()
    if total_time == 0: return features
    features[0] = total_time
    features[1] = len(midi_data.instruments)

    # Tempo
    tempo_changes = midi_data.get_tempo_changes()
    features[2] = np.mean(tempo_changes[1]) if len(tempo_changes[1]) > 0 else 120.0

    notes = [note for inst in midi_data.instruments if not inst.is_drum for note in inst.notes]
    if not notes: return features
    pitches = [n.pitch for n in notes]
    durations = [n.end - n.start for n in notes]
    velocities = [n.velocity for n in notes]
    features[3] = np.mean(pitches)
    features[4] = np.std(pitches)
    features[5] = max(pitches) - min(pitches)
    features[6] = np.mean(durations)
    features[7] = np.std(durations)
    features[8] = len(notes) / total_time
    features[9] = np.mean(velocities)
    features[10] = np.std(velocities)
    # Harmony
    onset_groups = defaultdict(list)
    for note in notes:
        onset_time = round(note.start * 4) / 4
        onset_groups[onset_time].append(note.pitch)
    if onset_groups:
        sizes = [len(p) for p in onset_groups.values()]
        features[11] = np.mean(sizes)
        features[12] = max(sizes)
    sorted_notes = sorted(notes, key=lambda n: n.start)
    if len(sorted_notes) > 1:
        intervals = [abs(sorted_notes[i+1].pitch - sorted_notes[i].pitch) for i in range(len(sorted_notes)-1)]
        if intervals:
            features[13] = np.mean(intervals)
            features[14] = sum(1 for i in intervals if i <= 2) / len(intervals)
    pitch_classes = [p % 12 for p in pitches]
    pitch_class_counts = [pitch_classes.count(pc) for pc in range(12)]
    features[15] = np.std(pitch_class_counts)
    features[16] = max(pitch_class_counts) / len(pitches)
    drum_notes = [n for inst in midi_data.instruments if inst.is_drum for n in inst.notes]
    features[17] = len(drum_notes) / total_time if drum_notes else 0
    features[18] = len(set(n.pitch for n in drum_notes)) if drum_notes else 0
    features[19] = len(set(pitches))
    return features

# === Build dataset from df_filtered ===
X = []
y = []
for _, row in df.iterrows():
    features = extract_features(row['Path'])
    if features is not None:
        X.append(features)
        y.append(row['Genre'])

X = np.array(X)
y = label_encoder.transform(y)
X_scaled = scaler.transform(X)

# === Split
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Predict + Evaluate
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
