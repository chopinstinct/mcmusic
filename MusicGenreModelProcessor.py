import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import os
import librosa

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

class MusicGenreModelProcessor:
    """
    Handles training, prediction, feature extraction, and genre evaluation.
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.genre_list = GENRES

    def train(self, features_list, labels):
        """
        Train the classifier with example music samples.
        """
        self.feature_names = list(features_list[0].keys())
        X = np.array([[features[name] for name in self.feature_names] for features in features_list])
        y = np.array(labels)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        print(f"Model trained with {len(X)} music samples")

    def predict(self, features):
        """
        Predict the genre of a music sample based on its audio features.
        """
        if self.model is None:
            raise Exception("Model not trained yet! Please train or load a model first.")

        feature_vector = np.array([[features[name] for name in self.feature_names]])
        genre_index = self.model.predict(feature_vector)[0]
        confidence_scores = self.model.predict_proba(feature_vector)[0]

        return {
            'genre': GENRES[genre_index],
            'confidence_scores': {GENRES[i]: score for i, score in enumerate(confidence_scores)}
        }

    def save(self, file_path="genre_model.pkl"):
        """
        Save the trained model to a file so we can use it later.
        """
        if self.model is None:
            raise Exception("No model to save! Please train a model first.")

        with open(file_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names
            }, f)

        print(f"Model saved to {file_path}")

    def load(self, file_path="genre_model.pkl"):
        """
        Load a previously trained model from a file.
        """
        if not os.path.exists(file_path):
            raise Exception(f"Model file {file_path} not found!")

        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']

        print(f"Model loaded from {file_path}")

    def extract_features(self, signal, samplerate=44100):
        """
        Extract audio features from a signal.
        """
        features = {}

        chroma = librosa.feature.chroma_stft(y=signal, sr=samplerate)
        features["chroma_avg"] = np.mean(chroma)
        features["chroma_std"] = np.var(chroma)

        energy = librosa.feature.rms(y=signal)
        features["rms_avg"] = np.mean(energy)
        features["rms_std"] = np.var(energy)

        centroid = librosa.feature.spectral_centroid(y=signal, sr=samplerate)
        features["centroid_avg"] = np.mean(centroid)
        features["centroid_std"] = np.var(centroid)

        bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=samplerate)
        features["bandwidth_avg"] = np.mean(bandwidth)
        features["bandwidth_std"] = np.var(bandwidth)

        rolloff = librosa.feature.spectral_rolloff(y=signal, sr=samplerate)
        features["rolloff_avg"] = np.mean(rolloff)
        features["rolloff_std"] = np.var(rolloff)

        zero_cross = librosa.feature.zero_crossing_rate(y=signal)
        features["zero_crossing_rate_avg"] = np.mean(zero_cross)
        features["zero_crossing_rate_std"] = np.var(zero_cross)

        harmonic = librosa.effects.harmonic(y=signal)
        features["harmonic_avg"] = np.mean(harmonic)
        features["harmonic_std"] = np.var(harmonic)

        flatness = librosa.feature.spectral_flatness(y=signal)
        features["flatness_avg"] = np.mean(flatness)
        features["flatness_std"] = np.var(flatness)

        tempo = librosa.beat.tempo(onset_envelope=librosa.onset.onset_strength(y=signal, sr=samplerate), sr=samplerate)
        features["tempo"] = float(tempo[0])

        mfcc = librosa.feature.mfcc(y=signal, sr=samplerate, n_mfcc=20)
        for index in range(20):
            features[f"mfcc{index+1}_avg"] = np.mean(mfcc[index])
            features[f"mfcc{index+1}_std"] = np.var(mfcc[index])

        return features

    def modify_features(self, features, genre):
        """
        Changes the features to match the genre better
        """
        # Make a copy so we don't mess up the original
        new_features = features.copy()

        # Blues sounds slower and more mellow
        if genre == "blues":
            new_features['tempo'] = new_features['tempo'] * 0.9
            new_features['rms_avg'] = new_features['rms_avg'] * 0.95

        # Classical has less bass and more dynamics
        elif genre == "classical":
            new_features['tempo'] = new_features['tempo'] * 0.85
            new_features['rms_avg'] = new_features['rms_avg'] * 0.8
            new_features['rms_std'] = new_features['rms_std'] * 1.4  # more dynamic range

        # Country has guitar and medium tempo
        elif genre == "country":
            new_features['tempo'] = new_features['tempo'] * 0.95
            new_features['harmonic_avg'] = new_features['harmonic_avg'] * 1.1  # more harmonic

        # Disco is fast with a beat
        elif genre == "disco":
            new_features['tempo'] = new_features['tempo'] * 1.15
            new_features['rms_avg'] = new_features['rms_avg'] * 1.1  # louder
            # Disco has more bass
            if 'mfcc1_avg' in new_features:
                new_features['mfcc1_avg'] = new_features['mfcc1_avg'] * 1.15

        # Hip hop has lots of bass and beats
        elif genre == "hiphop":
            if 'mfcc1_avg' in new_features:
                new_features['mfcc1_avg'] = new_features['mfcc1_avg'] * 1.2  # more bass
            new_features['tempo'] = new_features['tempo'] * 0.9

        # Jazz has complex harmony
        elif genre == "jazz":
            if 'mfcc2_avg' in new_features:
                new_features['mfcc2_avg'] = new_features['mfcc2_avg'] * 1.15
            new_features['harmonic_avg'] = new_features['harmonic_avg'] * 1.2

        # Metal is loud and fast
        elif genre == "metal":
            new_features['rms_avg'] = new_features['rms_avg'] * 1.3  # metal is loud!
            new_features['tempo'] = new_features['tempo'] * 1.25  # and fast!
            # Metal has distortion so more zero crossings
            new_features['zero_crossing_rate_avg'] = new_features['zero_crossing_rate_avg'] * 1.3

        # Pop has catchy vocals
        elif genre == "pop":
            new_features['tempo'] = new_features['tempo'] * 1.05  # upbeat
            if 'mfcc2_avg' in new_features and 'mfcc3_avg' in new_features:
                # These mfccs might represent vocals
                new_features['mfcc2_avg'] = new_features['mfcc2_avg'] * 1.1
                new_features['mfcc3_avg'] = new_features['mfcc3_avg'] * 1.1

        # Reggae has offbeat rhythm
        elif genre == "reggae":
            new_features['tempo'] = new_features['tempo'] * 0.85  # slower tempo
            # Emphasize certain frequencies
            if 'mfcc4_avg' in new_features:
                new_features['mfcc4_avg'] = new_features['mfcc4_avg'] * 1.2

        # Rock has guitar and drums
        elif genre == "rock":
            new_features['rms_avg'] = new_features['rms_avg'] * 1.2  # rock is loud
            new_features['zero_crossing_rate_avg'] = new_features['zero_crossing_rate_avg'] * 1.2  # distortion
            # Middle frequencies for guitars
            if 'mfcc3_avg' in new_features:
                new_features['mfcc3_avg'] = new_features['mfcc3_avg'] * 1.15

        # Return the changed features
        return new_features

    def try_genre(self, node):
        """
        Evaluate a genre candidate node and return its confidence score.
        """
        prediction = self.predict(node.state)
        genre_tested = None

        for genre in node.parent.children:
            if node.parent.children[genre] == node:
                genre_tested = genre
                break

        return prediction['confidence_scores'].get(genre_tested, 0.01)

    def pick_best_genre(self, root):
        """
        Choose the most likely genre from multiple simulated candidates.
        """
        best_genre = None
        best_score = -1
        genre_scores = {}

        for genre in root.children:
            child = root.children[genre]
            if child.visits > 0:
                average = child.score / child.visits
                genre_scores[genre] = average
                if average > best_score:
                    best_score = average
                    best_genre = genre

        total = sum(genre_scores.values())
        confidence_scores = {}

        for genre in genre_scores:
            confidence_scores[genre] = genre_scores[genre] / total

        return {
            'genre': best_genre,
            'confidence_scores': confidence_scores
        }
