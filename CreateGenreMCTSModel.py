import os
import numpy as np
from MusicGenreModelProcessor import MusicGenreModelProcessor, GENRES
import librosa

def CreateGenreMCTSModel():
    """
    Build an MCTS Model by reading wav files and then generating the output model.
    """
    print("Starting genre model creation...")

    data_path = "GTZAN/genres_split/train"
    featuresList = []
    labels = []

    # create the classifier instance
    classifier = MusicGenreModelProcessor()

    for genreIdx, genre in enumerate(GENRES):
        genreFolder = os.path.join(data_path, genre)

        if not os.path.exists(genreFolder):
            print(f"\nNo folder found for {genre}. Skipping the processing.")
            continue

        print(f"Processing {genre} songs...")

        audioFiles = [f for f in os.listdir(genreFolder)
                      if f.endswith('.mp3') or f.endswith('.wav')]

        for audioFile in audioFiles:
            filePath = os.path.join(genreFolder, audioFile)
            print(f"  - Extracting features from {audioFile}")

            try:
                # load audio file (use only first 30 secs)
                audioData, sampleRate = librosa.load(filePath, sr=44100, duration=30)

                # extract audio features using the processor
                features = classifier.extract_features(audioData, sampleRate)

                featuresList.append(features)
                labels.append(genreIdx)

            except FileNotFoundError:
                print(f"\nFile not found: {filePath}. Skipping...")
            except Exception as e:
                print(f"\nError processing {audioFile}: {e}, Skipping this file and continuing...")

    if len(featuresList) < 5:
        print("\nTo train a model we have need at least 5 songs.")
        return

    print(f"\nTraining with {len(featuresList)} songs across {len(set(labels))} genres")

    classifier.train(featuresList, labels)
    classifier.save()

    print("\nTraining complete! Model saved to 'genre_model.pkl'")


if __name__ == "__main__":
    CreateGenreMCTSModel()
