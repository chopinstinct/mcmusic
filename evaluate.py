import os
import librosa
import numpy as np
from MusicGenreModelProcessor import MusicGenreModelProcessor
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def test_model():

    test_folder = 'GTZAN/genres_split/test'
    print(f"Loading model from: {'genre_model.pkl'}")
    processor = MusicGenreModelProcessor()
    try:
        processor.load('genre_model.pkl')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    y_true = []
    y_pred = []
    
    results = {genre: {'correct': 0, 'total': 0} for genre in processor.genre_list}
    
    for genre in processor.genre_list:
        genre_folder = os.path.join(test_folder, genre)
        if not os.path.exists(genre_folder):
            print(f"Warning: No folder found for {genre}")
            continue
            
        print(f"\nTesting {genre} songs...")
        
        for file in os.listdir(genre_folder):
            if file.endswith(('.mp3', '.wav')):
                file_path = os.path.join(genre_folder, file)
                try:
                    audio_data, sr = librosa.load(file_path, sr=44100)
                    
                    features = processor.extract_features(audio_data, sr)
                    
                    prediction = processor.predict(features)
                    predicted_genre = prediction['genre']
                    
                    y_true.append(genre)
                    y_pred.append(predicted_genre)
                    
                    results[genre]['total'] += 1
                    if predicted_genre == genre:
                        results[genre]['correct'] += 1
                    
                    print(f"File: {file}")
                    print(f"True genre: {genre}")
                    print(f"Predicted: {predicted_genre}")
                    print(f"Confidence scores: {prediction['confidence_scores']}")
                    print("-" * 50)
                    
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    print("\nOverall Results:")
    print("-" * 50)
    total_correct = 0
    total_files = 0
    
    for genre in processor.genre_list:
        if results[genre]['total'] > 0:
            accuracy = results[genre]['correct'] / results[genre]['total'] * 100
            print(f"{genre}: {results[genre]['correct']}/{results[genre]['total']} correct ({accuracy:.1f}%)")
            total_correct += results[genre]['correct']
            total_files += results[genre]['total']
    
    if total_files > 0:
        overall_accuracy = total_correct / total_files * 100
        print(f"\nOverall accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_files} correct)")
        
        print("\nGenerating confusion matrix...")
        cm = confusion_matrix(y_true, y_pred, labels=processor.genre_list)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=processor.genre_list)
        
        plt.figure(figsize=(12, 8))
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

def main():
    
    test_model()

if __name__ == "__main__":
    main() 