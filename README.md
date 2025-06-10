# mcmusic

FOR THE MCTS RUN 
1. CreateGenreMCTSModelL.py
2. preprocess.py
3. main.py

THOSE INSTRUCTIONS WILL ACTIVATE THE GUI 


### What This NEURAL NETWORK Does

Reads and parses genre labels from msd_tagtraum_cd1.cls
Matches genre-labeled tracks with MIDI files from lmd_matched
Extracts 20 musical features (e.g., pitch stats, rhythm, harmony, etc.)
Trains a neural network classifier to recognize genres
Predicts genre probabilities for any new MIDI input

### Usage Instructions

1. Set Up Your Files

Download the Lakh MIDI Dataset (LMD-matched) from [colinraffel.com/projects/lmd](https://colinraffel.com/projects/lmd/) and extract it into a folder named lmd_matched/ in your project directory.
Make sure the genre label file msd_tagtraum_cd1.cls is placed in the root directory of your project.
2. Train the Genre Classifier
Open the Python notebook.
Run the following blocks in order:
Block 2: Loads and processes genre data
Block 3: Matches MIDI files with genres and filters top genres
Block 6: Initializes the classifier and trains the neural network
This will extract features, train the model, and save it to disk.

3. Predict Genre for a New MIDI File
Go to Block 7.
Replace the example file ("Enya_-_Bard_Dance.mid") with the path to your own .mid file.
Run the block to see the genre probabilities predicted by the trained model.


{
  'Pop_Rock': 0.74,
  'Electronic': 0.18,
  'Country': 0.08
}

### Features Extracted

Each MIDI file is converted into a 20-dimensional feature vector including:

Song length
Number of instruments
Tempo statistics
Pitch mean, std, range
Rhythm patterns
Chord size (harmony)
Loudness stats (velocity)
Melodic interval stats
Percussion density
Unique pitch diversity

### Target Genres

By default, this model is trained on:

Pop_Rock
Electronic
Country
You can easily modify this list in the filtered_genres variable.

### Evaluation

After training, the model prints final test accuracy and shows training progress. You can further improve it by:

Increasing epochs
Using more advanced architectures
Expanding feature set
Adding more genres

### Troubleshooting

No valid MIDI files found – check the midi_path is correct and matches the expected LMD format.
Low accuracy? – consider balancing the dataset, checking feature quality, or increasing training time.


### Saving/Loading

Saves the model as my_genre_classifier_model.h5
Saves scaler and label encoder in .pkl format for reuse
Load with load_model() anytime


### Credits

MIDI parsing via pretty_midi
Genre labels via Tagtraum Industries
Dataset: Lakh MIDI Dataset