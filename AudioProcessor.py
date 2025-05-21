import pyaudio
import numpy as np
from MusicGenreModelProcessor import MusicGenreModelProcessor
from MCTS import MCTS
from MCTSNode import MCTSNode

class AudioProcessor:
    """
        Extracts features from raw audio and executes MCTS to identify the genre.
    """

    def __init__(self, sample_rate=44100, chunk=1024):
        # Audio setup
        self.pyaudio_instance = pyaudio.PyAudio()
        self.sample_rate = sample_rate
        self.chunk = chunk
        self.stream = None

        # Classifier and MCTS setup
        self.processor = MusicGenreModelProcessor()
        try:
            self.processor.load()
            print("Model loaded!")
        except FileNotFoundError:
            print("Model file not found. Did you run the training script first?")
        except Exception as e:
            print(f"Unexpected error loading model: {e}")

        self.mcts = MCTS()

    # --- Audio Input ---

    def start_recording(self):
        self.stream = self.pyaudio_instance.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        print("recording...")

    def get_audio_size(self):
        if self.stream is None:
            return None
        data = self.stream.read(self.chunk)
        return np.frombuffer(data, dtype=np.float32)

    def stop_recording(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio_instance.terminate()
        print("not recording...")

    # --- Classification ---

    def process_audio(self, audio_data):
        """
        Extracts features from raw audio and runs MCTS to classify genre.
        """
        features = self.processor.extract_features(audio_data, self.sample_rate)
        root = MCTSNode(state=features)
        result = self.mcts.run_search(root, self.processor, num_simulations=50)
        return result
