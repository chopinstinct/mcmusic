import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import numpy as np


class GenreIdentifierForm:

    """
    Form to record audio and displays a chart of the confidence of the genre identified
    """

    def __init__(self, processor, root = None):

        # initialize/store variables
        self.processor = processor

        # creation of the main window as long as not given
        if root is None:

            self.root = tk.Tk()
            self.root.title("What is the genre?")
            self.root.geometry("800x600")

        else:

            self.root = root

        # track if the model is currently recording, start as false default
        self.is_recording = False

        # thread for parallel processing of audio in the background
        self.recording_thread = None

        # creates the gui elements
        self.create_controls()


    def create_controls(self):

        top_frame = ttk.Frame(self.root, padding = 10)
        top_frame.pack(fill = tk.X)

        # stop and record button
        self.record_button = ttk.Button(top_frame,text= "Start recording", command = self.toggle_recording)
        self.record_button.pack(side = tk.LEFT, padx = 5)

        # status label
        self.status_label = ttk.Label(top_frame, text = "Recording")
        self.status_label.pack(side = tk.LEFT, padx = 10)

        # middle section
        # aka frame displaying detected genres
        genre_frame = ttk.LabelFrame(self.root, text = "Detected genre")
        genre_frame.pack(fill = tk.BOTH, expand = True, padx = 10, pady = 5)

        # show the current genre
        self.genre_label = ttk.Label(genre_frame, text = "Genre", font = ("Arial", 24))
        self.genre_label.pack(pady = 20)

        # create a matplot figure for the confidence bar chart

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=genre_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # empty chart
        self.update_confidence({})


    def toggle_recording(self):

        if not self.is_recording:

            self.is_recording = True
            self.record_button.config(text="Stop recording")
            self.status_label.config(text="Listening for music")

            # start getting audio
            self.processor.start_recording()

            # starts processing audio in a separate thread
            self.recording_thread = threading.Thread(target=self.process_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
        else:
            # stopping recording
            self.is_recording = False
            self.record_button.config(text="Start listening")
            self.status_label.config(text="Stopped")

            # Stop capturing audio
            self.processor.stop_recording()


    def process_audio(self):

        buffer = []
        # num of chunks of audio to analyze
        buffer_size = 5

        while self.is_recording:
            # get current audio piece
            chunk = self.processor.get_audio_size()
            if chunk is None:
                # wait to see if no audio
                time.sleep(0.1)
                continue

            # add chunk to buffer
            buffer.append(chunk)

            # when enough data based on size, process
            if len(buffer) >= buffer_size:
                # put all audio chunks together to make one audio piece
                audio_data = np.concatenate(buffer)

                # process audio and update ui
                try:
                    # get genre pred
                    result = self.processor.process_audio(audio_data)

                    # update ui must be done from main thread
                    self.root.after(0, lambda: self.update_display(result))
                except Exception as e:
                    print(f"Error processing audio: {e}")

                # clear buffer, last chunk is for smooth transitions
                buffer = [buffer[-1]]

            time.sleep(0.01)


    def update_display(self, result):

        if result:
            self.genre_label.config(text=result['genre'])

            self.update_confidence(result['confidence_scores'])


    def update_confidence(self, confidence_scores):

        # clear prev chart
        self.ax.clear()

        if not confidence_scores:

            self.ax.text(0.5, 0.5, "Waiting for music...", ha='center', va='center')

        else:

            # sort genres by confidence highest CI to lowest
            sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
            genres = [x[0] for x in sorted_scores]
            scores = [x[1] for x in sorted_scores]

            # bars in graph
            bars = self.ax.bar(genres, scores, color='skyblue')

            # top genre in a different color
            bars[0].set_color('navy')

            # label and more formatting
            self.ax.set_ylabel('Confidence')

            # confidence 0 to 1
            self.ax.set_ylim(0, 1)
            plt.xticks(rotation=45, ha='right')

        # Refresh the chart
        self.canvas.draw()


    def run(self):

        self.root.mainloop()