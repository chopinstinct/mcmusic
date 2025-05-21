from AudioProcessor import AudioProcessor
from GenreIdentifierForm import GenreIdentifierForm

def main():
    print("Starting Music Genre Classifier...")

    processor = AudioProcessor()

    # create and run UI
    print("Launch GenreIdentifierForm... Press the 'Start Listening' button to begin.")
    form = GenreIdentifierForm(processor)
    form.run()

if __name__ == "__main__":
    main()
