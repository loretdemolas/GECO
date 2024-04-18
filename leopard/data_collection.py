import cv2
import time
import os
from tts import TextToSpeech
from stt import SpeechToText


class DataCollector:
    def __init__(self, frames_folder='dataset/newData'):
        self.frames_folder = frames_folder
        self.tts = TextToSpeech()
        self.stt = SpeechToText()

    def collect_data(self, feeling, frames):
        # Create a folder for the corresponding feeling if it doesn't exist
        feeling_folder = os.path.join(self.frames_folder, feeling)
        if not os.path.exists(feeling_folder):
            os.makedirs(feeling_folder)

        # Save captured frames
        for i, frame in enumerate(frames):
            frame_filename = f"{feeling}_{i}.jpg"  # Example: happy_0.jpg, happy_1.jpg, ...
            frame_path = os.path.join(feeling_folder, frame_filename)
            cv2.imwrite(frame_path, frame)

    def collect_data_prompt(self, frames):
        # Prompt user for how they are feeling
        self.tts.speak("How are you feeling right now?")
        time.sleep(1)  # Wait for a moment before starting recording

        # Get user response
        user_response = self.stt.listen()

        # Collect data based on user response
        self.collect_data(user_response, frames)

