import os
import cv2
from datetime import datetime
import numpy as np
import torchvision.transforms as transforms


from leopard.stt import SpeechToText
from leopard.tts import TextToSpeech


class DataCollector:
    def __init__(self, frames_folder='dataset/newData'):
        self.frames_folder = frames_folder
        self.tts = TextToSpeech()
        self.stt = SpeechToText()
        self.collection_full = False
        self.arrayOfFrames = []
        self.user_response = ''

        # Define the transformation pipeline for preprocessing
        self.preprocess_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def collect_frames(self, frame):
        if len(self.arrayOfFrames) >= 50:
            self.collection_full = True
            print("Collection is full.")
        else:
            self.arrayOfFrames.append(frame)
            count = len(self.arrayOfFrames)
            strcount = str(count)+ "/50 frames collected. "
            print(strcount)
    def collect_data(self):
        # Create a folder for the corresponding feeling if it doesn't exist
        feeling_folder = os.path.join(self.frames_folder, self.user_response)
        if not os.path.exists(feeling_folder):
            os.makedirs(feeling_folder)

        # Get the current time
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")

        # Preprocess and save each frame
        for i, frame in enumerate(self.arrayOfFrames):
            # Preprocess the frame
            preprocessed_frame = self.preprocess_transform(frame)
            np_frame = np.transpose(preprocessed_frame.numpy(), (1, 2, 0)) #Rearrange the dimensions
            np_frame = (np_frame * 255).astype(np.uint8)# Convert to uint8 and scale to [0, 255]

            # Construct the frame filename with user_response, current time, and an incrementing number
            frame_filename = f"{self.user_response}_{current_time}_{i}.jpg"
            frame_path = os.path.join(feeling_folder, frame_filename)

            # Save the preprocessed frame to the folder with the constructed filename
            cv2.imwrite(frame_path, np_frame)

        # Cleans up after saving all frames
        self.arrayOfFrames = []
        self.collection_full = False
        self.user_response = ''
        self.stt.responded = False

    def collect_data_prompt(self):
        # Get user response
        self.user_response = self.stt.listen()

        # Prompt user for how they are feeling
        self.tts.speak("How are you feeling right now?")


        if self.stt.responded:
            self.tts.speak("Thank You")

        # Collect data based on user response
        self.collect_data()

