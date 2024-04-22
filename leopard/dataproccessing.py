import os
import cv2
from datetime import datetime
import numpy as np
import torchvision.transforms as transforms
from leopard.config import Config


class DataCollector:
    def __init__(self,  frames_folder='dataset/newData'):
        self.config = Config()
        self.net = cv2.dnn.readNetFromCaffe(self.config.prototxt_path, self.config.caffemodel_path)
        self.vs = cv2.VideoCapture(0)
        self.frames_folder = frames_folder
        self.image_padding = 10
        self.confidence = 0.5
        self.image_size = 300

        # Define the transformation pipeline for preprocessing
        self.preprocess_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

    def detect_faces(self):
        while True:
            (grabbed, frame) = self.vs.read()
            if not grabbed:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (self.image_size, self.image_size))
            self.net.setInput(blob)
            detections = self.net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.confidence:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (start_x, start_y, end_x, end_y) = box.astype("int")

                    # Add padding to the bounding box coordinates
                    padding = self.image_padding
                    start_x_padded = max(0, start_x - padding)
                    start_y_padded = max(0, start_y - padding)
                    end_x_padded = min(w, end_x + padding)
                    end_y_padded = min(h, end_y + padding)

                    # Extract the portion of the frame with padding
                    face = frame[start_y_padded:end_y_padded, start_x_padded:end_x_padded]

                    self.collect_data(face)
                    cv2.imshow("detection", face)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        self.vs.release()

    def collect_data(self, frame):
        # Create a folder for the corresponding feeling if it doesn't exist
        uncategorized_folder = os.path.join(self.frames_folder, 'uncategorized')
        if not os.path.exists(uncategorized_folder):
            os.makedirs(uncategorized_folder)

        # Get the current time
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")

        # Preprocess the frame
        preprocessed_frame = self.preprocess_transform(frame)
        np_frame = np.transpose(preprocessed_frame.numpy(), (1, 2, 0))  # Rearrange the dimensions
        np_frame = (np_frame * 255).astype(np.uint8)  # Convert to uint8 and scale to [0, 255]

        # Construct the frame filename with user_response, current time, and an incrementing number
        frame_filename = f"{current_time}.jpg"
        frame_path = os.path.join(uncategorized_folder, frame_filename)

        # Save the preprocessed frame to the folder with the constructed filename
        cv2.imwrite(frame_path, np_frame)


if __name__ == "__main__":
    collector = DataCollector()
    collector.detect_faces()





