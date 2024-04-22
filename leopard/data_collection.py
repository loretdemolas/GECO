import os
import cv2
from datetime import datetime


class DataCollector:
    def __init__(self, frames_folder='dataset/newData'):
        self.vs = cv2.VideoCapture(0)
        self.frames_folder = frames_folder

    def detect_faces(self):
        while True:
            (grabbed, frame) = self.vs.read()
            if not grabbed:
                break

            # Save the raw frame
            self.collect_data(frame)

            # Display the frame (optional)
            cv2.imshow("Raw Frame", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        self.vs.release()

    def collect_data(self, frame):
        # Create a folder for the raw frames if it doesn't exist
        raw_frames_folder = os.path.join(self.frames_folder, 'raw_frames')
        if not os.path.exists(raw_frames_folder):
            os.makedirs(raw_frames_folder)

        # Get the current time
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")

        # Construct the frame filename with timestamp
        frame_filename = f"{current_time}.jpg"
        frame_path = os.path.join(raw_frames_folder, frame_filename)

        # Save the raw frame to the folder with the constructed filename
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])


if __name__ == "__main__":
    collector = DataCollector()
    collector.detect_faces()
