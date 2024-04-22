import cv2
import os
import shutil
from leopard.config import Config


class ImageAnnotator:
    def __init__(self, dataset='dataset'):
        self.config = Config()
        self.dataset = self.config.dataset_folder
        self.frames_folder = os.path.join(dataset, 'newData', 'raw_frames')
        self.image_files = sorted(os.listdir(self.frames_folder))
        self.current_index = 0
        self.annotations = []

    def annotate_images(self):
        for image_file in self.image_files:
            image_path = os.path.join(self.frames_folder, image_file)
            image = cv2.imread(image_path)

            cv2.imshow('Image', image)
            key = cv2.waitKey(0)

            if key == ord('q'):
                break
            elif key == ord('up'):
                annotation = 'neutral'
            elif key == ord('left'):
                annotation = 'positive'
            elif key == ord('right'):
                annotation = 'negative'
            elif key == ord('down') and self.current_index > 0:
                self.current_index -= 1
                self.annotations.pop()
                continue
            else:
                continue

            self.annotations.append((image_file, annotation))
            print(f"Annotated '{image_file}' as '{annotation}'")

        cv2.destroyAllWindows()

    def save_annotations(self):
        annotations_folder = os.path.join(self.dataset, 'annotations')
        if not os.path.exists(annotations_folder):
            os.makedirs(annotations_folder)

        # Move the annotated images to their corresponding folders
        for image_file, annotation in self.annotations:
            annotation_folder = os.path.join(annotations_folder, annotation)
            if not os.path.exists(annotation_folder):
                os.makedirs(annotation_folder)

            src_path = os.path.join(self.frames_folder, image_file)
            dst_path = os.path.join(annotation_folder, image_file)
            shutil.move(src_path, dst_path)


if __name__ == "__main__":
    annotator = ImageAnnotator()
    annotator.annotate_images()
    annotator.save_annotations()
