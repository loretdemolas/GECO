import os


class Config:

    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.prototxt_path = os.path.join(os.getcwd(), 'model', 'deploy.prototxt.txt')
        self.caffemodel_path = os.path.join(os.getcwd(), 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
