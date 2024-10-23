# dataset_managers/few_shot_dataset_manager.py

import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pydicom

class FewShotDatasetManager:
    def __init__(self, data_dir: str, n_way: int, k_shot: int, query_size: int):
        self.data_dir = data_dir
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size
        self.data = {}
        self.support_set = []
        self.query_set = []
        self.image_shape = None

    def load_data(self):
        """
        Load DICOM data into the dataset manager and classify them by directory names.
        """
        for root, dirs, files in os.walk(self.data_dir):
            for file_name in files:
                if file_name.endswith('.dcm'):
                    class_name = os.path.basename(root)
                    file_path = os.path.join(root, file_name)
                    if class_name not in self.data:
                        self.data[class_name] = []
                    self.data[class_name].append(self._load_file(file_path))
        self.image_shape = next(iter(self.data.values()))[0].shape

    def _load_file(self, file_path):
        ds = pydicom.dcmread(file_path)
        return ds.pixel_array

    def split_few_shot(self):
        """
        Create support and query sets for few-shot learning.
        """
        sampled_classes = random.sample(self.data.keys(), self.n_way)
        
        for class_name in sampled_classes:
            images = self.data[class_name]
            random.shuffle(images)
            support_images = images[:self.k_shot]
            query_images = images[self.k_shot:self.k_shot + self.query_size]
            
            self.support_set.append((support_images, class_name))
            self.query_set.append((query_images, class_name))

    def get_image_shape(self):
        return self.image_shape

    def get_support_and_query_sets(self):
        return self.support_set, self.query_set
