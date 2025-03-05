import os
import pydicom
import pandas as pd
import cv2
import numpy as np
from pydicom.errors import InvalidDicomError

class BaseDatasetManager:
    def __init__(self, root_dir, target_size=(224, 224)):
        self.root_dir = root_dir
        self.target_size = target_size
        self.metadata = pd.DataFrame()

    def load_data(self):
        """Load and preprocess DICOM images, generating metadata."""
        data = []
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                for file_name in os.listdir(class_path):
                    if file_name.endswith('.dcm'):
                        file_path = os.path.join(class_path, file_name)
                        try:
                            dicom_data = pydicom.dcmread(file_path, force=True)
                            image_array = dicom_data.pixel_array
                            preprocessed_image = self.preprocess_image(image_array)
                            data.append({
                                'file_path': file_path,
                                'class_name': class_name,
                                'width': preprocessed_image.shape[1],
                                'height': preprocessed_image.shape[0]
                            })
                        except (InvalidDicomError, AttributeError) as e:
                            print(f"Warning: {file_path} is not a valid DICOM file. Skipping.")
        
        self.metadata = pd.DataFrame(data)

    def preprocess_image(self, image):
        resized_image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        normalized_image = resized_image / np.max(resized_image)
        return normalized_image
