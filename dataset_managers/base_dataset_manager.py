# dataset_managers/base_dataset_manager.py

# Other imports remain unchanged
import os
import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from skimage.transform import resize
import matplotlib.pyplot as plt
import seaborn as sns


class BaseDatasetManager:
    def __init__(self, data_dir: str, target_image_size: tuple = (28, 28)):
        self.data_dir = data_dir
        self.data = {}
        self.metadata = {}
        self.target_image_size = target_image_size

    def load_data(self):
        """Loads and preprocesses DICOM files."""
        for root, dirs, files in os.walk(self.data_dir):
            for file_name in files:
                if file_name.endswith('.dcm'):
                    file_path = os.path.join(root, file_name)
                    class_name = os.path.basename(root)
                    if class_name not in self.data:
                        self.data[class_name] = []
                    img = self._load_file(file_path)
                    resized_img = self._preprocess_image(img)
                    self.data[class_name].append(resized_img)
        self.generate_metadata()

    def generate_metadata(self):
        """Generates metadata such as class counts and class balance."""
        class_counts = {class_name: len(files) for class_name, files in self.data.items()}
        self.metadata = {
            'class_counts': class_counts,
            'total_samples': sum(class_counts.values()),
            'class_balance': {
                class_name: count / sum(class_counts.values()) for class_name, count in class_counts.items()
            }
        }
        self.metadata['num_classes'] = len(class_counts)
        self.metadata['image_size'] = self.target_image_size

    def visualize_data_distribution(self):
        """Visualizes the class distribution."""
        class_counts = self.metadata.get('class_counts', {})
        sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def save_metadata_to_excel(self, output_file: str):
        """Saves metadata to an Excel file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        metadata_df = pd.DataFrame.from_dict(self.metadata['class_counts'], orient='index', columns=['Count'])
        metadata_df['Class Balance'] = metadata_df['Count'] / self.metadata['total_samples']
        metadata_df.to_excel(output_file, index_label='Class')

    def split_data(self, test_size=0.3):
        train_data = {}
        test_data = {}
        for class_name, files in self.data.items():
            if len(files) < 20:
                # If there's only 1 sample, add it all to the training set and skip splitting
                print(f"Class '{class_name}' has less than 20 samples. Skipping split.")
                train_data[class_name] = files
                test_data[class_name] = []  # Empty test data
            else:
                train, test = train_test_split(files, test_size=test_size, random_state=42)
                train_data[class_name] = train
                test_data[class_name] = test

        return train_data, test_data

    def _load_file(self, file_path: str) -> np.ndarray:
        ds = pydicom.dcmread(file_path)
        return ds.pixel_array

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        resized_img = resize(img, self.target_image_size, anti_aliasing=True)
        return resized_img
