import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import pydicom
import h5py

class BaseDatasetManager:
    def __init__(self, data_dir: str):
        """
        Initialize the BaseDatasetManager with a directory path.

        Parameters:
        data_dir (str): Path to the dataset directory.
        """
        self.data_dir = data_dir
        self.data = {}
        self.metadata = {}
        self.normalizer = None

    def load_data(self) -> None:
        """
        Load DICOM data from directories into a dictionary where each key is a class name 
        and each value is a list of data samples.
        """
        for root, dirs, files in os.walk(self.data_dir):
            for file_name in files:
                if file_name.endswith('.dcm'):
                    file_path = os.path.join(root, file_name)
                    class_name = os.path.basename(root)  # Assuming class name is the folder name
                    if class_name not in self.data:
                        self.data[class_name] = []
                    self.data[class_name].append(self._load_file(file_path))
        self.generate_metadata()

    def generate_metadata(self) -> None:
        """
        Generate metadata such as number of samples per class, total samples, and class balance.
        """
        class_counts = {class_name: len(files) for class_name, files in self.data.items()}
        self.metadata = {
            'class_counts': class_counts,
            'total_samples': sum(class_counts.values()),
            'class_balance': {
                class_name: count / sum(class_counts.values()) for class_name, count in class_counts.items()
            }
        }

    def check_data_quality(self) -> dict:
        """
        Perform data quality checks and return any issues found.

        Returns:
        dict: Dictionary containing quality issues per class.
        """
        quality_issues = {}
        for class_name, files in self.data.items():
            issues = []
            for i, file_data in enumerate(files):
                if not self._is_valid(file_data):
                    issues.append(f"File {i} in class {class_name} is corrupted")
            quality_issues[class_name] = issues
        self.metadata['quality_issues'] = quality_issues
        return quality_issues

    def preprocess_data(self, method: str = 'standard') -> None:
        """
        Preprocess the data using the specified normalization method.

        Parameters:
        method (str): Normalization method ('standard' or 'minmax').
        """
        if method == 'standard':
            self.normalizer = StandardScaler()
        elif method == 'minmax':
            self.normalizer = MinMaxScaler()
        else:
            raise ValueError(f"Unknown method: {method}")

        for class_name, files in self.data.items():
            flattened = np.array([file.flatten() for file in files])
            normalized = self.normalizer.fit_transform(flattened)
            self.data[class_name] = [norm.reshape(files[0].shape) for norm in normalized]

    def split_data(self, test_size: float = 0.3) -> tuple:
        """
        Split the data into training and test sets.

        Parameters:
        test_size (float): Proportion of data to use for testing.

        Returns:
        tuple: Training and test datasets.
        """
        train_data = {}
        test_data = {}
        for class_name, files in self.data.items():
            if len(files) == 0:
                print(f"Warning: Class '{class_name}' has no data.")
                continue
            train, test = train_test_split(files, test_size=test_size, random_state=42)
            train_data[class_name] = train
            test_data[class_name] = test
        return train_data, test_data

    def visualize_data_distribution(self) -> None:
        """
        Visualize the distribution of classes in the dataset.
        """
        class_counts = self.metadata.get('class_counts', {})
        sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.show()

    def visualize_sample(self, class_name: str, index: int = 0) -> None:
        """
        Visualize a single data sample from a specified class.

        Parameters:
        class_name (str): The class from which to visualize a sample.
        index (int): The index of the sample within the class.
        """
        if class_name in self.data:
            sample = self.data[class_name][index]
            plt.imshow(sample, cmap='gray')
            plt.title(f"Class: {class_name}, Index: {index}")
            plt.show()
        else:
            print(f"Class '{class_name}' not found.")

    def convert_to_h5(self, output_file: str) -> None:
        """
        Convert the dataset to HDF5 format.

        Parameters:
        output_file (str): Path to save the HDF5 file.
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with h5py.File(output_file, 'w') as h5f:
            for class_name, files in tqdm(self.data.items(), desc="Saving to HDF5"):
                group = h5f.create_group(class_name)
                for i, file_data in enumerate(files):
                    group.create_dataset(f'data_{i}', data=file_data)

    def extract_dicom_metadata(self) -> list:
        """
        Extract metadata from DICOM files.

        Returns:
        list: List of metadata dictionaries.
        """
        metadata_list = []
        for root, dirs, files in os.walk(self.data_dir):
            for file_name in files:
                if file_name.endswith('.dcm'):
                    file_path = os.path.join(root, file_name)
                    try:
                        ds = pydicom.dcmread(file_path)
                        metadata = {
                            "File Name": file_name,
                            "Patient ID": ds.get("PatientID", "N/A"),
                            "Patient Name": ds.get("PatientName", "N/A"),
                            "Study Date": ds.get("StudyDate", "N/A"),
                            "Modality": ds.get("Modality", "N/A"),
                            "Institution Name": ds.get("InstitutionName", "N/A"),
                            "Body Part Examined": ds.get("BodyPartExamined", "N/A"),
                            "Study Description": ds.get("StudyDescription", "N/A"),
                            "Series Description": ds.get("SeriesDescription", "N/A"),
                        }
                        metadata_list.append(metadata)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        return metadata_list

    def save_metadata_to_excel(self, output_file: str) -> None:
        """
        Save the extracted DICOM metadata to an Excel file.

        Parameters:
        output_file (str): Path to save the Excel file.
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        metadata = self.extract_dicom_metadata()
        df = pd.DataFrame(metadata)
        df.to_excel(output_file, index=False)

    def _load_file(self, file_path: str) -> np.ndarray:
        """
        Load a DICOM file and return the pixel array.

        Parameters:
        file_path (str): Path to the file.

        Returns:
        np.ndarray: Loaded file data.
        """
        ds = pydicom.dcmread(file_path)
        return ds.pixel_array

    def _is_valid(self, file_data: np.ndarray) -> bool:
        """
        Check if the data is valid (e.g., not corrupted).

        Parameters:
        file_data (np.ndarray): Data to check.

        Returns:
        bool: True if the data is valid, False otherwise.
        """
        return file_data is not None
