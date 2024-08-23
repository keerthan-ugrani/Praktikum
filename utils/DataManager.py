import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DatasetManager:
    def __init__(self, data_dir: str):
        """
        Initialize the DatasetManager with a directory path.

        Parameters:
        data_dir (str): Path to the dataset directory.
        """
        self.data_dir = data_dir
        self.data = {}
        self.metadata = {}
        self.normalizer = None

    def load_data(self) -> None:
        """
        Load data from directories into a dictionary where each key is a class name 
        and each value is a list of data samples.
        """
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if os.path.isdir(class_path):
                self.data[class_dir] = []
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    self.data[class_dir].append(self._load_file(file_path))
        self.generate_metadata()

    def generate_metadata(self) -> None:
        """
        Generate metadata such as number of samples per class, total samples, and class balance.
        """
        class_counts = {class_name: len(files) for class_name, files in self.data.items()}
        self.metadata['class_counts'] = class_counts
        self.metadata['total_samples'] = sum(class_counts.values())
        self.metadata['class_balance'] = {
            class_name: count / self.metadata['total_samples']
            for class_name, count in class_counts.items()
        }

    def check_data_quality(self) -> Dict[str, List[str]]:
        """
        Perform data quality checks and return any issues found.

        Returns:
        Dict[str, List[str]]: Dictionary containing quality issues per class.
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
            flattened = [file.flatten() for file in files]
            normalized = self.normalizer.fit_transform(flattened)
            self.data[class_name] = [norm.reshape(files[0].shape) for norm in normalized]

    def split_data(self, test_size: float = 0.2) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Split the data into training and test sets.

        Parameters:
        test_size (float): Proportion of data to use for testing.

        Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: Training and test datasets.
        """
        train_data = {}
        test_data = {}
        for class_name, files in self.data.items():
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
        with h5py.File(output_file, 'w') as h5f:
            for class_name, files in self.data.items():
                group = h5f.create_group(class_name)
                for i, file_data in enumerate(files):
                    group.create_dataset(f'data_{i}', data=file_data)

    def _load_file(self, file_path: str) -> np.ndarray:
        """
        Placeholder for a method to load a file, customized based on data type.

        Parameters:
        file_path (str): Path to the file.

        Returns:
        np.ndarray: Loaded file data.
        """
        # Implement actual data loading logic based on the file format
        return np.random.rand(100, 100)  # Dummy data for example

    def _is_valid(self, file_data: np.ndarray) -> bool:
        """
        Check if the data is valid (e.g., not corrupted).

        Parameters:
        file_data (np.ndarray): Data to check.

        Returns:
        bool: True if the data is valid, False otherwise.
        """
        # Implement actual validity checks based on your data type
        return file_data is not None  # Simplified example


if __name__ == "__main__":
    # Example usage of the DatasetManager class
    data_manager = DatasetManager('/path/to/training/folder')
    data_manager.load_data()

    # Generate metadata and check data quality
    data_manager.generate_metadata()
    print("Metadata:", data_manager.metadata)
    quality_issues = data_manager.check_data_quality()
    print("Quality Issues:", quality_issues)

    # Preprocess data and visualize it
    data_manager.preprocess_data(method='minmax')
    data_manager.visualize_data_distribution()
    data_manager.visualize_sample(class_name='01', index=0)

    # Split the data and save to HDF5 format
    train_data, test_data = data_manager.split_data(test_size=0.3)
    data_manager.convert_to_h5('/path/to/output_file.h5')
