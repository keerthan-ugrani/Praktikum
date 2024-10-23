import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import pydicom

class DatasetManager:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.data = {}
        self.metadata = {}
        self.normalizer = None

    def load_data(self) -> None:
        for root, dirs, files in os.walk(self.data_dir):
            for file_name in files:
                if file_name.endswith('.dcm'):
                    file_path = os.path.join(root, file_name)
                    class_name = os.path.basename(root)
                    if class_name not in self.data:
                        self.data[class_name] = []
                    self.data[class_name].append(self._load_file(file_path))
        self.generate_metadata()

    def generate_metadata(self) -> None:

        class_counts = {class_name: len(files) for class_name, files in self.data.items()}
        self.metadata = {
            'class_counts': class_counts,
            'total_samples': sum(class_counts.values()),
            'class_balance': {
                class_name: count / sum(class_counts.values()) for class_name, count in class_counts.items()
            }
        }

    def check_data_quality(self) -> Dict[str, List[str]]:

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

    def split_data(self, test_size: float = 0.3) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        
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
        
        class_counts = self.metadata.get('class_counts', {})
        sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.show()

    def visualize_sample(self, class_name: str, index: int = 0) -> None:
        
        if class_name in self.data:
            sample = self.data[class_name][index]
            plt.imshow(sample, cmap='gray')
            plt.title(f"Class: {class_name}, Index: {index}")
            plt.show()
        else:
            print(f"Class '{class_name}' not found.")

    def convert_to_h5(self, output_file: str) -> None:
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with h5py.File(output_file, 'w') as h5f:
            for class_name, files in tqdm(self.data.items(), desc="Saving to HDF5"):
                group = h5f.create_group(class_name)
                for i, file_data in enumerate(files):
                    group.create_dataset(f'data_{i}', data=file_data)

    def extract_dicom_metadata(self) -> List[Dict[str, str]]:
        
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
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        metadata = self.extract_dicom_metadata()
        df = pd.DataFrame(metadata)
        df.to_excel(output_file, index=False)

    def _load_file(self, file_path: str) -> np.ndarray:
        
        ds = pydicom.dcmread(file_path)
        return ds.pixel_array

    def _is_valid(self, file_data: np.ndarray) -> bool:
        
        return file_data is not None  


if __name__ == "__main__":
    data_manager = DatasetManager('../dataset/TCGA-KIRP')

    # Load the data
    data_manager.load_data()

    # Generate metadata and check data quality
    data_manager.generate_metadata()
    print("Metadata:", data_manager.metadata)
    quality_issues = data_manager.check_data_quality()
    print("Quality Issues:", quality_issues)

    # Preprocess the data using the specified method (standardization or min-max scaling)
    data_manager.preprocess_data(method='minmax')

    # Visualize the data distribution
    data_manager.visualize_data_distribution()

    # Visualize a sample
    data_manager.visualize_sample(class_name='01', index=0)

    # Split the data and save to HDF5 format
    train_data, test_data = data_manager.split_data(test_size=0.3)
    data_manager.convert_to_h5('/TCGA-KIRP/output_file.h5')

    # Extract and save DICOM metadata to Excel
    data_manager.save_metadata_to_excel('/TCGA-KIRP/dicom_metadata.xlsx')
