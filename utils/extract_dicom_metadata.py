# extract_dicom_metadata.py

import os
import pydicom
import pandas as pd
from typing import List, Dict

class DICOMMetadataExtractor:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
    
    def extract_metadata(self) -> List[Dict[str, str]]:
        metadata_list = []
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.dcm'):
                    file_path = os.path.join(subdir, file)
                    try:
                        ds = pydicom.dcmread(file_path)
                        metadata = {
                            "File Name": file,
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

    def save_to_excel(self, metadata: List[Dict[str, str]], output_file: str) -> None:
        df = pd.DataFrame(metadata)
        df.to_excel(output_file, index=False)

def main():
    root_dir = '/path/to/dicom/files'
    output_file = 'dicom_metadata.xlsx'
    
    extractor = DICOMMetadataExtractor(root_dir)
    metadata = extractor.extract_metadata()
    extractor.save_to_excel(metadata, output_file)
    
    print(f"Metadata extraction complete. Excel file saved to {output_file}")

if __name__ == "__main__":
    main()
