from .base_dataset_manager import BaseDatasetManager
import numpy as np
import os
import pydicom
from models.dcgan import DCGAN
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

class GANManager(BaseDatasetManager):
    def __init__(self, root_dir, gan_generator_model):
        super().__init__(root_dir)
        self.gan_generator = gan_generator_model

    def augment_with_gan(self, num_samples_per_class=100):
        """Uses the DCGAN model to generate synthetic images and augment the dataset."""
        self.load_data()
        class_counts = self.metadata['class_name'].value_counts()
        max_count = class_counts.max()

        for class_name, count in class_counts.items():
            class_folder = os.path.join(self.root_dir, class_name)
        
            # Remove old synthetic images
            for file_name in os.listdir(class_folder):
                if file_name.startswith("synthetic_") and file_name.endswith(".dcm"):
                    os.remove(os.path.join(class_folder, file_name))
        
            # Calculate the number of synthetic images needed
            num_samples_needed = max_count - count

            # Skip generating images if no additional samples are needed
            if num_samples_needed <= 0:
                print(f"No synthetic images needed for class '{class_name}'. Skipping...")
                continue

            # Generate synthetic images
            print(f"Generating {num_samples_needed} synthetic images for class '{class_name}'...")
            synthetic_images = self.gan_generator.generate_images(num_samples_needed)

            # Save each generated image as a DICOM file
            for i, img in enumerate(synthetic_images):
                img = (img * 127.5 + 127.5).astype(np.uint8)
                save_path = os.path.join(class_folder, f"synthetic_{i}.dcm")
                self.save_image_as_dicom(img, save_path)


    def save_image_as_dicom(self, image_array, file_path):
        file_meta = pydicom.Dataset()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

        ds = pydicom.Dataset()
        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.SOPClassUID = SecondaryCaptureImageStorage
        ds.Rows, ds.Columns = image_array.shape[:2]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = image_array.tobytes()

        with open(file_path, 'wb') as f:
            f.write(b'\x00' * 128)
            f.write(b'DICM')
            pydicom.filewriter.write_file(f, ds)
