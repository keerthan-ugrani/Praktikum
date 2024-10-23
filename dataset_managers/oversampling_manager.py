# dataset_managers/oversampling_manager.py

from dataset_managers.base_dataset_manager import BaseDatasetManager
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class DatasetManagerWithOversampling(BaseDatasetManager):
    def __init__(self, data_dir, target_image_size=(28, 28)):
        super().__init__(data_dir, target_image_size)
        self.augmentor = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                            shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    def oversample_data(self, target_class_size):
        """
        Oversamples minority classes and augments data using ImageDataGenerator.
        Args:
            target_class_size (int): The desired number of samples for each class.
        """
        for class_name, files in self.data.items():
            augmented_data = []
            if len(files) < target_class_size:
                num_to_add = target_class_size - len(files)
                for i in range(num_to_add):
                    img = files[i % len(files)]
                    
                    # Ensure the image is 4D: (1, height, width, channels)
                    if len(img.shape) == 2:  # For grayscale images (height, width)
                        img = np.expand_dims(img, axis=-1)  # Add channel dimension
                    img = np.expand_dims(img, axis=0)  # Add batch dimension
                    
                    # Generate augmented image
                    augmented_img = next(self.augmentor.flow(img, batch_size=1))[0].astype(np.uint8)
                    
                    # Remove batch dimension before appending
                    augmented_img = np.squeeze(augmented_img)
                    augmented_data.append(augmented_img)
                
                self.data[class_name].extend(augmented_data)
        
        self.generate_metadata()  # Update metadata after oversampling
