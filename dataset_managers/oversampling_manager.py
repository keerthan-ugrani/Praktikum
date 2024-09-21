from base_dataset_manager import BaseDatasetManager
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class DatasetManagerWithOversampling(BaseDatasetManager):
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.augmentor = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                            shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    def oversample_data(self, target_class_size):
        for class_name, files in self.data.items():
            augmented_data = []
            if len(files) < target_class_size:
                num_to_add = target_class_size - len(files)
                for i in range(num_to_add):
                    img = files[i % len(files)]
                    img = np.expand_dims(img, axis=0)
                    augmented_img = next(self.augmentor.flow(img))[0].astype(np.uint8)
                    augmented_data.append(augmented_img)
                self.data[class_name].extend(augmented_data)
