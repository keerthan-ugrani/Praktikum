# dataset_managers/gan_manager.py

import tensorflow as tf
import os
from keras.layers import Dense, Reshape, Flatten, LeakyReLU, Conv2DTranspose, Conv2D, BatchNormalization
from keras.models import Sequential
import numpy as np
from dataset_managers.base_dataset_manager import BaseDatasetManager
from keras import backend as K
import pydicom

class DatasetManagerWithGANs(BaseDatasetManager):
    def __init__(self, data_dir, target_image_size, latent_dim=100, gan_type="wgan-gp"):
        super().__init__(data_dir)
        self.target_image_size = target_image_size
        self.latent_dim = latent_dim
        self.gan_type = gan_type
        self.data = {}  # Dictionary to hold image data by class

        # Build GAN components
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def _load_file(self, file_path):
        """Load and resize a DICOM file to the target image size."""
        ds = pydicom.dcmread(file_path)
        img = ds.pixel_array.astype(np.float32)
        
        # Ensure the image has 3 dimensions (height, width, channels)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)  # Add channel dimension to make it (height, width, 1)
        
        # Resize to target size
        img = tf.image.resize(img, self.target_image_size).numpy()  # Resize to (28, 28, 1)
        return img

    def load_data(self):
        """Load images from the directory, resizing to target shape."""
        for root, dirs, files in os.walk(self.data_dir):
            class_name = os.path.basename(root)
            if class_name not in self.data:
                self.data[class_name] = []
            for file_name in files:
                if file_name.endswith('.dcm'):
                    file_path = os.path.join(root, file_name)
                    img = self._load_file(file_path)
                    self.data[class_name].append(img)

    def build_generator(self):
        model = Sequential([
            Dense(128 * (self.target_image_size[0] // 4) * (self.target_image_size[1] // 4), activation="relu", input_dim=self.latent_dim),
            Reshape((self.target_image_size[0] // 4, self.target_image_size[1] // 4, 128)),
            Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2D(1, kernel_size=7, activation='tanh', padding='same')
        ])
        return model

    def build_discriminator(self):
        model = Sequential([
            Conv2D(64, kernel_size=3, strides=2, input_shape=(self.target_image_size[0], self.target_image_size[1], 1), padding='same'),
            LeakyReLU(alpha=0.2),
            Flatten(),
            Dense(1)
        ])
        return model

    def get_minority_class(self):
        """Returns the class with the least number of images."""
        return min(self.data, key=lambda k: len(self.data[k]))
