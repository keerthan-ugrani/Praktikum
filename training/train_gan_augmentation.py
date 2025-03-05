from dataset_managers.gan_manager import GANManager
from models.dcgan import DCGAN
import cv2
import numpy as np

def resize_images(dataset, target_shape=(128, 128)):
    """Resize images in the dataset to the target shape."""
    resized_dataset = []
    for img in dataset:
        resized_img = cv2.resize(img, target_shape, interpolation=cv2.INTER_AREA)
        resized_dataset.append(resized_img)
    return np.array(resized_dataset)

def train_gan_augmentation(root_dir, dataset, latent_dim=100):
    """
    Train the GAN model and generate synthetic images.

    Args:
        root_dir (str): Root directory of the dataset.
        dataset (numpy.ndarray): Dataset to use for GAN training.
        latent_dim (int): Latent dimension for the GAN.

    Returns:
        DCGAN: The trained GAN model.
    """
    # Resize dataset to match DCGAN input shape
    dataset = resize_images(dataset, target_shape=(128, 128))

    # Ensure dataset has the correct shape
    if len(dataset.shape) == 3:
        dataset = np.expand_dims(dataset, axis=-1)  # Add channel dimension if necessary

    # Initialize the DCGAN model
    gan_model = DCGAN(image_shape=(128, 128, 1), latent_dim=latent_dim)

    # Train the GAN
    print("Training GAN...")
    gan_model.train(dataset, batch_size=32, epochs=5000, sample_interval=100)

    # Generate synthetic images and save them
    gan_manager = GANManager(root_dir, gan_generator_model=gan_model)
    gan_manager.augment_with_gan()

    return gan_model  # Return the trained GAN model
