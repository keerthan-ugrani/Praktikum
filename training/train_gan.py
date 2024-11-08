# training/train_gan.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

class ADA:
    def __init__(self, probability=0.6):
        """Initialize ADA with a probability for each augmentation."""
        self.probability = probability

    def apply(self, images):
        """Apply augmentations with specified probability."""
        if np.random.rand() < self.probability:
            images = tf.image.random_flip_left_right(images)
        if np.random.rand() < self.probability:
            images = tf.image.random_brightness(images, max_delta=0.1)
        if np.random.rand() < self.probability:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images


def visualize_images(images, epoch, n=3, save_dir='generated_images'):
    """Visualize n generated images from a batch and optionally save them."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')  # Rescale to [0, 255]
        plt.axis('off')
    
    # Save the image grid
    plt.suptitle(f"Generated Images at Epoch {epoch}")
    plt.savefig(os.path.join(save_dir, f"generated_epoch_{epoch}.png"))
    plt.show()


def compute_gradient_penalty(discriminator, real_images, fake_images):
    """Calculate the gradient penalty for WGAN-GP."""
    batch_size = real_images.shape[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_images + (1 - alpha) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated)
    grads = tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp
def initialize_model_weights(model):
    """Initialize weights with normal distribution."""
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose, tf.keras.layers.Dense)):
            layer.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

def visualize_images(images, epoch, n=3, save_dir='generated_images'):
    """Visualize n generated images from a batch and optionally save them."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')  # Rescale to [0, 255]
        plt.axis('off')
    
    # Save the image grid
    plt.suptitle(f"Generated Images at Epoch {epoch}")
    plt.savefig(os.path.join(save_dir, f"generated_epoch_{epoch}.png"))
    plt.show()

def compute_gradient_penalty(discriminator, real_images, fake_images):
    """Calculate the gradient penalty for WGAN-GP."""
    batch_size = real_images.shape[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_images + (1 - alpha) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated)
    grads = tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

def train_gan_with_visualization(generator, discriminator, dataset, config):
    # Initialize weights for stability
    initialize_model_weights(generator)
    initialize_model_weights(discriminator)
    
    latent_dim = config['latent_dim']
    batch_size = config['batch_size']
    epochs = config['gan_epochs']
    
    # Learning rates
    g_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)
    d_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)
    
    lambda_gp = config.get('lambda_gp', 10)  # Reduced gradient penalty weight
    n = 3  # Number of images to visualize

    for epoch in range(epochs):
        d_loss, g_loss = 0, 0  # Initialize losses at the start of each epoch

        for real_images in dataset.batch(batch_size):
            print("Real images batch shape:", real_images.shape)  # Diagnostic print
            print("Real images batch sample values (before scaling):", real_images[0, :5, :5])  # Sample values
            
            # Scale and confirm scaling
            real_images = (real_images - 127.5) / 127.5
            print("Real images batch sample values (after scaling):", real_images[0, :5, :5])  # Sample values

            # Discriminator training
            for _ in range(1):
                noise = tf.random.normal([batch_size, latent_dim])
                fake_images = generator(noise, training=True)

                with tf.GradientTape() as tape:
                    d_loss_real = -tf.reduce_mean(discriminator(real_images, training=True))
                    d_loss_fake = tf.reduce_mean(discriminator(fake_images, training=True))
                    d_loss = d_loss_real + d_loss_fake

                    # Gradient penalty
                    gp = compute_gradient_penalty(discriminator, real_images, fake_images)
                    d_loss += lambda_gp * gp

                d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
                d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

            # Print discriminator gradients for debugging
            print("Discriminator gradient norm:", np.mean([tf.norm(g).numpy() for g in d_grads if g is not None]))

            # Generator training
            with tf.GradientTape() as tape:
                noise = tf.random.normal([batch_size, latent_dim])
                fake_images = generator(noise, training=True)
                g_loss = -tf.reduce_mean(discriminator(fake_images, training=True))

            g_grads = tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

            # Print generator gradients for debugging
            print("Generator gradient norm:", np.mean([tf.norm(g).numpy() for g in g_grads if g is not None]))

        # Print losses and visualize images periodically
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Discriminator Loss: {d_loss:.4f}, Generator Loss: {g_loss:.4f}")
            noise = tf.random.normal([n, latent_dim])  # Generate noise to visualize images
            generated_images = generator(noise, training=False)
            visualize_images(generated_images.numpy(), epoch, n=n)  # Pass n for visualization
