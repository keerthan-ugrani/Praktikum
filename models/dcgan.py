import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, LeakyReLU, BatchNormalization, Dropout
from keras.models import Model, Sequential
import numpy as np
import matplotlib.pyplot as plt
tf.config.run_functions_eagerly(True)

class DCGAN:
    def __init__(self, image_shape=(128, 128, 1), latent_dim=100):
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

        # Compile the generator for safe prediction usage
        self.generator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy")
        
        # Lists to store the loss values
        self.d_losses = []
        self.g_losses = []

    def build_generator(self):
        model = Sequential()
        model.add(Dense(8 * 8 * 128, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 128)))

        model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(32, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(self.image_shape[-1], kernel_size=4, padding="same", activation="tanh"))
        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=4, strides=2, padding="same", input_shape=self.image_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))

        model.add(Conv2D(64, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))
        return model

    def build_gan(self):
        # Compile discriminator
        self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy", metrics=["accuracy"])
        self.discriminator.trainable = False

        # Build the GAN model
        gan_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        generated_image = self.generator(gan_input)
        gan_output = self.discriminator(generated_image)
        gan_model = Model(gan_input, gan_output)
        gan_model.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy", run_eagerly=True)  # Debugging
        return gan_model

    def train(self, dataset, batch_size=32, epochs=100, sample_interval=100):
        real_label, fake_label = 0.9, 0.0
        for epoch in range(epochs):
            idx = np.random.randint(0, dataset.shape[0], batch_size)
            real_images = dataset[idx]

            # Ensure batch size is valid
            if real_images.shape[0] == 0:
                print(f"Warning: Empty batch at epoch {epoch}. Skipping...")
                continue

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_images = self.generator.predict(noise)

            # Train discriminator on real and fake images
            d_loss_real = self.discriminator.train_on_batch(real_images, np.ones((batch_size, 1)) * real_label)
            d_loss_fake = self.discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, np.ones((batch_size, 1)) * real_label)

            # Store losses
            self.d_losses.append(d_loss[0])
            self.g_losses.append(g_loss)

            # Print progress
            if epoch % sample_interval == 0:
                print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")

    def generate_images(self, n_images):
        """Generates synthetic images using the generator."""
        # Validate the number of images
        if n_images <= 0:
            raise ValueError(f"Cannot generate {n_images} images. Number of images must be greater than zero.")
    
        # Create noise vector
        noise = np.random.normal(0, 1, (n_images, self.latent_dim))
        print(f"Generating images with noise shape: {noise.shape}")  # Debugging noise shape
    
        # Ensure generator is compiled and ready
        if not hasattr(self.generator, 'optimizer'):
            self.generator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy")
            print("Generator was not compiled. Compiling now.")  # Debugging compilation status

        # Generate images
        try:
            synthetic_images = self.generator.predict(noise, verbose=1)
            return synthetic_images
        except Exception as e:
            print(f"Error during image generation: {e}")
            raise

    def plot_losses(self):
        """Plot the D and G losses over epochs."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.d_losses, label="Discriminator Loss")
        plt.plot(self.g_losses, label="Generator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("DCGAN Losses During Training")
        plt.legend()
        plt.show()
