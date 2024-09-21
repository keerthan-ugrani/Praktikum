from base_dataset_manager import BaseDatasetManager
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, Conv2DTranspose, Conv2D
from tensorflow.keras.models import Sequential
import numpy as np

class DatasetManagerWithGANs(BaseDatasetManager):
    def __init__(self, data_dir, latent_dim=100):
        super().__init__(data_dir)
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.compile_gan()

    def build_generator(self):
        model = Sequential()
        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(1, kernel_size=7, activation='tanh', padding='same'))
        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model

    def compile_gan(self):
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.discriminator.trainable = False
        gan_input = tf.keras.Input(shape=(self.latent_dim,))
        generated_image = self.generator(gan_input)
        gan_output = self.discriminator(generated_image)
        gan = tf.keras.Model(gan_input, gan_output)
        gan.compile(loss='binary_crossentropy', optimizer='adam')
        return gan

    def train_gan(self, real_images, epochs=10000, batch_size=32):
        real_images = (real_images.astype(np.float32) - 127.5) / 127.5
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, real_images.shape[0], batch_size)
            real_imgs = real_images[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_images = self.generator.predict(noise)
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_imgs, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_images, fake_labels)

            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, real_labels)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} - D Loss: {d_loss_real}, G Loss: {g_loss}")
