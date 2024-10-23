# dataset_managers/gan_manager.py

import tensorflow as tf
from keras.layers import Dense, Reshape, Flatten, LeakyReLU, Conv2DTranspose, Conv2D, BatchNormalization
from keras.models import Sequential
import numpy as np
from dataset_managers.base_dataset_manager import BaseDatasetManager
from keras import backend as K

class DatasetManagerWithGANs(BaseDatasetManager):
    def __init__(self, data_dir, target_image_size, latent_dim=100, gan_type="standard"):
        super().__init__(data_dir, target_image_size)
        self.latent_dim = latent_dim
        self.gan_type = gan_type  # Set the GAN type ('standard' or 'wgan-gp')
        self.lambda_gp = 10  # Gradient penalty for WGAN-GP

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        if self.gan_type == "standard":
            self.gan = self.compile_standard_gan()
        elif self.gan_type == "wgan-gp":
            self.gan = None  # WGAN-GP uses a different training loop

    def build_generator(self):
        model = Sequential()
        model.add(Dense(128 * (self.target_image_size[0] // 4) * (self.target_image_size[1] // 4), activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((self.target_image_size[0] // 4, self.target_image_size[1] // 4, 128)))

        model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(1, kernel_size=7, activation='tanh', padding='same'))  # Output image [-1, 1]
        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=(self.target_image_size[0], self.target_image_size[1], 1), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1))  # No activation for WGAN-GP, sigmoid for Standard GAN
        return model

    def compile_standard_gan(self):
        # Standard GAN uses binary crossentropy loss and Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        self.discriminator.trainable = False  # Freeze discriminator when training generator
        gan_input = tf.keras.Input(shape=(self.latent_dim,))
        generated_image = self.generator(gan_input)
        gan_output = self.discriminator(generated_image)
        gan = tf.keras.Model(gan_input, gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=optimizer)
        return gan

    def gradient_penalty(self, real_images, fake_images):
        """
        Compute the gradient penalty for WGAN-GP.
        Ensures that the real_images and fake_images have the same shape.
        """
        batch_size = real_images.shape[0]

        # Ensure that real_images and fake_images have the same shape
        real_images = tf.reshape(real_images, [batch_size, 28, 28, 1])  # Reshape real_images to match fake_images
        fake_images = tf.reshape(fake_images, [batch_size, 28, 28, 1])  # Ensure fake_images are in the right shape

        # Alpha needs to be broadcastable to the shape of real_images and fake_images
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)

        # Interpolation
        interpolated_images = alpha * real_images + (1 - alpha) * fake_images

        with tf.GradientTape() as tape:
            tape.watch(interpolated_images)
            pred = self.discriminator(interpolated_images)

        grads = tape.gradient(pred, interpolated_images)

        # Compute the L2 norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))

        # Gradient penalty
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp


    def train_gan(self, real_images, epochs=1000, batch_size=32, d_steps=5):
        real_images = (real_images.astype('float32') - 127.5) / 127.5  # Normalize real images to [-1, 1]

        if self.gan_type == "standard":
            # Train Standard GAN
            self.train_standard_gan(real_images, epochs, batch_size)
        elif self.gan_type == "wgan-gp":
            # Train WGAN-GP
            self.train_wgan_gp(real_images, epochs, batch_size, d_steps)

    def train_standard_gan(self, real_images, epochs, batch_size):
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Randomly sample a batch of real images
            idx = np.random.randint(0, real_images.shape[0], batch_size)
            real_imgs = real_images[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_images = self.generator.predict(noise)

            # Train the discriminator with real and fake images
            d_loss_real = self.discriminator.train_on_batch(real_imgs, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_images, fake_labels)

            # Train the generator to fool the discriminator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, real_labels)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} - D Loss Real: {d_loss_real}, D Loss Fake: {d_loss_fake}, G Loss: {g_loss}")

    def train_wgan_gp(self, real_images, epochs, batch_size, d_steps):
        g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)

        for epoch in range(epochs):
            for _ in range(d_steps):
                idx = np.random.randint(0, real_images.shape[0], batch_size)
                real_imgs = real_images[idx]
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                fake_imgs = self.generator.predict(noise)

                with tf.GradientTape() as tape:
                    real_preds = self.discriminator(real_imgs)
                    fake_preds = self.discriminator(fake_imgs)
                    d_loss = tf.reduce_mean(fake_preds) - tf.reduce_mean(real_preds)

                    gp = self.gradient_penalty(real_imgs, fake_imgs)
                    d_loss += self.lambda_gp * gp

                d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
                d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_imgs = self.generator(noise)
                fake_preds = self.discriminator(fake_imgs)
                g_loss = -tf.reduce_mean(fake_preds)

            g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

            if epoch % 100 == 0:
                print(f"WGAN-GP Epoch {epoch}/{epochs} - D Loss: {d_loss}, G Loss: {g_loss}")

    def get_minority_class(self):
        # Get the class with the fewest samples
        class_counts = self.metadata['class_counts']
        return min(class_counts, key=class_counts.get)
