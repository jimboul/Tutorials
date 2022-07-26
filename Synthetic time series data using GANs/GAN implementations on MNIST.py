from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Embedding, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.merge import _Merge
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
from keras.preprocessing import image
import keras.backend as K

from functools import partial

import matplotlib.pyplot as plt
#%matplotlib inline - used with Jupyter Notebook

import sys
import time

import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DCGAN():
    '''
        A class for a Deep Convolutional Generative Adversarial Network.
    
    '''
    
    def __init__(self):
        # Properties initialisation
        self._rows = 28
        self._cols = 28
        self._channels = 1
        self._input_shape = (self._rows, self._cols, self._channels)
        self._latent_dim = 100 # shape of the noise
        
        opt = Adam(0.0002, 0.5)
                
        # Get a discriminator
        self._discriminator = self.get_discriminator()
        self._discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        # Get a generator
        self._generator = self.get_generator()
        
        # Feed the generator with noise to generate adversarial input/images
        noise = Input(shape=(self._latent_dim,))
        adv_input = self._generator(noise)
        
        # The discriminator will not be trained in the full GAN model
        self._discriminator.trainable = False
        
        # Feed the discriminator with the adversarial input
        validity = self._discriminator(adv_input)
        
        # Get the full GAN model (generator -> discriminator)
        self._full_gan = Model(noise, validity)
        self._full_gan.compile(loss='binary_crossentropy', optimizer=opt)
    
    def get_discriminator(self):
        '''
            This function builds the architecture of the discriminator model (or validator)
            In our case, the discriminator is a deep CNN-based neural architecture and is 
            responsible for the validation of the given image (legit or fake).
            
        '''
        
        model = Sequential()
        
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self._input_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        
        disc_input = Input(shape=self._input_shape)
        validity = model(disc_input)
        
        return Model(disc_input, validity)
    
    
    def get_generator(self):
        '''
            This function builds the architecture of the generator model (or adversary)
            In our case, the generator is a deep CNN-based neural architecture and is 
            responsible for the generating adversarial input to the discriminator.
            
        '''
        
        model = Sequential()
        
        # The number of units of the fully connected layer is multiplied by the size of the downsampled image (7x7 pixels)
        model.add(Dense(128*7*7, activation="relu", input_dim=self._latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation("relu"))
        
        model.add(Conv2D(self._channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))
        
        noise = Input(shape=(self._latent_dim,))
        adv_input = model(noise)
        
        return Model(noise, adv_input)
    
    
    def predictions(self, epoch):
        '''
            On a per-epoch basis prediction. The GAN predicts the new generated images 
            and generates new samples in this way.
        
        '''
        
        num_rows = 5
        num_cols = 5
        
        # Sample random noise
        noise = np.random.normal(0, 1, (num_rows*num_cols, self._latent_dim))
        
        # Get images from the generator
        gen_images = self._generator.predict(noise)
        
        # Rescale images to 0-1
        gen_images = gen_images/2 + 0.5
        
        # Save the generated images
        fig, axs = plt.subplots(num_rows, num_cols)
        counter = 0
        for i in range(num_rows):
            for j in range(num_cols):
                axs[i,j].imshow(gen_images[counter, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                counter += 1
        
        fig.savefig('mnist_dcgan_sample_%d.png' % epoch)
        plt.close()
    
    
    def plot_results(self, disc_losses, gen_losses, model_accuracies):
        '''
            Plot the results of the training of the GAN
        
        '''
        
        # Plot losses
        plt.figure()
        plt.plot(disc_losses)
        plt.plot(gen_losses)
        plt.title('DCGAN loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Discriminator', 'Generator'], loc='upper left')
        plt.savefig('dcgan_losses.png')
        
        # Plot accuracy
        plt.figure()
        plt.plot(model_accuracies)
        plt.title('DCGAN accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['DCGAN'], loc='upper left')
        plt.savefig('dcgan_accuracy.png')
    
    
    def run(self, epochs, batch_size, interval_size, verbose=0):
        '''
            This function is responsible for loading and manipulating the data as well as training the
            Generative Adversarial Network.
            
            epochs: The number of epochs which the model will use for its training.
            batch_size: The batch size for the (batch) training of the model.
            save_images: A flag indicating whether we want to save the generated images or not.
        
        '''
        
        # Load the MNIST dataset from Keras's repository
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)
        
        # Adversarial ground truths
        valid = np.ones((batch_size, 1)) # real are labelled as one
        adversaries = np.zeros((batch_size, 1)) # adversaries/fake are labelled as zero
        
        disc_losses = np.zeros((epochs,))
        gen_losses = np.zeros((epochs,))
        model_accuracies = np.zeros((epochs,))
        
        start = time.time()
        
        for epoch in range(epochs):
            
            # Train the discriminator
            '''
                The goal of the discriminator is to detect which input is legit or adversary.
                The discriminator must not be fooled by the adversary generator model at all times.
            
            '''
            
            # Random sampling of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            images = X_train[idx]
            
            # Sample noise
            noise = np.random.normal(0, 1, (batch_size, self._latent_dim))
            
            # Generate noisy/adversarial images (input to the discriminator/validator)
            adv_images = self._generator.predict(noise)
            
            # Train the discriminator and get the loss from the real and adversary input
            disc_loss_real = self._discriminator.train_on_batch(images, valid)
            disc_loss_adv = self._discriminator.train_on_batch(adv_images, adversaries)
            
            # Calculate loss of discriminator
            disc_loss = np.add(disc_loss_real, disc_loss_adv) / 2
            
            # Train the generator
            '''
                The goal of the generator is to fool successfully the discriminator.
                The ultimate target is to have a quite robust discriminator.
            
            '''
            
            # Calculate the loss of the generator
            gen_loss = self._full_gan.train_on_batch(noise, valid)
            
            # Save the training history
            disc_losses[epoch] = disc_loss[0]
            gen_losses[epoch] = gen_loss
            model_accuracies[epoch] = disc_loss[1]
            
            if verbose == 1:
                print ("%d [Discriminator loss: %f, Discriminator accuracy: %.2f%%] Generator loss: %f" % (epoch, disc_loss[0], 100*disc_loss[1], gen_loss))
            
            print('Epoch:', epoch+1)
            
            # Batch predictions (according to interval size)
            if epoch % interval_size == 0:
                self.predictions(epoch)
        
        print('Time elapsed during training: ', time.time()-start, 'seconds.')
        
        self.plot_results(disc_losses, gen_losses, model_accuracies)
        
        print('Mean discriminator loss:', np.mean(disc_loss[0]))
        print('Mean discriminator accuracy:', np.mean(disc_loss[1]))
        print('Mean generator loss:', np.mean(gen_loss))


class DC2GAN():
    '''
        A class for a conditional Deep Convolutional Generative Adversarial Network.
    
    '''
    
    def __init__(self):
        # Properties initialisation
        self._rows = 28
        self._cols = 28
        self._channels = 1
        self._input_shape = (self._rows, self._cols, self._channels)
        self._latent_dim = 100 # shape of the noise
        self._num_classes = 10
        
        opt = Adam(0.0001)
                
        # Get a discriminator
        disc_input = Input(shape=self._input_shape)
        disc_conditioned_labels = Input(shape=(self._num_classes,)) # 10 different classes existing in the dataset
        self._discriminator, self._discriminator_output_layer = self.get_discriminator(disc_input, disc_conditioned_labels)
        self._discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        # Get a generator
        gen_input = Input(shape=(self._latent_dim,))
        gen_conditioned_labels = Input(shape=(self._num_classes,))
        self._generator, self._generator_output_layer = self.get_generator(gen_input, gen_conditioned_labels)
        
        # Feed the generator with noise and the target label to generate adversarial input/images
        noise = Input(shape=(self._latent_dim,))
        adv_input = self._generator([noise, gen_conditioned_labels])
        
        # The discriminator will not be trained in the full GAN model
        self._discriminator.trainable = False
        
        # Feed the discriminator with the adversarial input
        validity = self._discriminator([adv_input, disc_conditioned_labels])
        
        # Get the full GAN model (generator -> discriminator)
        self._full_gan = Model([noise, gen_conditioned_labels, disc_conditioned_labels], validity)
        self._full_gan.compile(loss='binary_crossentropy', optimizer=opt)
    
    def get_discriminator(self, input_layer, conditioned_layer):
        '''
            This function builds the architecture of the discriminator model (or validator)
            In our case, the discriminator is a deep CNN-based neural architecture and is 
            responsible for the validation of the given image (legit or fake).
            
        '''
        
        hidden_layer = Conv2D(128, kernel_size=3, strides=2, input_shape=self._input_shape, padding="same")(input_layer)
        hidden_layer = LeakyReLU(alpha=0.2)(hidden_layer)
        hidden_layer = Dropout(0.25)(hidden_layer)
        hidden_layer = Conv2D(128, kernel_size=3, strides=2, padding="same")(hidden_layer)
        hidden_layer = ZeroPadding2D(padding=((0,1),(0,1)))(hidden_layer)
        hidden_layer = BatchNormalization(momentum=0.9)(hidden_layer)
        hidden_layer = LeakyReLU(alpha=0.2)(hidden_layer)
        hidden_layer = Dropout(0.25)(hidden_layer)

        hidden_layer = Conv2D(128, kernel_size=3, strides=2, padding="same")(hidden_layer)
        hidden_layer = BatchNormalization(momentum=0.9)(hidden_layer)
        hidden_layer = LeakyReLU(alpha=0.2)(hidden_layer)
        hidden_layer = Dropout(0.25)(hidden_layer)

        hidden_layer = Conv2D(128, kernel_size=3, strides=1, padding="same")(hidden_layer)
        hidden_layer = BatchNormalization(momentum=0.9)(hidden_layer)
        hidden_layer = LeakyReLU(alpha=0.2)(hidden_layer)
        hidden_layer = Dropout(0.25)(hidden_layer)

        hidden_layer = Flatten()(hidden_layer)

        merged_layer = Concatenate()([hidden_layer, conditioned_layer])

        hidden_layer = Dense(512, activation='relu')(merged_layer)
        output_layer = Dense(1, activation='sigmoid')(hidden_layer)
        
        return Model([input_layer, conditioned_layer], output_layer), output_layer
    
    
    def get_generator(self, input_layer, conditioned_layer):
        '''
            This function builds the architecture of the generator model (or adversary)
            In our case, the generator is a deep CNN-based neural architecture and is 
            responsible for the generating adversarial input to the discriminator.
            
        '''
        
        merged_input = Concatenate()([input_layer, conditioned_layer])

        # The number of units of the fully connected layer is multiplied by the size of the downsampled image (7x7)
        hidden_layer = Dense(128*7*7, activation="relu", input_dim=self._latent_dim)(merged_input)
        hidden_layer = Reshape((7, 7, 128))(hidden_layer)
        hidden_layer = UpSampling2D()(hidden_layer)
        hidden_layer = Conv2D(128, kernel_size=3, padding="same")(hidden_layer)
        hidden_layer = BatchNormalization(momentum=0.9)(hidden_layer)
        hidden_layer = Activation("relu")(hidden_layer)
        hidden_layer = UpSampling2D()(hidden_layer)
        hidden_layer = Conv2D(128, kernel_size=3, padding="same")(hidden_layer)
        hidden_layer = BatchNormalization(momentum=0.9)(hidden_layer)
        hidden_layer = Activation("relu")(hidden_layer)
        hidden_layer = Conv2D(self._channels, kernel_size=3, padding="same")(hidden_layer)
        output_layer = Activation("tanh")(hidden_layer)

        return Model([input_layer, conditioned_layer], output_layer), output_layer
    
    
    def predictions(self, epoch):
        '''
            On a per-epoch basis prediction. The GAN predicts the new generated images 
            and generates new samples in this way.
        
        '''
        
        digits = [0,1,2,3,4,5,6,7,8,9]
        
        fig, axs = plt.subplots(5, 6, figsize=(10,6))
        plt.subplots_adjust(hspace=0.3, wspace=0.1)
        
        for classlabel in range(10):
            row = int(classlabel/2)
            coloffset = (classlabel % 2)*3
            labels = to_categorical([classlabel]*3, self._num_classes)
            noise = np.random.normal(0, 1, (3, self._latent_dim))
            gen_images = self._generator.predict([noise, labels])
            
            for i in range(3):
                img = image.array_to_img(gen_images[i], scale=True)
                axs[row,i+coloffset].imshow(img)
                axs[row,i+coloffset].axis('off')
                if i ==1:
                    axs[row,i+coloffset].set_title("Digit: %d" % digits[classlabel])
                
        fig.savefig('mnist_dc2gan_sample_%d.png' % epoch)
        plt.close()
        
    
    def plot_results(self, disc_losses, gen_losses, model_accuracies):
        '''
            Plot the results of the training of the GAN
        
        '''
        
        # Plot losses
        plt.figure()
        plt.plot(disc_losses)
        plt.plot(gen_losses)
        plt.title('DC2GAN loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Discriminator', 'Generator'], loc='upper left')
        plt.savefig('dc2gan_losses.png')
        
        # Plot accuracy
        plt.figure()
        plt.plot(model_accuracies)
        plt.title('DC2GAN accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['DC2GAN'], loc='upper left')
        plt.savefig('dc2gan_accuracy.png')
    
    
    def run(self, epochs, batch_size, interval_size, verbose=0):
        '''
            This function is responsible for loading and manipulating the data as well as training the
            Generative Adversarial Network.
            
            epochs: The number of epochs which the model will use for its training.
            batch_size: The batch size for the (batch) training of the model.
            save_images: A flag indicating whether we want to save the generated images or not.
        
        '''
        
        # Load the MNIST dataset from Keras's repository
        (X_train, y_train), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train.astype(np.float32)
        X_train = X_train / 127.5 - 1
        X_train = np.expand_dims(X_train, axis=3)
#         y_train = y_train.reshape(-1,1)
        y_train = to_categorical(y_train, self._num_classes)
        
        # Adversarial ground truths
        valid = np.ones((batch_size, 1)) # real are labelled as one
        adversaries = np.zeros((batch_size, 1)) # adversaries/fake are labelled as zero
        
        disc_losses = np.zeros((epochs,))
        gen_losses = np.zeros((epochs,))
        model_accuracies = np.zeros((epochs,))
        
        start = time.time()
        
        for epoch in range(epochs):
            
            # Train the discriminator
            '''
                The goal of the discriminator is to detect which input is legit or adversary.
                The discriminator must not be fooled by the adversary generator model at all times.
            
            '''
            
            # Random sampling of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            images = X_train[idx]
            labels = y_train[idx]
            
            # Sample noise
            noise = np.random.normal(0, 1, (batch_size, self._latent_dim))
            
            # Generate noisy/adversarial images (input to the discriminator/validator)
            adv_images = self._generator.predict([noise, labels])
            
            # Train the discriminator and get the loss from the real and adversary input
            disc_loss_real = self._discriminator.train_on_batch([images, labels], valid)
            disc_loss_adv = self._discriminator.train_on_batch([adv_images, labels], adversaries)
            
            # Calculate loss of discriminator
            disc_loss = np.add(disc_loss_real, disc_loss_adv) / 2
            
            # Train the generator
            '''
                The goal of the generator is to fool successfully the discriminator.
                The ultimate target is to have a quite robust discriminator.
            
            '''
            
            # Calculate the loss of the generator
            conditioned_labels = np.random.randint(0, self._num_classes, batch_size)
            conditioned_labels = to_categorical(conditioned_labels, self._num_classes)
            gen_loss = self._full_gan.train_on_batch([noise, conditioned_labels, conditioned_labels], valid)
            
            # Save the training history
            disc_losses[epoch] = disc_loss[0]
            gen_losses[epoch] = gen_loss
            model_accuracies[epoch] = disc_loss[1]
            
            if verbose == 1:
                print ("%d [Discriminator loss: %f, Discriminator accuracy: %.2f%%] Generator loss: %f" % (epoch, disc_loss[0], 100*disc_loss[1], gen_loss))
            
            print('Epoch:', epoch+1)
            
            # Batch predictions (according to interval size)
            if epoch % interval_size == 0:
                self.predictions(epoch)
        
        print('Time elapsed during training: ', time.time()-start, 'seconds.')
        
        self.plot_results(disc_losses, gen_losses, model_accuracies)
        
        print('Mean discriminator loss:', np.mean(disc_loss[0]))
        print('Mean discriminator accuracy:', np.mean(disc_loss[1]))
        print('Mean generator loss:', np.mean(gen_loss))


class DC2GANExperienceReplay():
    '''
        A class for a conditional Deep Convolutional Generative Adversarial Network.
        This method implements a (sampling) technique inspired by Reinforcement Learning, 
        called Experience Replay.
    
    '''
    
    def __init__(self):
        # Properties initialisation
        self._rows = 28
        self._cols = 28
        self._channels = 1
        self._input_shape = (self._rows, self._cols, self._channels)
        self._latent_dim = 100 # shape of the noise
        self._num_classes = 10
        
        opt = Adam(0.0001)
                
        # Get a discriminator
        disc_input = Input(shape=self._input_shape)
        disc_conditioned_labels = Input(shape=(self._num_classes,)) # 10 different classes existing in the dataset
        self._discriminator, self._discriminator_output_layer = self.get_discriminator(disc_input, disc_conditioned_labels)
        self._discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        # Get a generator
        gen_input = Input(shape=(self._latent_dim,))
        gen_conditioned_labels = Input(shape=(self._num_classes,))
        self._generator, self._generator_output_layer = self.get_generator(gen_input, gen_conditioned_labels)
        
        # Feed the generator with noise and the target label to generate adversarial input/images
        noise = Input(shape=(self._latent_dim,))
        adv_input = self._generator([noise, gen_conditioned_labels])
        
        # The discriminator will not be trained in the full GAN model
        self._discriminator.trainable = False
        
        # Feed the discriminator with the adversarial input
        validity = self._discriminator([adv_input, disc_conditioned_labels])
        
        # Get the full GAN model (generator -> discriminator)
        self._full_gan = Model([noise, gen_conditioned_labels, disc_conditioned_labels], validity)
        self._full_gan.compile(loss='binary_crossentropy', optimizer=opt)
    
    def get_discriminator(self, input_layer, conditioned_layer):
        '''
            This function builds the architecture of the discriminator model (or validator)
            In our case, the discriminator is a deep CNN-based neural architecture and is 
            responsible for the validation of the given image (legit or fake).
            
        '''
        
        hidden_layer = Conv2D(128, kernel_size=3, strides=2, input_shape=self._input_shape, padding="same")(input_layer)
        hidden_layer = LeakyReLU(alpha=0.2)(hidden_layer)
        hidden_layer = Dropout(0.25)(hidden_layer)
        hidden_layer = Conv2D(128, kernel_size=3, strides=2, padding="same")(hidden_layer)
        hidden_layer = ZeroPadding2D(padding=((0,1),(0,1)))(hidden_layer)
        hidden_layer = BatchNormalization(momentum=0.9)(hidden_layer)
        hidden_layer = LeakyReLU(alpha=0.2)(hidden_layer)
        hidden_layer = Dropout(0.25)(hidden_layer)

        hidden_layer = Conv2D(128, kernel_size=3, strides=2, padding="same")(hidden_layer)
        hidden_layer = BatchNormalization(momentum=0.9)(hidden_layer)
        hidden_layer = LeakyReLU(alpha=0.2)(hidden_layer)
        hidden_layer = Dropout(0.25)(hidden_layer)

        hidden_layer = Conv2D(128, kernel_size=3, strides=1, padding="same")(hidden_layer)
        hidden_layer = BatchNormalization(momentum=0.9)(hidden_layer)
        hidden_layer = LeakyReLU(alpha=0.2)(hidden_layer)
        hidden_layer = Dropout(0.25)(hidden_layer)

        hidden_layer = Flatten()(hidden_layer)

        merged_layer = Concatenate()([hidden_layer, conditioned_layer])

        hidden_layer = Dense(512, activation='relu')(merged_layer)
        output_layer = Dense(1, activation='sigmoid')(hidden_layer)
        
        return Model([input_layer, conditioned_layer], output_layer), output_layer
    
    
    def get_generator(self, input_layer, conditioned_layer):
        '''
            This function builds the architecture of the generator model (or adversary)
            In our case, the generator is a deep CNN-based neural architecture and is 
            responsible for the generating adversarial input to the discriminator.
            
        '''
        
        merged_input = Concatenate()([input_layer, conditioned_layer])

        # The number of units of the fully connected layer is multiplied by the size of the downsampled image (7x7)
        hidden_layer = Dense(128*7*7, activation="relu", input_dim=self._latent_dim)(merged_input)
        hidden_layer = Reshape((7, 7, 128))(hidden_layer)
        hidden_layer = UpSampling2D()(hidden_layer)
        hidden_layer = Conv2D(128, kernel_size=3, padding="same")(hidden_layer)
        hidden_layer = BatchNormalization(momentum=0.9)(hidden_layer)
        hidden_layer = Activation("relu")(hidden_layer)
        hidden_layer = UpSampling2D()(hidden_layer)
        hidden_layer = Conv2D(128, kernel_size=3, padding="same")(hidden_layer)
        hidden_layer = BatchNormalization(momentum=0.9)(hidden_layer)
        hidden_layer = Activation("relu")(hidden_layer)
        hidden_layer = Conv2D(self._channels, kernel_size=3, padding="same")(hidden_layer)
        output_layer = Activation("tanh")(hidden_layer)

        return Model([input_layer, conditioned_layer], output_layer), output_layer
    
    
    def predictions(self, epoch):
        '''
            On a per-epoch basis prediction. The GAN predicts the new generated images 
            and generates new samples in this way.
        
        '''
        
        digits = [0,1,2,3,4,5,6,7,8,9]
        
        fig, axs = plt.subplots(5, 6, figsize=(10,6))
        plt.subplots_adjust(hspace=0.3, wspace=0.1)
        
        for classlabel in range(10):
            row = int(classlabel/2)
            coloffset = (classlabel % 2)*3
            labels = to_categorical([classlabel]*3, self._num_classes)
            noise = np.random.normal(0, 1, (3, self._latent_dim))
            gen_images = self._generator.predict([noise, labels])
            
            for i in range(3):
                img = image.array_to_img(gen_images[i], scale=True)
                axs[row,i+coloffset].imshow(img)
                axs[row,i+coloffset].axis('off')
                if i ==1:
                    axs[row,i+coloffset].set_title("Digit: %d" % digits[classlabel])
                
        fig.savefig('mnist_dc2gan_exprep_sample_%d.png' % epoch)
        plt.close()
        
    
    def plot_results(self, disc_losses, gen_losses, model_accuracies):
        '''
            Plot the results of the training of the GAN
        
        '''
        
        # Plot losses
        plt.figure()
        plt.plot(disc_losses)
        plt.plot(gen_losses)
        plt.title('DC2GANExperienceReplay loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Discriminator', 'Generator'], loc='upper left')
        plt.savefig('dc2gan_exp_rep_losses.png')
        
        # Plot accuracy
        plt.figure()
        plt.plot(model_accuracies)
        plt.title('DC2GANExperienceReplay accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['DC2GANExperienceReplay'], loc='upper left')
        plt.savefig('dc2gan_exprep_accuracy.png')
    
    
    def run(self, epochs, batch_size, interval_size, verbose=0):
        '''
            This function is responsible for loading and manipulating the data as well as training the
            Generative Adversarial Network.
            
            epochs: The number of epochs which the model will use for its training.
            batch_size: The batch size for the (batch) training of the model.
            save_images: A flag indicating whether we want to save the generated images or not.
        
        '''
        
        # Load the MNIST dataset from Keras's repository
        (X_train, y_train), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train.astype(np.float32)
        X_train = X_train / 127.5 - 1
        X_train = np.expand_dims(X_train, axis=3)
#         y_train = y_train.reshape(-1,1)
        y_train = to_categorical(y_train, self._num_classes)
        
        # Adversarial ground truths
        valid = np.ones((batch_size, 1)) # real are labelled as one
        adversaries = np.zeros((batch_size, 1)) # adversaries/fake are labelled as zero
        
        disc_losses = np.zeros((epochs,))
        gen_losses = np.zeros((epochs,))
        model_accuracies = np.zeros((epochs,))
        
        experience_replay = []
        
        start = time.time()
        
        for epoch in range(epochs):
            
            # Train the discriminator
            '''
                The goal of the discriminator is to detect which input is legit or adversary.
                The discriminator must not be fooled by the adversary generator model at all times.
            
            '''
            
            # Random sampling of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            images = X_train[idx]
            labels = y_train[idx]
            
            # Sample noise
            noise = np.random.normal(0, 1, (batch_size, self._latent_dim))
            
            # Generate noisy/adversarial images (input to the discriminator/validator)
            adv_images = self._generator.predict([noise, labels])
            
            # Train the discriminator and get the loss from the real and adversary input
            disc_loss_real = self._discriminator.train_on_batch([images, labels], valid)
            disc_loss_adv = self._discriminator.train_on_batch([adv_images, labels], adversaries)
            
            # Experience Replay sampling
            idx = np.random.randint(batch_size)
            experience_replay.append([adv_images[idx], labels[idx], adversaries[idx]])
            if len(experience_replay) == batch_size:
                adv_images = np.array([p[0] for p in experience_replay])
                labels = np.array([p[1] for p in experience_replay])
                adversaries = np.array([p[2] for p in experience_replay])
                disc_loss_adv = self._discriminator.train_on_batch([adv_images, labels], adversaries)
            
            # Calculate loss of discriminator
            disc_loss = np.add(disc_loss_real, disc_loss_adv) / 2
            
            # Train the generator
            '''
                The goal of the generator is to fool successfully the discriminator.
                The ultimate target is to have a quite robust discriminator.
            
            '''
            
            # Calculate the loss of the generator
            conditioned_labels = np.random.randint(0, self._num_classes, batch_size)
            conditioned_labels = to_categorical(conditioned_labels, self._num_classes)
            gen_loss = self._full_gan.train_on_batch([noise, conditioned_labels, conditioned_labels], valid)
            
            # Save the training history
            disc_losses[epoch] = disc_loss[0]
            gen_losses[epoch] = disc_loss[1]
            model_accuracies[epoch] = gen_loss
            
            if verbose == 1:
                print ("%d [D loss: %f, acc.: %.2f%%] G loss: %f" % (epoch, disc_loss[0], 100*disc_loss[1], gen_loss))
            
            print('Epoch:', epoch+1)
            
            # Batch predictions (according to interval size)
            if epoch % interval_size == 0:
                self.predictions(epoch)
        
        print('Time elapsed during training: ', time.time()-start, 'seconds.')
        
        self.plot_results(disc_losses, gen_losses, model_accuracies)
        
        print('Mean discriminator loss:', np.mean(disc_loss[0]))
        print('Mean discriminator accuracy:', np.mean(disc_loss[1]))
        print('Mean generator loss:', np.mean(gen_loss))

class RandomWeightedAverage(_Merge):
    '''
        Provides a randomly-weighted average between real and generated image samples

    '''

    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class WGAN_GP():
    '''
        A class for a Wasserstein distance Deep Convolutional Generative Adversarial Network.
    
    '''

    def __init__(self):
        ''' Properties initialisation '''
        self._rows = 28
        self._cols = 28
        self._channels = 1
        self._input_shape = (self._rows, self._cols, self._channels)
        self._latent_dim = 100
        self._num_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Get the generative models
        self._generator = self.get_generator()
        self._discriminator = self.get_discriminator()

        ''' Computational graph of the discriminator/critic '''

        # Freeze generator's layers while training discriminator
        self._generator.trainable = False

        real_input = Input(shape=self._input_shape) # real sample

        # Noisy input
        disc_noise = Input(shape=(self._latent_dim,))
        adv_input = self._generator(disc_noise) # generated samples

        # Discriminator determines validity of the real and adv_output images
        adv_output = self._discriminator(adv_input)
        real_output = self._discriminator(real_input)

        # Interpolate real and adversarial input using weighted average
        interpolated_input = RandomWeightedAverage()([real_input, adv_input])
        interpolated_output = self._discriminator(interpolated_input)

        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=interpolated_input)
        partial_gp_loss.__name__ = 'gradient_penalty'

        self._discriminator_model = Model(inputs=[real_input, disc_noise], outputs=[real_output, adv_output, interpolated_output])
        self._discriminator_model.compile(loss=[self.wasserstein_distance,
                                                self.wasserstein_distance,
                                                partial_gp_loss],
                                          optimizer=optimizer,
                                          loss_weights=[1, 1, 10])

        ''' Computational graph of the generator '''

        # Discriminator's layers are frozen for training the generator
        self._discriminator.trainable = False
        self._generator.trainable = True

        # Generated samples
        generated_noise = Input(shape=(self._latent_dim,))
        generated_sample = self._generator(generated_noise)

        real_output = self._discriminator(generated_sample)

        self._generator_model = Model(generated_noise, real_output)
        self._generator_model.compile(loss=self.wasserstein_distance, optimizer=optimizer)

    def wasserstein_distance(self, ground_truth, preds):
        '''
                L1-norm Wasserstein distance (loss function).

        '''
        return K.mean(ground_truth * preds)

    def gradient_penalty_loss(self, ground_truth, preds, averaged_samples):
        '''
            L2-norm Gradient Penalty loss function using the predicted labels
            and the interpolation of the reak and adversarial samples.

        '''

        gradients = K.gradients(preds, averaged_samples)[0]
        # Euclidean distance gradients
        gradients_sqr = K.square(gradients)

        # L2-norm gradients
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)

        # compute lambda * (1 - ||grad||)^2 for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)

        return K.mean(gradient_penalty)

    def get_discriminator(self):
        '''
            This function builds the architecture of the discriminator model (or validator)
            In our case, the discriminator is a deep CNN-based neural architecture and is 
            responsible for the validation of the given image (legit or adversarial_labels).
            
        '''

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self._input_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        disc_sample = Input(shape=self._input_shape)
        discriminator_output = model(disc_sample)

        return Model(disc_sample, discriminator_output)

    def get_generator(self):
        '''
            This function builds the architecture of the generator model (or adversary)
            In our case, the generator is a deep CNN-based neural architecture and is 
            responsible for the generating adversarial input to the discriminator.
            
        '''

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self._latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self._channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self._latent_dim,))
        gen_sample = model(noise)

        return Model(noise, gen_sample)

    def predictions(self, epoch):
        rows = 5
        columns = 5
        noise = np.random.normal(0, 1, (rows*columns, self._latent_dim))
        gen_samples = self._generator.predict(noise)

        # Transform generated samples into [0,1] range (scaling)
        gen_samples = 0.5 * gen_samples + 0.5

        # Save predictions
        fig, axs = plt.subplots(rows, columns)
        counter = 0
        for i in range(rows):
            for j in range(columns):
                axs[i,j].imshow(gen_samples[counter, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                counter += 1
        fig.savefig('mnist_wgan_gp_sample_%d.png' % epoch)
        plt.close()

    def plot_results(self, disc_real_losses, disc_adv_losses, disc_grad_pen_losses, gen_losses):
        '''
            Plot the results of the training of the GAN
        
        '''
        
        # Plot real losses
        plt.figure()
        plt.plot(disc_real_losses)
        plt.plot(gen_losses)
        plt.title('WGAN_GP Wasserstein loss on real samples')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Discriminator', 'Generator'], loc='upper left')
        plt.savefig('wgan_gp_real_losses.png')
        
        # Plot adversarial losses
        plt.figure()
        plt.plot(disc_adv_losses)
        plt.plot(gen_losses)
        plt.title('WGAN_GP Wasserstein loss on adversarial samples')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Discriminator', 'Generator'], loc='upper left')
        plt.savefig('wgan_gp_adv_losses.png')
        
        # Plot Gradient Penalty losses
        plt.figure()
        plt.plot(disc_grad_pen_losses)
        plt.plot(gen_losses)
        plt.title('WGAN_GP loss on gradient penalty samples')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Discriminator', 'Generator'], loc='upper left')
        plt.savefig('wgan_gp_grad_pen_losses.png')

    def run(self, epochs, batch_size, interval_size=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        real_labels = -np.ones((batch_size, 1))
        adversarial_labels =  np.ones((batch_size, 1))
        dummy_labels = np.zeros((batch_size, 1))
        
        disc_real_losses = np.zeros((epochs,))
        disc_adv_losses = np.zeros((epochs,))
        disc_grad_pen_losses = np.zeros((epochs,))
        gen_losses = np.zeros((epochs,))
        
        start = time.time()
        
        for epoch in range(epochs):

            for _ in range(self._num_critic):

                ''' Train the dicriminator/critic '''

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                random_samples = X_train[idx]
                noise = np.random.normal(0, 1, (batch_size, self._latent_dim))
                discriminator_loss = self._discriminator_model.train_on_batch([random_samples, noise],
                                                                [real_labels, adversarial_labels, dummy_labels])

            ''' Train the generator '''

            generator_loss = self._generator_model.train_on_batch(noise, real_labels)
            
            # Save the training history
            disc_real_losses[epoch] = discriminator_loss[0]
            disc_adv_losses[epoch] = abs(discriminator_loss[1])
            disc_grad_pen_losses = discriminator_loss[2]
            gen_losses[epoch] = abs(generator_loss)
            
            print('Epoch:', epoch+1)
        
            if epoch % interval_size == 0:
                self.predictions(epoch)
                
        print('Time elapsed during training: ', time.time()-start, 'seconds.')
        
        self.plot_results(disc_real_losses, disc_adv_losses, disc_grad_pen_losses, gen_losses)

        print('Mean discriminator Wasserstein loss on real samples:', np.mean(disc_real_losses))
        print('Mean discriminator Wasserstein loss on adversarial samples:', np.mean(disc_adv_losses))
        print('Mean discriminator Gradient Penalty loss:', np.mean(disc_grad_pen_losses))
        print('Mean generator loss:', np.mean(gen_losses))


model = DCGAN()
model.run(epochs=20000, batch_size=128, interval_size=100)

model = DC2GAN()
model.run(epochs=20000, batch_size=100, interval_size=100)

model = DC2GANExperienceReplay()
model.run(epochs=20000, batch_size=100, interval_size=100)

model = WGAN_GP()
model.run(epochs=30000, batch_size=32, interval_size=100)