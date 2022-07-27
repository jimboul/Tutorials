#!/usr/bin/env python
# coding: utf-8

# # Synthetic time series data using GAN tutorial

# ## Part B: Experiments on Air Quality dataset

# ### Note!: 
# 
# 1. We choose the DC2GAN model to use for the creation of synthetic Time Series data training it on the Air Quality dataset.
# 
# 2. We slightly modify the DC2GAN class to work properly on the time series data.

# In[ ]:


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
get_ipython().run_line_magic('matplotlib', 'inline')

import sys
import time

import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ### Stationarity check

# In[42]:


def test_stationarity(timeseries):
    
    #Determing rolling statistics
    roll_mean = timeseries.rolling(12).mean()
    roll_std = timeseries.rolling(12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(roll_mean, color='red', label='Rolling Mean')
    std = plt.plot(roll_std, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    ### Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    
    # The number of lags is chosen to minimise the Akaike's Information Criterion (AIC)
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
    print('\n\n')
    
    ### Perform KPSS test:
    print ('Results of KPSS Test:')
    
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)


# ### Load the data

# In[11]:


data = pd.read_csv('~/Desktop/AirQuality/air_quality.csv', delimiter=';')


# In[12]:


data = data.drop(['Date', 'Time'], axis=1)


# In[13]:


scaler = MinMaxScaler()
data_transformed = pd.DataFrame(scaler.fit_transform(data))


# In[14]:


data_transformed.head()


# In[16]:


original_data = pd.DataFrame(scaler.inverse_transform(data_transformed))


# In[17]:


original_data.head()


# ### Check for missing/NaN/infinite values

# In[45]:


for i in range(data.shape[1]):
    if np.isnan(data.iloc[:,i]).any() == True:
        print('Nan values exist in the ', data.columns[i], ' column')
    if np.isfinite(data.iloc[:,i]).any() == False:
        print('Infinite values exist in the ', data.columns[i], ' column')


# ### Check if the multivariate time series data are stationary

# In[46]:


for i in range(data.shape[1]):
    print('===============================================\n')
    print(data.columns[i], ' column under investigation...')
    
    test_stationarity(data.iloc[:,i])
    
    print('\n')
    print('===============================================\n')


# In[47]:


data.shape[0] # self._rows ~ number of timesteps


# In[48]:


data.shape[1] # self._cols ~ number of features


# In[2]:


class GAN():
    '''
        A class for a Generative Adversarial Network.
    
    '''
    
    def __init__(self):
        # Properties initialisation
        self._rows = 13
        self._cols = 1
        self._channels = 1
        self._input_shape = (self._rows, self._cols)
        self._latent_dim = 100 # shape of the noise
        self._synthetic_data = pd.DataFrame()
        self._scaler = MinMaxScaler()
        
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

        model.add(Flatten(input_shape=self._input_shape))
        
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(1, activation='sigmoid'))
        
        model.summary()
        
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
        
        model.add(Dense(256, input_dim=self._latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.9))
        
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.9))
        
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.9))
        
        model.add(Dense(np.prod(self._input_shape), activation='sigmoid'))
        model.add(Reshape(self._input_shape))
        
        model.summary()
        
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
        
        # Get samples from the generator
        gen_samples = self._generator.predict(noise)
        gen_samples = np.squeeze(gen_samples)
        gen_samples = pd.DataFrame(gen_samples)
        print('Generated samples: \n\n',gen_samples.head())
        
        for i in range(gen_samples.shape[0]):
            self._synthetic_data = pd.concat([self._synthetic_data, gen_samples], axis=0)
        
    
    def plot_results(self, disc_losses, gen_losses, model_accuracies):
        '''
            Plot the results of the training of the GAN
        
        '''
        
        # Plot losses
        plt.figure()
        plt.plot(disc_losses)
        plt.plot(gen_losses)
        plt.title('GAN loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Discriminator', 'Generator'], loc='upper left')
        plt.savefig('gan_losses.png')
        
        # Plot accuracy
        plt.figure()
        plt.plot(model_accuracies)
        plt.title('GAN accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['GAN'], loc='upper left')
        plt.savefig('gan_accuracy.png')
    
    
    def run(self, epochs, batch_size, interval_size, verbose=0):
        '''
            This function is responsible for loading and manipulating the data as well as training the
            Generative Adversarial Network.
            
            epochs: The number of epochs which the model will use for its training.
            batch_size: The batch size for the (batch) training of the model.
            save_images: A flag indicating whether we want to save the generated images or not.
        
        '''
        
        original_data = pd.read_csv('~/Desktop/AirQuality/air_quality.csv', delimiter=';')
        original_data = original_data.drop(['Date', 'Time'], axis=1)
        original_data = self._scaler.fit_transform(original_data)
        pd.DataFrame(original_data).to_csv('air_quality_original_transformed_data.csv')
        original_data = np.expand_dims(original_data, axis=3)
        
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
            idx = np.random.randint(0, original_data.shape[0], batch_size)
            samples = original_data[idx]
            
            # Sample noise
            noise = np.random.normal(0, 1, (batch_size, self._latent_dim))
            
            # Generate noisy/adversarial images (input to the discriminator/validator)
            adv_samples = self._generator.predict(noise)
            
            # Train the discriminator and get the loss from the real and adversary input
            disc_loss_real = self._discriminator.train_on_batch(samples, valid)
            disc_loss_adv = self._discriminator.train_on_batch(adv_samples, adversaries)
            
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
        
        self._synthetic_data.to_csv('air_quality_synthetic_transformed_data.csv')
                


# In[3]:


gan_model = GAN()

gan_model.run(epochs=20000, batch_size=8, interval_size=100)


# ### Original time series data sample

# In[5]:


original_data = pd.read_csv('air_quality_original_transformed_data.csv')
original_data.head()


# ### Synthetic time series data

# In[6]:


synthetic_data = pd.read_csv('air_quality_synthetic_transformed_data.csv')
synthetic_data.head()


# In[ ]:




