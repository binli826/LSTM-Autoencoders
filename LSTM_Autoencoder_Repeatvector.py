
import pandas as pd
import numpy as np
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model,Sequential
import math


# LSTM-based autoencoder
class foo:
	def __init__(self,bar)
		self.bar = bar
	def p (self):
		print(self.bar)

class LSTM_Autoencoder_RepeatVector:

	def __init__(self, dataset, timesteps, latent_dim, n_epoch,n_batch):
		self.dataset = dataset
		self.timesteps = timesteps
		self.latent_dim = latent_dim 
		self.n_batch = n_batch
		self.n_epoch = n_epoch
'''
    def reshape(self):
        sample_num = math.floor(self.dataset.shape[0]/self.timesteps)
        new_dataset = np.reshape(self.dataset[:sample_num*self.timesteps],(sample_num,self.timesteps,self.dataset.shape[1]))
        return new_dataset

    def lstm_autoencoder_repeatvector(self,optimizer='adadelta',loss='mse'):
        input_dim = self.dataset.shape[1]
        inputs = Input(shape=(self.timesteps,input_dim))
        encoded = LSTM(self.latent_dim)(inputs)
        decoded = RepeatVector(self.timesteps)(encoded)
        decoded = LSTM(input_dim,return_sequences=True)(decoded)
        autoencoder = Model(inputs,decoded)
        encoder = Model(inputs,encoded)
        autoencoder.compile(optimizer=optimizer,loss=loss)

        reversed_dataset = np.empty([0,self.dataset.shape[-1]])
        for i in range(int(self.dataset.shape[0]/self.timesteps)):
            temp = self.dataset[i*self.timesteps:i*self.timesteps+self.timesteps,:]
            temp = temp[::-1,:]
            reversed_dataset = np.concatenate((reversed_dataset,temp))
        new_dataset = self.reshape(self.dataset,self.timesteps)
        reversed_dataset = reshape(reversed_dataset,self.timesteps)
        history = autoencoder.fit(new_dataset,reversed_dataset,
                    epochs=self.n_epoch,
                    batch_size=self.n_batch,
                    validation_split=0.33
                    )

        encoded_dataset = encoder.predict(new_dataset)

        return encoded_dataset
'''
