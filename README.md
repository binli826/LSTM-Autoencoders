
Demo1: LSTM-Autoencoder using Keras
----------------

Target: Implement a LSTM-based autoencoder for reconstruction of time-series data.         
> LSTMs network take input with the form [samples, timesteps, input_dim]

> - **Encoder input**: batched of samples, each batch considering 'timesteps' samples from the time series, each sample has 'input_dim' dimensions

> - **Encoder output**: a single static vector that represent the input information (vector length = #hiden layer neurons)

>- **RepeatVector**: repeat the encoded vector 'timesteps' times, and then feed to the input layer of decoder

>- **Decoder input**: identical RepeatVectors

>- **Decoder output**: expected prediction with the same shape as the autoencoder input, but the order of the sequence is inverted.


The architecture of the LSTM-Autoencoder(RepeatVector) is shown below:

(While we consider timesteps=3 in the example, we need 3 RepeatVectors for the 3 inputs of decoder.)

![LSTM-Autoencoder](https://github.com/binli826/MasterThesis/blob/master/Figures/LSTM-Autoencoder%28RepeatVector%29.png)

The demo is shown in  [LSTM_Autoencoder_Repeatvector_KDD99.ipynb](https://github.com/binli826/MasterThesis/blob/master/LSTM_Autoencoder_Repeatvector_KDD99.ipynb) .
In order to show performance of the autoencoder, we use the KDDCup99 dataset, choose 34 numerical features, and the label.
Two SVM classifiers are trained on original dataset and encoded dataset. (**Note: As the Autoencoder always encodes t samples into a single static vector representation, where t is time steps, the encoded dataset is t times smaller than the original, so we take the last label of each sample group as the label of the encoded static vector.**) 
By using the hyper-parameter combination:

- timesteps = 30
- latent_dimension = 10
- epoch = 100
- batch_size = 100

we got following results:

![Performance](https://github.com/binli826/MasterThesis/blob/master/Figures/RepeatVectorPerformance.PNG)
