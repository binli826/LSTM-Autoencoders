
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

![LSTM-Autoencoder](https://github.com/binli826/MasterThesis/blob/master/Figures/LSTM-Autoencoder%28RepeatVector%29.png)
