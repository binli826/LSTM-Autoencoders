# Anomaly detection for streaming data using autoencoders

This project is my master thesis. The main target is to maintain an adaptive autoencoder-based anomaly detection framework that is able to not only detect contextual anomalies from streaming data, but also update itself according to the latest data feature.

## Quick access

  - [Thesis](https://github.com/binli826/LSTM-Autoencoders/blob/master/Thesis.pdf) 
  - [Slides](https://github.com/binli826/LSTM-Autoencoders/blob/master/Slides.pdf)
  - [Usage](https://github.com/binli826/LSTM-Autoencoders/tree/master#usage)

## Introduction
The high-volume and -velocity data stream generated from devices and applications from
different domains grows steadily and is valuable for big data research. One of the most
important topics is anomaly detection for streaming data, which has attracted attention
and investigation in plenty of areas, e.g., the sensor data anomaly detection, predictive
maintenance, event detection. Those efforts could potentially avoid large amount of financial
costs in the manufacture. However, different from traditional anomaly detection tasks,
anomaly detection in streaming data is especially difficult due to that data arrives along
with the time with latent distribution changes, so that a single stationary model doesn’t fit
streaming data all the time. An anomaly could become normal during the data evolution,
therefore it is necessary to maintain a dynamic system to adapt the changes. In this work,
we propose a LSTMs-Autoencoder anomaly detection model for streaming data. This is a
mini-batch based streaming processing approach. We experimented with streaming data
that containing different kinds of anomalies as well as concept drifts, the results suggest
that our model can sufficiently detect anomaly from data stream and update model timely
to fit the latest data property.

## Model
#### LSTM-Autoencoder
The LSTM-Autoencoder is based on the work of [Malhotra et al.] There are two LSTM units, one as encoder and the other one as decoder. Model will only be trained with normal data, so the reconstruction of anomalies is supposed to lead higher reconstruction error.

![LSTM-Autoencoder](https://github.com/binli826/LSTM-Autoencoders/blob/master/Figures/LSTM-Autoencoder.PNG)

> **Input/Output format**
>
> < Batch size, Time steps, Data dimensions > <br />
> Batch size: Number of windows contained in a single batch<br />
> Time steps: Number of instances within a window (T)<br />
> Data dimensions: Size of feature space

#### Online framework
Once the LSTM-Autoencoder is initialized with a subset of respective data streams, it is used for the online anomaly detection. For each accumulated batch of streaming data, the model predict each window as normal or anomaly. Afterwards, we introduce experts to label the windows and evaluate the performance. Hard windows will be appended into the updating buffers. Once the normal buffer is full, there will the a continue training of LSTM-Autoencoders only with the hard windows in the buffers.

![Online framework](https://github.com/binli826/LSTM-Autoencoders/blob/master/Figures/Online.PNG)

## Datasets
The model is experimenced with 5 datasets. [PowerDemand](https://github.com/binli826/LSTM-Autoencoders/blob/master/data/power_data.txt) dataset records the power demand over one year, the unnormal power demand on special days (e.g. festivals, christmas etc.) are labeled as anomalies.
SMTP and HTTP are extracted from the [KDDCup99 dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html). SMTP+HTTP is a direct connection of SMTP and HTTP, in order to simulate a concept drift in between.
Here treat the network attacks as anomalies. [FOREST](https://archive.ics.uci.edu/ml/datasets/covertype) dataset records statistics of 7 different forest cover types. We follow the same setting as [Dong et al.], take the smallest class Cottonwood/Willow as anomaly.
The following table shows statistical information of each dataset.(Only numerical features are taken into consideration)

| Dataset | Dimensionality | #Instances | Anomaly proportion (%) |
| :------: | :------: | :------: | :------: |
| PowerDemand | 1 | 35040 | 2.20 |
| SMTP | 34 | 96554 | 1.22 |
| HTTP | 34 | 623 091 | 0.65 |
| SMTP+HTTP | 34 | 719 645 | 0.72|
| FOREST | 7 | 581 012 | 0.47 |

## Results
Here is an reconstruction example of a normal window and an anomaly window of the PowerDemand data.
![Reconstruction example](https://github.com/binli826/LSTM-Autoencoders/blob/master/Figures/example.PNG)
>
With AUC as evaluation metric, we got following performance of the data stream anomaly detection.

| Dataset | AUC without updating | AUC with updating | #Updating |
| :------: | :------: | :------: | :------: |
| PowerDemand | 0.91 | 0.97 | 2 |
| SMTP | 0.94 | 0.98 | 2 |
| HTTP | 0.76 | 0.86 | 2 |
| SMTP+HTTP | 0.64 | 0.85 | 3|
| FOREST | 0.74 | 0.82 | 8 |


## Usage
#### Data preparation
Once datasets avaliable, covert the raw data into uniform format using [dataPreparation.py]（https://github.com/binli826/LSTM-Autoencoders/blob/master/src/dataPreparation.py）.

```sh
python dataPreparation.py dataset inputpath outputpath --powerlabel --kddcol
# Example
python dataPreparation.py kdd /mypath/kddcup.data.corrected /mypath/tosave --kddcol /mypath/columns.txt
```
#### Initialization


#### Online prediction



## Versions
This project works with
* Python 3.6
* Tensorflow 1.4.0
* Numpy 1.13.3

[Malhotra et al.]: <https://arxiv.org/pdf/1607.00148.pdf>
[Dong et al.]: <https://onlinelibrary.wiley.com/doi/abs/10.1111/coin.12146>