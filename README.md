# Anomaly detection for streaming data using autoencoders

This project is my master thesis, worked on between 02.2018 and 07.2018. The main target is to maintain an adaptive autoencoder-based anomaly detection framework that is able to not only detect contextual anomalies from streaming data, but also update itself according to the latest data feature.

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


## Datasets

## Results

## Usage

## Versions

## Reference
