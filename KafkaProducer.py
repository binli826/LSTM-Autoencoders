# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:29:32 2018

@author: Bin
"""
import time
from kafka import KafkaProducer
import pandas as pd

filename = "C:/Users/Bin/Documents/Datasets/KDD99/kddcup.data_10_percent_corrected"

producer = KafkaProducer(bootstrap_servers='localhost:9092')




chunksize = 1000
for chunk in pd.read_csv(filename,header=None, chunksize=chunksize):
    for index,row in chunk.iterrows():
        message = row.to_json(orient="split").split("data\":[")[1].strip("\"]}'").encode()
        producer.send('kdd99stream', message)
        print(message)
        time.sleep(0.5)