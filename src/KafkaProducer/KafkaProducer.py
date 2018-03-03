# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:29:32 2018

@author: Bin
"""
import time
from kafka import KafkaProducer
import pandas as pd

filename = "C:/Users/Bin/Documents/Datasets/KDD99/Continuously/smtp.csv"
filename = "C:/Users/Bin/Documents/Datasets/EncDec-AD dataset/power_data_labeled.csv"
producer = KafkaProducer(bootstrap_servers='localhost:9092')

chunksize = 100#10000
skiprows =1008#20000
for chunk in pd.read_csv(filename,names=None, chunksize=chunksize,skiprows=skiprows):

    for index,row in chunk.iterrows():
        prefix = (str(index+skiprows)+",").encode()
        suffix = row.to_json(orient="split").split("data\":[")[1].strip("\"]}'").replace("\"","").encode()
        message = prefix+suffix
        producer.send('kdd99stream', message)
        print(message)
        
        time.sleep(0.1)

