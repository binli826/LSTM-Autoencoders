# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:22:47 2018

@author: Bin
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import queue
from kafka import KafkaConsumer
import time, threading
import sys
sys.path.insert(0, 'C:/Users/Bin/Desktop/Thesis/code')
from EncDecAD_Pred import EncDecAD_Pred
from Conf_Prediction_KDD99 import Conf_Prediction_KDD99
from LocalPreprocessing import LocalPreprocessing

conf = Conf_Prediction_KDD99()
batch_num =conf.batch_num
step_num = conf.step_num

kafka_topic = 'kdd99stream'
g_id='test-consumer-group'
servers = ['localhost:9092']
offset = "earliest"

consumer = KafkaConsumer(kafka_topic,
                         group_id=g_id,    # defined in consumer.properties file
                         bootstrap_servers=servers,
                         auto_offset_reset = offset)
consumer.poll()
#go to end of the stream
consumer.seek_to_end()

dataframe = pd.DataFrame()
lock = threading.Lock()


def block_generator2queue(q,stop_event):
   
    while not stop_event.is_set():
        block = []
        
        for message in consumer:
            if stop_event.is_set():
                break
            row = message.value.decode("utf-8") 
#            list_of_str = row.replace(",null,null,null,null,null,null,null,null","").split(",")
#            list_of_str = row.strip(",null,null,null,null,null,null,null,null").split(",")
#            list_of_num = [float(n) for n in list_of_str]
#            block.append(list_of_num)
            block.append(row.split(","))
            if len(block)==batch_num*step_num:
                df = pd.DataFrame(np.array(block))
                q.put(df)
                block.clear()
                
def read_block_from_queue(q,stop_event):
    global dataframe
    
    while not stop_event.is_set():
        if q.empty() == False:
            b = q.get()
            if dataframe.size == 0:
                dataframe = b
            else:
                pd.concat((dataframe,b),axis=0)
                
        else :
            time.sleep(0.5)
            
def prediction(stop_event):
    global dataframe
    pred = EncDecAD_Pred(conf)
    local_preprocessing = LocalPreprocessing(conf.column_name_file ,conf.step_num)
    #  reload model
    sess = tf.Session()
    input_,output_,p_input,p_is_training,mu,sigma,threshold = pred.reloadModel(sess)

    print("LSTMs-Autoencoder Model reloaded.")
   
    while not stop_event.is_set():
        if dataframe.size == 0:
            sec = 5
            print("Currently not enough data for prediction, wait for %d seconds."%sec)
            time.sleep(sec)
        else:
            lock.acquire()
            try:
                print("Local preprocessing...")
                #After preprocessing, the second to last col is the string class label
                # and last col is the 0/1 grundtruth (1 stand for anomaly)
                dataframe = local_preprocessing.run(dataframe, for_training = False)
                
                print("Making prediction...")
              
                dataset = dataframe.iloc[:,:-2]
                label = dataframe.iloc[:,-1]
                class_list = dataframe.iloc[:,-2]
                pred.predict(dataset,label,sess,input_,output_,p_input,p_is_training,mu,sigma,threshold)
                print("Finish prediction.")
                
            finally:
                dataframe = pd.DataFrame()
                lock.release()
                
                
def main():
    q = queue.Queue()
    stop_event = threading.Event()
    
    write = threading.Thread(target=block_generator2queue, name='WriteThread',args=(q,stop_event,))
    read = threading.Thread(target=read_block_from_queue, name='ReadThread',args=(q,stop_event,))
    predict = threading.Thread(target=prediction, name='Prediction',args=(stop_event,))
   
    try:
        write.start()
        read.start()
        predict.start()
        
        while 1:
            time.sleep(.1)
    except (KeyboardInterrupt,SystemExit):
        stop_event.set()
        print("Threads closed.")
        
if __name__=="__main__":
    main()

    

    
