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
from EncDecAD_ReTrain import EncDecAD_ReTrain

conf = Conf_Prediction_KDD99()
batch_num =conf.batch_num
step_num = conf.step_num
MIN_TEST_BLOCK_NUM = conf.min_test_block_num
MIN_RETRAIN_BLOCK_NUM = conf.min_retrain_block_num
kafka_topic = 'kdd99stream'
g_id='test-consumer-group'
servers = ['localhost:9092']
offset = "latest"#"earliest"

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
            row_tmp = row.split(",")
            row_tmp[-1] = row_tmp[-1].strip("\".")
            block.append(row_tmp)
            if len(block)==batch_num*step_num:
                df = pd.DataFrame(np.array(block))
                q.put(df)
                block.clear()
                
def read_block_from_queue(q,stop_event):
    global dataframe
    
    while not stop_event.is_set():
        if q.empty() == False:
            b = q.get()
#            if b.shape[1] <50:   #hardcode         
            if dataframe.size == 0:
                dataframe = b
               
            else:    
                df_tmp = pd.concat((dataframe,b),axis=0).reset_index(drop=True)                  
                if df_tmp.shape[1] != b.shape[1] : 
                    print("Warn: Strange dataframe shape, dropped.") 
                else:
                    dataframe = df_tmp
        else :
            time.sleep(0.5)
            
def prediction(stop_event):
    global dataframe
    pred = EncDecAD_Pred(conf)
    local_preprocessing = LocalPreprocessing(conf.column_name_file ,conf.step_num)
    #  reload model
    sess = tf.Session()
    input_,output_,p_input,p_is_training,loss_,train_,mu,sigma,threshold = pred.reloadModel(sess)

    print("LSTMs-Autoencoder Model reloaded.")
    buffer = [] # for collecting hard examples used for retraining model
    while not stop_event.is_set():        
        lock.acquire()
        if dataframe.index.size < batch_num*step_num*MIN_TEST_BLOCK_NUM:
            sec = 10
            print("Currently not enough data for prediction, ",dataframe.index.size,"/",batch_num*step_num*MIN_TEST_BLOCK_NUM)
            lock.release()
            time.sleep(sec)
            
        else:
            try:
                print("Local preprocessing...")
                #After preprocessing, the second to last col is the string class label
                # and last col is the 0/1 grundtruth (1 stand for anomaly)
                
                dataframe_preprocessed = local_preprocessing.run(dataframe, for_training = False)        
                print("Making prediction...")
              
                dataset = dataframe_preprocessed.iloc[:,:-2]
                label = dataframe_preprocessed.iloc[:,-1]
                class_list = dataframe_preprocessed.iloc[:,-2]
                
                # window.size == step_num
                hard_example_window_index = pred.predict(dataset,label,sess,input_,output_,p_input,p_is_training,mu,sigma,threshold)
                 # got hard examples' index from prediction, then using this index to find the UNpreprocessed 
                 #hard examples from the original dataframe     
                
                dataframe.reset_index(drop=True,inplace=True)
                buffer.append(dataframe.loc[hard_example_window_index])                 
                
                buffer_data_len = sum([df.index.size for df in buffer])
                if buffer_data_len >= MIN_RETRAIN_BLOCK_NUM*batch_num:
                        print("It's time to Re-Training model.")
                        data_for_retrain = pd.concat(buffer,axis=0)
                        data_for_retrain.reset_index(drop=True,inplace=True)
                        #retrain dataset shape: (batch_num*step_num*MIN_RETRAIN_BLOCK_NUM,elem_num)
                        data_for_retrain = data_for_retrain.iloc[:data_for_retrain.index.size-data_for_retrain.index.size%batch_num,:]#buffer[0].shape[1]]#.....................
                        sn,vn1,vn2,tn,va,ta = local_preprocessing.run(data_for_retrain, for_training = True)
                        
#                        if min(sn.size,vn1.size,vn2.size,tn.size,va.size,ta.size) == 0:
                        if min(sn.size,vn1.size,vn2.size,va.size) == 0:
                            print("Not enough normal or anomaly data for retraining, still waiting for more data.")                          
                            print("Retrain Buffer: %d/%d.\n"%(buffer_data_len,MIN_RETRAIN_BLOCK_NUM*batch_num))
                            dataframe = pd.DataFrame()
                            continue
                        print("Re-Training Model...")
                        retrain = EncDecAD_ReTrain(sn,vn1,vn2,tn,va,ta)
                        retrain.continue_training(sess,loss_, train_,p_input,p_is_training)
                        
                        buffer.clear()
                else: 
                    print("Retrain Buffer: %d/%d.\n"%(buffer_data_len,MIN_RETRAIN_BLOCK_NUM*batch_num))
                    print("Finish prediction.Waiting for next batches of data.")
                
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

    

    
