# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 09:20:38 2018

@author: Bin
"""
import sys
import numpy as np
import pandas as pd
from kafka import KafkaConsumer
sys.path.insert(0, 'C:/Users/Bin/Desktop/Thesis/code')
from LocalPreprocessing import LocalPreprocessing


class Data_Helper(object):
    
    def __init__(self, path,step_num,training_data_source):
        self.path = path
        self.step_num = step_num
        self.training_data_source = training_data_source
        self.column_name_file = "C:/Users/Bin/Documents/Datasets/KDD99/columns.txt"
        self.training_set_size = 10000
        
        if training_data_source == "stream":
            df = self.read_stream()
            
        elif training_data_source == "file":
            df = pd.read_csv(self.path,header=None)
         
        else:
            print("Wrong option of training_data_source, could only choose stream or file.")
        
        
#        self.sn = pd.read_csv(self.root + "training_normal.csv",header=None)
#        self.vn1 = pd.read_csv(self.root + "validation_1.csv",header=None)
#        self.vn2 = pd.read_csv(self.root + "validation_2.csv",header=None)
#        self.tn = pd.read_csv(self.root + "test_normal.csv",header=None)
#
#        self.va = pd.read_csv(self.root + "validation_anomaly.csv",header=None)
#        self.ta = pd.read_csv(self.root + "test_anomaly.csv",header=None)    
        local_proprocessing = LocalPreprocessing(self.column_name_file ,self.step_num)
        self.sn,self.vn1,self.vn2,self.tn,self.va,self.ta = local_proprocessing.run(df, for_training=True)
        # data seriealization
        t1 = self.sn.shape[0]//step_num
        t2 = self.va.shape[0]//step_num
        t3 = self.vn1.shape[0]//step_num
        t4 = self.vn2.shape[0]//step_num
        t5 = self.tn.shape[0]//step_num
        t6 = self.ta.shape[0]//step_num
        
        self.sn_list = [self.sn[step_num*i:step_num*(i+1)].as_matrix() for i in range(t1)]
        self.va_list = [self.va[step_num*i:step_num*(i+1)].as_matrix() for i in range(t2)]
        self.vn1_list = [self.vn1[step_num*i:step_num*(i+1)].as_matrix() for i in range(t3)]
        self.vn2_list = [self.vn2[step_num*i:step_num*(i+1)].as_matrix() for i in range(t4)]
        
        self.tn_list = [self.tn[step_num*i:step_num*(i+1)].as_matrix() for i in range(t5)]
        self.ta_list = [self.ta[step_num*i:step_num*(i+1)].as_matrix() for i in range(t6)]

    def read_stream(self,):
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
        
        data = []
        for message in consumer:
            if len(data) > self.training_set_size:
                break
            else:
                row = message.value.decode("utf-8") 
                data.append(row.split(","))
        df = pd.DataFrame(np.array(data))
        return df