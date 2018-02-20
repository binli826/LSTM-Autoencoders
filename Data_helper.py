# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 09:20:38 2018

@author: Bin
"""
import numpy as np
import pandas as pd
from kafka import KafkaConsumer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
class Data_Helper(object):
    
    def __init__(self, path,step_num,batch_num,training_data_source):
        self.path = path
        self.step_num = step_num
        self.batch_num = batch_num
        self.training_data_source = training_data_source
        self.column_name_file = "C:/Users/Bin/Documents/Datasets/KDD99/columns.txt"
        self.training_set_size = 80000
        self.save_path ="C:/Users/Bin/Desktop/Thesis/Plotting/"
        if training_data_source == "stream":
            self.df = self.read_stream(self.training_set_size)
            
        elif training_data_source == "file":
            self.df = pd.read_csv(self.path,header=None)
         
        else:
            print("Wrong option of training_data_source, could only choose stream or file.")

        print("Preprocessing...")
        # the model always need continuous normal data (at least "batch_num * step_num" continue normal examples for each iteration
        if training_data_source == "stream":
            for count in range(10):
                if count == 9:
                    raise Exception("Time out, didn't got enough continuous normal data for trianing, pleace use data from file.")
                self.sn,self.vn1,self.vn2,self.tn,self.va,self.ta,class_labels = self.preprocessing(self.df)
                if min(self.sn.size,self.vn1.size,self.vn2.size,self.tn.size,self.va.size,self.ta.size) == 0:
                    print("Currently not enough continuous normal data in the stream for training, waiting for next batch...")
                    self.df = pd.concat((self.df,self.read_stream(10000)),axis=0).reset_index(drop=True) # read another 10000 rows from stream
                    continue
                else:
                    break
        else:            
             self.sn,self.vn1,self.vn2,self.tn,self.va,self.ta,class_labels = self.preprocessing(self.df)
             assert min(self.sn.size,self.vn1.size,self.vn2.size,self.tn.size,self.va.size,self.ta.size) > 0, "Not enough continuous data in file for training, ended."+str((self.sn.size,self.vn1.size,self.vn2.size,self.tn.size,self.va.size,self.ta.size))
        self.data_distribution_plotting(class_labels)
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
        
        
        print("Ready for training.")
    def read_stream(self,size):
        kafka_topic = 'kdd99stream'
        g_id='test-consumer-group'
        servers = ['localhost:9092']
        offset = "earliest"
        print("Connecting with kafka stream...")
        consumer = KafkaConsumer(kafka_topic,
                                 group_id=g_id,    # defined in consumer.properties file
                                 bootstrap_servers=servers,
                                 auto_offset_reset = offset)
        consumer.poll()
        #go to end of the stream
        consumer.seek_to_end()
        
        data = []
        print("Connection established.")
        print("Collecting data from stream...")
        for message in consumer:
            if len(data) > size:
                break
            else:
                row = message.value.decode("utf-8") 
                row_array = row.split(",")
                data.append(row_array)
            if len(data) %10000 ==0 and len(data)!=0:
                print(len(data),'/',size)
        df = pd.DataFrame(np.array(data))
        print("Data collection finished.\n")
        return df
    
    
    def preprocessing(self,df):
        
        with open(self.column_name_file) as col_file:
            line = col_file.readline()
        columns = line.split('.')
        col_names = []
        col_types = []
        for col in columns:
            col_names.append(col.split(': ')[0].strip())
            col_types.append(col.split(': ')[1])
        col_names.append("label")       
        
        
        
        continuous = df.iloc[:,np.array(pd.Series(col_types)=="continuous")]
        label = df.iloc[:,-1]
        print("continuous",continuous.shape)
        scaler = MinMaxScaler()
        scaler.fit(continuous)
        cont = scaler.transform(continuous)
            
        cont = pd.DataFrame(cont)
        cont.columns = continuous.columns.values
        data = pd.concat((cont,label),axis=1)
        print("data",data.shape)
        # split data according to window length
        L = self.step_num 
        n_list = []
        a_list = []
        temp = []
        
        for index, row in data.iterrows():
            if len(temp) ==1:
                for x in temp:
                    if data.iloc[x,-1] == "normal.":
                        n_list.append(x)
                    else:
                        a_list.append(x)
                temp.clear()
                temp.append(index)
                continue
            if len(temp) == 0:
                temp.append(index)
            elif row[row.size-1]== data.iloc[temp[0],-1]:
                temp.append(index)
            else:
                temp.clear()
                temp.append(index)
        normal = data.iloc[np.array(n_list),:-1]
        anomaly = data.iloc[np.array(a_list),:-1]
        print("normal,anomaly",normal.shape,anomaly.shape)
        a_labels = data.iloc[np.array(a_list),-1]

        tmp = normal.index.size//10 # 4:2:2:2, va.size == vn2.size
        sn = normal.iloc[:tmp*4,:]
        vn1 = normal.iloc[tmp*4:tmp*6,:]
        vn2 = normal.iloc[tmp*6:tmp*8,:]
        tn = normal.iloc[tmp*8:,:]

        va = anomaly.iloc[0:tmp*2,:] if anomaly.index.size >tmp else anomaly[0:anomaly.index.size//2]
        ta = anomaly.iloc[va.index.size:,:]
        class_labels = ['normal' for _ in range(sn.shape[0]+vn1.shape[0]+vn2.shape[0]+tn.shape[0])]

        class_labels += list(a_labels)
        print("Local preprocessing finished.")
        return sn,vn1,vn2,tn,va,ta,class_labels
    def data_distribution_plotting(self,class_labels):
        fig, ax = plt.subplots(1,1,figsize=(13,6))  
        
        ax.set_title("Initialization dataset distribution")
        ax.set_xlabel("Subsets")
        ax.set_ylabel("Count")

        count = []
        labels = []
        class_labels = pd.Series(class_labels)
        for class_label in class_labels.unique():
            count.append(class_labels[class_labels==class_label].size)
            labels.append(class_label)
        ax.bar(range(len(labels)),count,width=0.2)
        if len(count)<10:
            r = 'horizontal'
        else:
            r = 'vertical'
        ax.set_xticklabels(labels, rotation=r)
        ax.set_xticks(range(len(labels)))
        
        ax.set_ylim([0,max(count)*1.3])
        rects = ax.patches
        for rect in rects:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
    
            ax.annotate(y_value, (x_value, y_value), xytext=(0, 5), textcoords="offset points", 
                ha='center', va='bottom')     
        t = str(int(time.time()))
        plt.savefig(self.save_path+"Initial trainset distribution"+t+".png")
        plt.show()
        plt.close()
        
        
                
        
    