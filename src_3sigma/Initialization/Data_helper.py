# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 09:20:38 2018

@author: Bin
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Data_Helper(object):
    
    def __init__(self, path,training_set_size,step_num,batch_num,training_data_source):
        self.path = path
        self.step_num = step_num
        self.batch_num = batch_num
        self.training_data_source = training_data_source
        self.training_set_size = training_set_size
       
        self.df = pd.read_csv(self.path).iloc[:self.training_set_size,:]

        print("Preprocessing...")
        
        self.trainingset = self.preprocessing(self.df)
    
        tt = self.trainingset.shape[0] // step_num
        self.traininglist = [self.trainingset[step_num*i:step_num*(i+1)].as_matrix() for i in range(tt)]
        print("Ready for training.")
  
    
    def preprocessing(self,df):
        
        label = df.iloc[:,-1]
        scaler = MinMaxScaler()
        scaler.fit(df.iloc[:,:-1])
        cont = scaler.transform(df.iloc[:,:-1])
            
        cont = pd.DataFrame(cont)
#        cont.columns = continuous.columns.values
        data = pd.concat((cont,label),axis=1)
        
        # split data according to window length
        n_list = []
        a_list = []
        #new version: iter windows
        windows = [data.iloc[w*self.step_num:(w+1)*self.step_num,:] for w in range(data.index.size//self.step_num)]
        for win in windows:
            label = win.iloc[:,-1]
            if label[label!="normal"].size == 0:
                n_list += [i for i in win.index]
            else:
                a_list += [i for i in win.index]
                
        normal = data.iloc[np.array(n_list),:-1]
        anomaly = data.iloc[np.array(a_list),:-1]

        return normal
        
                
        
    