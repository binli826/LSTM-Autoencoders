# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:53:38 2018

@author: Bin
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Given a initialization dataset, split it into normal lists and abnormal lists of different subsets.

class Data_Helper(object):
    
    def __init__(self, path,training_set_size,step_num,batch_num,training_data_source,log_path):
        
        self.path = path
        self.step_num = step_num
        self.batch_num = batch_num
        self.training_data_source = training_data_source
        self.training_set_size = training_set_size
        
        

        self.df = pd.read_csv(self.path).iloc[:self.training_set_size,:]
            
        print("Preprocessing...")
        
        self.sn,self.vn1,self.vn2,self.tn,self.va,self.ta,self.va_labels = self.preprocessing(self.df,log_path)
        assert min(self.sn.size,self.vn1.size,self.vn2.size,self.tn.size,self.va.size,self.ta.size) > 0, "Not enough continuous data in file for training, ended."+str((self.sn.size,self.vn1.size,self.vn2.size,self.tn.size,self.va.size,self.ta.size))
           
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
        
        self.va_label_list =  [self.va_labels[step_num*i:step_num*(i+1)].as_matrix() for i in range(t2)]
        
        print("Ready for training.")
        
    def preprocessing(self,df,log_path):
        
        #scaling
        label = df.iloc[:,-1]
        scaler = MinMaxScaler()
        scaler.fit(df.iloc[:,:-1])
        cont = pd.DataFrame(scaler.transform(df.iloc[:,:-1]))
        data = pd.concat((cont,label),axis=1)
        
        # split data according to window length
        # split dataframe into segments of length L, if a window contains mindestens one anomaly, then this window is anomaly wondow
        n_list = []
        a_list = []
        
        windows = [data.iloc[w*self.step_num:(w+1)*self.step_num,:] for w in range(data.index.size//self.step_num)]
        for win in windows:
            label = win.iloc[:,-1]
            if label[label!="normal."].size == 0:
                n_list += [i for i in win.index]
            else:
                a_list += [i for i in win.index]

        normal = data.iloc[np.array(n_list),:-1]
        anomaly = data.iloc[np.array(a_list),:-1]
        print("Info: Initialization set contains %d normal windows and %d abnormal windows."%(normal.shape[0],anomaly.shape[0]))

        a_labels = data.iloc[np.array(a_list),-1]
        
        # split into subsets
        tmp = normal.index.size//self.step_num//10 
        assert tmp > 0 ,"Too small normal set %d rows"%normal.index.size
        sn = normal.iloc[:tmp*5*self.step_num,:]
        vn1 = normal.iloc[tmp*5*self.step_num:tmp*8*self.step_num,:]
        vn2 = normal.iloc[tmp*8*self.step_num:tmp*9*self.step_num,:]
        tn = normal.iloc[tmp*9*self.step_num:,:]
        
        tmp_a = anomaly.index.size//self.step_num//2 
        va = anomaly.iloc[:tmp_a*self.step_num,:] if tmp_a !=0 else anomaly
        ta = anomaly.iloc[tmp_a*self.step_num:,:] if tmp_a !=0 else anomaly
        a_labels = a_labels[:va.index.size]
        
        print("Local preprocessing finished.")
        print("Subsets contain windows: sn:%d,vn1:%d,vn2:%d,tn:%d,va:%d,ta:%d\n"%(sn.shape[0]/self.step_num,vn1.shape[0]/self.step_num,vn2.shape[0]/self.step_num,tn.shape[0]/self.step_num,va.shape[0]/self.step_num,ta.shape[0]/self.step_num))
        
        f = open(log_path,'a')
        
        f.write("Info: Initialization set contains %d normal windows and %d abnormal windows.\n"%(normal.shape[0],anomaly.shape[0]))
        f.write("Subsets contain windows: sn:%d,vn1:%d,vn2:%d,tn:%d,va:%d,ta:%d\n"%(sn.shape[0]/self.step_num,vn1.shape[0]/self.step_num,vn2.shape[0]/self.step_num,tn.shape[0]/self.step_num,va.shape[0]/self.step_num,ta.shape[0]/self.step_num))
        f.close()
        
        return sn,vn1,vn2,tn,va,ta,a_labels