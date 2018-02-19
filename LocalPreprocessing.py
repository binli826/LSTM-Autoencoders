# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:35:20 2018

@author: Bin
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class LocalPreprocessing(object):
    
    def __init__(self, column_name_file ,step_num):
        # read in column names of KDD99 dataset
        
        with open(column_name_file) as col_file:
            line = col_file.readline()
        columns = line.split('.')
        self.col_names = []
        self.col_types = []
        for col in columns:
            self.col_names.append(col.split(': ')[0].strip()) # size ==41
            self.col_types.append(col.split(': ')[1])
        self.col_names.append("label")
        self.L = step_num

        
    def run(self,dataset, for_training):
        df = dataset.iloc[:,1:]
        df_index = dataset.iloc[:,0]
        assert len(self.col_names) == df.shape[1], "Length inconsistant: col_names(%d), df.columns(%d)"%(len(self.col_names), df.shape[1])
        df.columns = self.col_names
        label = df.iloc[:,-1]
        df = df.iloc[:,:-1]
        continuous = df.iloc[:,np.array(pd.Series(self.col_types)=="continuous")] # 34 continuous features
     
        grundtruth = np.zeros(label.size)
        grundtruth[np.array(label)!="normal"] = 1
        grundtruth = pd.Series(grundtruth)
        # scaling 
        scaler = MinMaxScaler()
        scaler.fit(continuous)
        cont = scaler.transform(continuous)

        cont = pd.DataFrame(cont)
        cont.columns = continuous.columns.values
        assert cont.index.size == label.size
        [cont,label] = [cl.reset_index(drop=True) for cl in [cont,label]]
        data = pd.concat((cont,label),axis=1)
        assert data.index.size == grundtruth.size
        data = pd.concat((data,grundtruth),axis=1)
        
        # for test, return a scaled data block,
        # with second to last col being the string class label
        # and last col being the 0/1 grundtruth (1 stand for anomaly)
        if for_training == False:
            return data
        
        # for training or retraining, return a list of sub-dataset for different uses
        else:
            # split data according to window length
            n_list = []
            a_list = []
            temp = []
            data = pd.concat((df_index,data),axis=1)
            # for training set split, considering only the continous data points with in a length L window
            for index, row in data.iterrows():
                if len(temp) ==self.L:
                    for x in temp:
                        if data.iloc[x,-2] == "normal":
                            n_list.append(x)
                        else:
                            a_list.append(x)
                    temp.clear()
                    temp.append(index)
                    continue
                if len(temp) == 0:
                    temp.append(index)
                elif row.label == data.iloc[temp[0],-2]:
                    temp.append(index)
                else:
                    temp.clear()
                    temp.append(index)
        
            normal = data.iloc[np.array(n_list),:-2]
            anomaly = data.iloc[np.array(a_list),:-2]
            
#            n_labels = data.iloc[np.array(n_list),-2]
            a_labels = data.iloc[np.array(a_list),:-2]
            # size of sn:vn1:vn2:tn == 3:1:1:4 (self defined)
#            x = int(normal.shape[0]/self.L)
#            sn = normal[:(x//2)*self.L]
#            vn1 = normal[(x//2)*self.L:(x//2)*self.L+(x//6)*self.L]
#            vn2 = normal[(x//2)*self.L+(x//6)*self.L:(x//2)*self.L+(x//3)*self.L]
#            tn = normal[(x//2)*self.L+(x//3)*self.L:]
            
            tmp = normal.index.size//10 # 4:2:2:2, va.size == vn2.size
            sn = normal.iloc[:tmp*4,:]
            vn1 = normal.iloc[tmp*4:tmp*6,:]
            vn2 = normal.iloc[tmp*6:tmp*8,:]
            tn = normal.iloc[tmp*8:,:]
            # size of va:ta == 1:3 (self defined)
#            y = int(anomaly.shape[0]/self.L)
#            va = anomaly[:(y//4)*self.L]
#            ta = anomaly[(y//4)*self.L:]
            va = anomaly.iloc[0:tmp*2,:] if anomaly.index.size >tmp else anomaly[0:anomaly.index.size//2]
            ta = anomaly.iloc[va.index.size:,:]
            class_labels = []
            for nd in [sn,vn1,vn2,tn]:
                class_labels.append(np.array(['normal' for _ in range(nd.index.size)]))
            class_labels.append(a_labels.as_matrix)
            return sn,vn1,vn2,tn,va,ta,class_labels
            
        
        