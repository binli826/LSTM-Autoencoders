
# coding: utf-8

# # Load Dataset

# In[5]:


get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# In[2]:


class Loaddata(object):
    def __init__(self,dataset):
        self.dataset = dataset
        self.folder = "C:/Users/Bin/Documents/Datasets/"
        self.kdd99_col_name_suffix = "KDD99/columns.txt"
        self.kdd99_dataset_suffix = "KDD99/kddcup.data_10_percent_corrected"
        self.power_demand_suffix = "EncDec-AD dataset/power_data.txt"
        
    def read(self):
        if self.dataset == "kdd99":
            with open(self.folder+self.kdd99_col_name_suffix) as col_file:
                line = col_file.readline()
            columns = line.split('.')
            col_names = []
            col_types = []
            for col in columns:
                col_names.append(col.split(': ')[0].strip())
                col_types.append(col.split(': ')[1])
            col_names.append("label")
            df = pd.read_csv(self.folder+self.kdd99_dataset_suffix,names=col_names)
            data = df.iloc[:,np.array(pd.Series(col_types)=="continuous")].as_matrix()   #Select only numeric features
            label = df.iloc[:,-1]

            # Scaling
            scaler = MinMaxScaler()
            scaler.fit(data)
            data = scaler.transform(data) 
            
            return data
        
        elif self.dataset == "power_demand":
            power = pd.read_csv("C:/Users/Bin/Documents/Datasets/EncDec-AD dataset/power_data.txt",names=["power_demand"])
            # downsample the dataset by 8 to obtain non-overlapping sequences with L=84 such that each window corresponds to one week
            sub_power = pd.Series(power[490:].reset_index(drop=True)["power_demand"])
            index = [8*t for t in range(sub_power.shape[0]//8 +1)]
            sub_power = sub_power[index].reset_index(drop=True)  # shape(4319,)
            #Scaling
            sub_power = sub_power.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaler.fit(sub_power)
            sub_power = scaler.transform(sub_power) 
            
            return sub_power
        
        elif self.dataset == "space_shuttle":
            tek17 = pd.read_csv("C:/Users/Bin/Documents/Datasets/EncDec-AD dataset/TEK17.txt",header=None)
            tek16 = pd.read_csv("C:/Users/Bin/Documents/Datasets/EncDec-AD dataset/TEK16.txt",header=None)
            tek14 = pd.read_csv("C:/Users/Bin/Documents/Datasets/EncDec-AD dataset/TEK14.txt",header=None)
            tek = pd.concat([tek14,tek16,tek17],axis=0).reset_index(drop=True)
            # downsample the dataset by 3 
            sub_tek = pd.Series(tek[0])
            index = [3*t for t in range(tek.shape[0]//3)]
            sub_tek = sub_tek[index].reset_index(drop=True)
            #Scaling
            sub_tek = sub_tek.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaler.fit(sub_tek)
            sub_tek = scaler.transform(sub_tek) 
            #Applying sliding window, window length 1500/3=500, step_size 500/3= 166
            STEP_SIZE = 500//3 # downsampled by 3
            WINDOW_LENGTH = 1500//3   # downsampled by 3
            t = 1
            sequence = sub_tek[:WINDOW_LENGTH]
            while t*STEP_SIZE+WINDOW_LENGTH <=sub_tek.size:
                sequence = np.concatenate((sequence,sub_tek[t*STEP_SIZE:t*STEP_SIZE+WINDOW_LENGTH]))
                t = t+1
            return sequence
        
        elif self.dataset == "ecg":
            # use the first channel of the qtdb/sel102 dataset
            ecg = pd.read_csv("C:/Users/Bin/Documents/Datasets/EncDec-AD dataset/qtdbsel102.txt",header=None,usecols=[1],sep="\t")
            ecg = pd.Series(ecg[1])
            
            #Scaling
            ecg = ecg.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaler.fit(ecg)
            ecg = scaler.transform(ecg) 
            
            return ecg
            
            
        else:
            print("Wrong dataset name")
            return 

