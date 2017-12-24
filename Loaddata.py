
# coding: utf-8

# # Load Dataset

# In[9]:


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
        
        elif self.dataset == "tek17":
            tek17 = pd.read_csv("C:/Users/Bin/Documents/Datasets/EncDec-AD dataset/TEK17.txt",names=["tek17"])
            # downsample the dataset by 3 
            tek17 = pd.Series(tek17[:]["tek17"])
            index = [3*t for t in range(tek17.shape[0]//3)]
            sub_tek17 = tek17[index].reset_index(drop=True)
            sub_tek17 = sub_tek17[:1500]
            
            return sub_tek17
            
        else:
            print("Wrong dataset name")
            return 

