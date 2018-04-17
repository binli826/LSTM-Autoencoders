# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 09:48:10 2018

@author: Bin


Select continuously features from KDDCup99 dataset, and rename label 'normal.' as 'normal',
keep the rest labels not being changed.

"""

import pandas as pd
import numpy as np

column_name_file = "C:/Users/Bin/Documents/Datasets/KDD99/columns.txt"
input_path = "C:/Users/Bin/Documents/Datasets/KDD99/kddcup.data.corrected"
output_path = "C:/Users/Bin/Documents/Datasets/KDD99/Continuously/kddcup99.csv"

with open(column_name_file) as col_file:
    line = col_file.readline()
columns = line.split('.')
col_names = []
col_types = []
for col in columns:
    col_names.append(col.split(': ')[0].strip())
    col_types.append(col.split(': ')[1])    


df = pd.read_csv(input_path,header=None).iloc[:,1:]

continuous = df.iloc[:,np.array(pd.Series(col_types)=="continuous")]
label = df.iloc[:,-1]
label[label=='normal.'] = 'normal'

continuous = pd.concat((continuous,label),axis=1)

continuous.to_csv(output_path,header=None,index=None)
