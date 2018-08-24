# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:14:24 2017

@author: Bin
"""

import argparse
import pandas as pd
import numpy as np

def parseArguments():
    parser = argparse.ArgumentParser()
    # Positional mandatory arguments
    parser.add_argument("dataset", help="power/kdd/forest", type=str)
    parser.add_argument("datapath", help="input data path", type=str)
    parser.add_argument("savepath", help="folder to save the processed data", type=str)
    
    # Optional arguments
    parser.add_argument("-pl", "--powerlabel", help="Label file of power demand dataset ", type=str)
    parser.add_argument("-kc", "--kddcol", help="Column file of KDD dataset ", type=str)

    # Parse arguments
    args = parser.parse_args()
    
    return args



def power(pathPowerData, pathPowerLabel, pathPowerSave):
    '''Process PowerDemand dataset
    
        we do downsampling with rate 8 on the PowerDemand dataset, 
        and remove the first and last half week, preserve only the 51 full weeks 
        with each week benn described by 84 instances.
    '''

    powerData = pd.read_csv(pathPowerData,header=None)[::8][60:-36].reset_index(drop=True)
    
    
    # The PowerDemand dataset is manually label according to the special days in a year.
    
    
    powerLabel = pd.read_csv(pathPowerLabel,header=None)
    
    PowerDemand = pd.concat((powerData,powerLabel),axis=1)
    
    PowerDemand.to_csv(pathPowerSave+"/PowerDemand.csv",header=None,index=None)


def kdd(pathKDDcol_name, pathKDD, pathKDDSave):
    '''Process KDD99 dataset
    
        SMTP and HTTP are extracted from the KDD99 dataset, only with numerical features
        SMTP+HTTP is a connection of SMTP and HTTP
    '''

    try: 
        with open(pathKDDcol_name) as col_file:
            line = col_file.readline()
    except EnvironmentError:
        print('File not found.')
        
    columns = line.split('.')
    col_names = []
    col_types = []
    for col in columns:
        col_names.append(col.split(': ')[0].strip())
        col_types.append(col.split(': ')[1])    
    
    
    df = pd.read_csv(pathKDD,header=None)

    continuous = df.iloc[:,np.array(pd.Series(col_types)=="continuous")]
    continuous = pd.concat((continuous,df.iloc[:,-1]),axis=1)
    SMTP = continuous[df.iloc[:,2] == "smtp"].reset_index(drop=True)
    HTTP = continuous[df.iloc[:,2] == "http"].reset_index(drop=True)
    SMTPHTTP = pd.concat((SMTP,HTTP),axis=0).reset_index(drop=True)
    
    SMTP.to_csv(pathKDDSave+"/SMTP.csv",header=None,index=None)
    HTTP.to_csv(pathKDDSave+"/HTTP.csv",header=None,index=None)
    SMTPHTTP.to_csv(pathKDDSave+"/SMTPHTTP.csv",header=None,index=None)

def forest(pathForestData,pathForestSave):

    '''Process Forest cover dataset
    
        take the smallest class TYP4 as anomaly, rest are normal
        only use the 7 numerical features
    '''
    
    forest = pd.read_csv(pathForestData,header=None)
    numerical_col = [0,1,2,3,4,5,9]
    
    forestData = forest.iloc[:,numerical_col]
    forestLabel = forest.iloc[:,-1]
    forestLabel[forestLabel != 4] = 'normal.'
    forestLabel[forestLabel == 4] = 'anomaly.'
    
    forest = pd.concat((forestData,forestLabel),axis=1)
    forest.to_csv(pathForestSave+"/FOREST.csv",header=None,index=None)


if __name__ == '__main__':

    args = parseArguments()
    dataset = args.__dict__['dataset']
    datapath = args.__dict__['datapath']
    savepath= args.__dict__['savepath']

    if dataset == "power":
       powerlabel = args.__dict__['powerlabel']
       power(datapath,powerlabel,savepath)
    elif dataset == "kdd":
       kddcol = args.__dict__['kddcol']
       kdd(kddcol,datapath,savepath)
    elif dataset == "forest":
        forest(datapath,savepath)
    else:
        print("Please input the dataset name: power/kdd/forest.")
