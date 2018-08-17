# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:44:42 2018

@author: Bin
"""

import argparse
from initTrain import Initialization_Train

def parseArguments():
    parser = argparse.ArgumentParser()
    # Positional mandatory arguments
    parser.add_argument("dataset", help="power/smtp/http/forest", type=str)
    parser.add_argument("dataPath", help="input data path", type=str)
    parser.add_argument("modelSavePath", help="folder to save the trained model", type=str)
    
    # Parse arguments
    args = parser.parse_args()
    
    return args

if __name__=="__main__":
    
    args = parseArguments()
    dataset = args.__dict__['dataset']
    dataPath = args.__dict__['dataPath']
    modelSavePath= args.__dict__['modelSavePath']

    Initialization_Train(dataset,dataPath, modelSavePath)


    