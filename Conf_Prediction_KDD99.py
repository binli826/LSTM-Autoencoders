# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 09:09:00 2018

@author: Bin
"""


class Conf_Prediction_KDD99(object):
    
    def __init__(self,):
        
        self.batch_num = 20
        self.hidden_num = 100
        self.step_num = 20
        self.iteration = 1000
        self.modelpath_root = "C:/Users/Bin/Desktop/Thesis/tmp/EncDecADModel/"
        self.modelpath = self.modelpath_root + "LSTMAutoencoder_kdd99_v1.ckpt"
        self.modelmeta = self.modelpath_root + "LSTMAutoencoder_kdd99_v1.ckpt.meta"
        self.modelpath_p = self.modelpath_root + "LSTMAutoencoder_kdd99_v1_para.ckpt"
        self.modelmeta_p = self.modelpath_root + "LSTMAutoencoder_kdd99_v1_para.ckpt.meta"
        self.decode_without_input =  False
        self.column_name_file = "C:/Users/Bin/Documents/Datasets/KDD99/columns.txt"
     