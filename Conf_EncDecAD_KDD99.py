# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 09:09:00 2018

@author: Bin
"""
import sys
sys.path.insert(0, 'C:/Users/Bin/Desktop/Thesis/code')
from Data_helper import Data_Helper

class Conf_EncDecAD_KDD99(object):
    
    def __init__(self,  optimizer=None, decode_without_input=False):
        
        self.batch_num = 20
        self.hidden_num = 100
        self.step_num = 20
        # data used for off-line training
        self.input_root = "C:/Users/Bin/Documents/Datasets/KDD99/kddcup.data_10_percent_corrected"  # to be edit to e.g. 5%KDD......
        self.iteration = 500
        self.modelpath_root = "C:/Users/Bin/Desktop/Thesis/tmp/EncDecADModel/"
        self.modelpath = self.modelpath_root + "LSTMAutoencoder_kdd99_v1.ckpt"
        self.modelmeta = self.modelpath_root + "LSTMAutoencoder_kdd99_v1.ckpt.meta"
        self.modelpath_p = self.modelpath_root + "LSTMAutoencoder_kdd99_v1_para.ckpt"
        self.modelmeta_p = self.modelpath_root + "LSTMAutoencoder_kdd99_v1_para.ckpt.meta"
        self.decode_without_input =  False
        
        
        # import dataset
        # The dataset is divided into 6 parts, namely training_normal, validation_1,
        # validation_2, test_normal, validation_anomaly, test_anomaly.
        
#        self.training_data_source = "stream"
#        data_helper = Data_Helper(None,self.step_num,self.training_data_source)
        
        self.training_data_source = "file"
        data_helper = Data_Helper(self.input_root,self.step_num,self.batch_num,self.training_data_source)
        
        self.sn_list = data_helper.sn_list
        self.va_list = data_helper.va_list
        self.vn1_list = data_helper.vn1_list
        self.vn2_list = data_helper.vn2_list
        self.tn_list = data_helper.tn_list
        self.ta_list = data_helper.ta_list
        self.data_list = [self.sn_list, self.va_list, self.vn1_list, self.vn2_list, self.tn_list, self.ta_list]
        
        self.elem_num = data_helper.sn.shape[1]
     