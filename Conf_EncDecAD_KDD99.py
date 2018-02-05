# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 09:09:00 2018

@author: Bin
"""
import sys
sys.path.insert(0, 'C:/Users/Bin/Desktop/Thesis/code')
from Data_helper import Data_Helper
#from EncDecAD import EncDecAD
import tensorflow as tf

class Conf_EncDecAD_KDD99(object):
    
    def __init__(self, input_root, optimizer=None, decode_without_input=False):
        
        self.batch_num = 20
        self.hidden_num = 100
        self.step_num = 20
        self.input_root = input_root
        self.iteration = 100
        self.modelpath_root = "C:/Users/Bin/Desktop/Thesis/tmp/52test/"
        self.modelpath = self.modelpath_root + "LSTMAutoencoder_kdd99_v1.ckpt"
        self.modelmeta = self.modelpath_root + "LSTMAutoencoder_kdd99_v1.ckpt.meta"
        self.decode_without_input =  False
        # import dataset
        # The dataset is divided into 6 parts, namely training_normal, validation_1,
        # validation_2, test_normal, validation_anomaly, test_anomaly.
        
        data_helper = Data_Helper(self.input_root,self.step_num)
        
        self.sn_list = data_helper.sn_list
        self.va_list = data_helper.va_list
        self.vn1_list = data_helper.vn1_list
        self.vn2_list = data_helper.vn2_list
        self.tn_list = data_helper.tn_list
        self.ta_list = data_helper.ta_list
        self.data_list = [self.sn_list, self.va_list, self.vn1_list, self.vn2_list, self.tn_list, self.ta_list]
        
        self.elem_num = data_helper.sn.shape[1]


        # input placeholder
#        self.p_input = tf.placeholder(tf.float32, shape=(self.batch_num, self.step_num, self.elem_num))
#        self.p_inputs = [tf.squeeze(t, [1]) for t in tf.split(self.p_input, self.step_num, 1)]
#        self.ae = EncDecAD(self.hidden_num, self.p_inputs,is_training = True, decode_without_input=False)


    
        
        