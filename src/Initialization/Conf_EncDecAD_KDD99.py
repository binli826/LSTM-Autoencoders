# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 09:09:00 2018

@author: Bin
"""
import sys
sys.path.insert(0, 'C:/Users/Bin/Desktop/Thesis/code/src/Initialization')
from Data_helper import Data_Helper

class Conf_EncDecAD_KDD99(object):
    
    def __init__(self, training_data_source = "file", optimizer=None, decode_without_input=False):
        
        self.batch_num = 1
        self.hidden_num = 40
        self.step_num = 84
        self.input_root ="C:/Users/Bin/Documents/Datasets/EncDec-AD dataset/power_data_labeled.csv"
        self.iteration = 500
        self.modelpath_root = "C:/Users/Bin/Desktop/Thesis/tmp/EncDecADModel_online_init/power_demand/Try9_1_40_84/"
        
        self.modelmeta = self.modelpath_root + "_"+str(self.batch_num)+"_"+str(self.hidden_num)+"_"+str(self.step_num)+"_.ckpt.meta"
        self.modelpath_p = self.modelpath_root + "_"+str(self.batch_num)+"_"+str(self.hidden_num)+"_"+str(self.step_num)+"_para.ckpt"
        self.modelmeta_p = self.modelpath_root + "_"+str(self.batch_num)+"_"+str(self.hidden_num)+"_"+str(self.step_num)+"_para.ckpt.meta"
        self.decode_without_input =  False
        self.data_helper_plot_path = "C:/Users/Bin/Desktop/Thesis/tmp/EncDecADModel_online_init/power_demand/"
        self.training_set_size = 84*13#20000
        # import dataset
        # The dataset is divided into 6 parts, namely training_normal, validation_1,
        # validation_2, test_normal, validation_anomaly, test_anomaly.
        
#        self.training_data_source = "stream"
#        data_helper = Data_Helper(None,self.step_num,self.training_data_source)
        
        self.training_data_source = training_data_source
        data_helper = Data_Helper(self.input_root,self.training_set_size,self.step_num,self.batch_num,self.training_data_source,self.data_helper_plot_path)
        
        self.sn_list = data_helper.sn_list
        self.va_list = data_helper.va_list
        self.vn1_list = data_helper.vn1_list
        self.vn2_list = data_helper.vn2_list
        self.tn_list = data_helper.tn_list
        self.ta_list = data_helper.ta_list
        self.data_list = [self.sn_list, self.va_list, self.vn1_list, self.vn2_list, self.tn_list, self.ta_list]
        
        self.elem_num = data_helper.sn.shape[1]
        self.va_label_list = data_helper.va_label_list 