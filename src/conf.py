# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:44:42 2018

@author: Bin
"""

from dataHelper import Data_Helper

class Configuration(object):
    
    def __init__(self, dataset, dataPath, modelSavePath, training_data_source = "file", optimizer=None, decode_without_input=False):
        
        
        if dataset == "power":

            self.batch_num = 8
            self.hidden_num = 15
            self.step_num = 84
            self.training_set_size = self.step_num*12
            
        elif dataset == "smtp":
            
            self.batch_num = 8
            self.hidden_num = 15
            self.step_num = 10
            self.training_set_size = self.step_num*6000
            
        elif dataset == "http":
            
            self.batch_num = 8
            self.hidden_num = 35
            self.step_num = 30
            self.training_set_size = self.step_num*30000
            
        elif dataset == "smtphttp":
            self.batch_num = 8
            self.hidden_num = 15
            self.step_num = 10
            self.training_set_size = self.step_num*2500
            
        elif dataset == "forest":
            self.batch_num = 8
            self.hidden_num = 25
            self.step_num = 10
            self.training_set_size = self.step_num*10000
        
        else: 
            print("Wrong dataset name input.")
            
            
        self.input_root =dataPath 
        self.iteration = 300
        self.modelpath_root = modelSavePath
        self.modelmeta = self.modelpath_root + "_"+str(self.batch_num)+"_"+str(self.hidden_num)+"_"+str(self.step_num)+"_.ckpt.meta"
        self.modelpath_p = self.modelpath_root + "_"+str(self.batch_num)+"_"+str(self.hidden_num)+"_"+str(self.step_num)+"_para.ckpt"
        self.modelmeta_p = self.modelpath_root + "_"+str(self.batch_num)+"_"+str(self.hidden_num)+"_"+str(self.step_num)+"_para.ckpt.meta"
        self.decode_without_input =  False
        
        self.log_path = modelSavePath + "log.txt"
            
            
        # import dataset
        # The dataset is divided into 6 parts, namely training_normal, validation_1,
        # validation_2, test_normal, validation_anomaly, test_anomaly.
       
        self.training_data_source = training_data_source
        data_helper = Data_Helper(self.input_root,self.training_set_size,self.step_num,self.batch_num,self.training_data_source,self.log_path)
        
        self.sn_list = data_helper.sn_list
        self.va_list = data_helper.va_list
        self.vn1_list = data_helper.vn1_list
        self.vn2_list = data_helper.vn2_list
        self.tn_list = data_helper.tn_list
        self.ta_list = data_helper.ta_list
        self.data_list = [self.sn_list, self.va_list, self.vn1_list, self.vn2_list, self.tn_list, self.ta_list]
        
        self.elem_num = data_helper.sn.shape[1]
        self.va_label_list = data_helper.va_label_list 
        
        
        f = open(self.log_path,'a')
        f.write("Batch_num=%d\nHidden_num=%d\nwindow_length=%d\ntraining_used_#windows=%d\n"%(self.batch_num,self.hidden_num,self.step_num,self.training_set_size//self.step_num))
        f.close()