# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 09:09:00 2018

@author: Bin
"""
import time
import os

class Conf_Prediction_KDD99(object):
    
    def __init__(self,):
        
#        self.train_with_stream = True
        self.batch_num = 1#20
        self.hidden_num = 40#100
        self.step_num = 84#20
        self.iteration = 300
        self.retrain_iteration = 100
        self.min_test_block_num = 1#2
        self.min_retrain_block_num = 2#5
#        self.modelpath_root = "C:/Users/Bin/Desktop/Thesis/tmp/EncDecADModel/"
        self.modelpath_root ="C:/Users/Bin/Desktop/Thesis/tmp/EncDecADModel_online_init/power_demand/Try7_1_40_84/"#"C:/Users/Bin/Desktop/Thesis/tmp/EncDecADModel_online_init/power_demand/"
        self.modelmeta = self.modelpath_root + "LSTMAutoencoder_power_"+str(self.batch_num)+"_"+str(self.hidden_num)+"_"+str(self.step_num)+"_.ckpt.meta"
        self.modelpath_p = self.modelpath_root + "LSTMAutoencoder_power_"+str(self.batch_num)+"_"+str(self.hidden_num)+"_"+str(self.step_num)+"_para.ckpt"
        self.modelmeta_p = self.modelpath_root + "LSTMAutoencoder_power_"+str(self.batch_num)+"_"+str(self.hidden_num)+"_"+str(self.step_num)+"_para.ckpt.meta"
    
#        self.modelpath = self.modelpath_root + "LSTMAutoencoder_smtp_v1.ckpt"
#        self.modelmeta = self.modelpath_root + "LSTMAutoencoder_smtp_v1.ckpt.meta"
#        self.modelpath_p = self.modelpath_root + "LSTMAutoencoder_smtp_v1_para.ckpt"
#        self.modelmeta_p = self.modelpath_root + "LSTMAutoencoder_smtp_v1_para.ckpt.meta"
        self.decode_without_input =  False
        self.column_name_file = "C:/Users/Bin/Documents/Datasets/KDD99/columns.txt"
#        self.class_label_path = "C:/Users/Bin/Documents/Datasets/KDD99/classes.txt"
        self.class_label_path = "C:/Users/Bin/Documents/Datasets/EncDec-AD dataset/power_demand_classlabel.txt"
        tmp = 0
        while True:
            tmp+=1
            plot_savepath = "C:/Users/Bin/Desktop/Thesis/Plotting/"+str(tmp)+"/"        
            if  not os.path.exists(plot_savepath):                
                self.plot_savepath = plot_savepath
                os.makedirs(self.plot_savepath)
                os.makedirs(self.plot_savepath+"Predictions/")
                break
            else:
                continue