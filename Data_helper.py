# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 09:20:38 2018

@author: Bin
"""

import pandas as pd

class Data_Helper(object):
    
    def __init__(self, root,step_num):
        self.root = root
        self.step_num = step_num
        
        self.sn = pd.read_csv(self.root + "training_normal.csv",header=None)
        self.vn1 = pd.read_csv(self.root + "validation_1.csv",header=None)
        self.vn2 = pd.read_csv(self.root + "validation_2.csv",header=None)
        self.tn = pd.read_csv(self.root + "test_normal.csv",header=None)

        self.va = pd.read_csv(self.root + "validation_anomaly.csv",header=None)
        self.ta = pd.read_csv(self.root + "test_anomaly.csv",header=None)    

        # data seriealization
        t1 = self.sn.shape[0]//step_num
        t2 = self.va.shape[0]//step_num
        t3 = self.vn1.shape[0]//step_num
        t4 = self.vn2.shape[0]//step_num
        t5 = self.tn.shape[0]//step_num
        t6 = self.ta.shape[0]//step_num
        
        self.sn_list = [self.sn[step_num*i:step_num*(i+1)].as_matrix() for i in range(t1)]
        self.va_list = [self.va[step_num*i:step_num*(i+1)].as_matrix() for i in range(t2)]
        self.vn1_list = [self.vn1[step_num*i:step_num*(i+1)].as_matrix() for i in range(t3)]
        self.vn2_list = [self.vn2[step_num*i:step_num*(i+1)].as_matrix() for i in range(t4)]
        
        self.tn_list = [self.tn[step_num*i:step_num*(i+1)].as_matrix() for i in range(t5)]
        self.ta_list = [self.ta[step_num*i:step_num*(i+1)].as_matrix() for i in range(t6)]

