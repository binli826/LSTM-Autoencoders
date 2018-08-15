# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:53:36 2018

@author: Bin
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'C:/Users/Bin/Desktop/Thesis/code')
from ReTrainParaHelper import ReTrainParaHelper
class EncDecAD_ReTrain(object):
    
    def __init__(self,conf, sn,vn1,vn2,tn,va,ta,):
        self.sn = sn
        self.vn1 = vn1
        self.vn2 = vn2
        self.tn = tn
        self.va = va
        self.ta = ta
        self.retrain_iteration = conf.retrain_iteration
        self.batch_num = conf.batch_num
        self.step_num = conf.step_num
        self.elem_num = vn1.shape[1]
        # data seriealization
        t1 = self.sn.shape[0]//self.step_num
        t2 = self.va.shape[0]//self.step_num
        t3 = self.vn1.shape[0]//self.step_num
        t4 = self.vn2.shape[0]//self.step_num
        t5 = self.tn.shape[0]//self.step_num
        t6 = self.ta.shape[0]//self.step_num
        
        self.sn_list = [self.sn[self.step_num*i:self.step_num*(i+1)].as_matrix() for i in range(t1)]
        self.va_list = [self.va[self.step_num*i:self.step_num*(i+1)].as_matrix() for i in range(t2)]
        self.vn1_list = [self.vn1[self.step_num*i:self.step_num*(i+1)].as_matrix() for i in range(t3)]
        self.vn2_list = [self.vn2[self.step_num*i:self.step_num*(i+1)].as_matrix() for i in range(t4)]
        
        self.tn_list = [self.tn[self.step_num*i:self.step_num*(i+1)].as_matrix() for i in range(t5)]
        self.ta_list = [self.ta[self.step_num*i:self.step_num*(i+1)].as_matrix() for i in range(t6)]
        
    def continue_training(self,sess,loss_, train_,p_input,p_inputs,p_is_training,input_,output_):
        loss = []
        for i in range(self.retrain_iteration):
            data =[]
            for temp in range(self.batch_num):
                ind = np.random.randint(0,len(self.sn_list)-1)
                sub = self.sn_list[ind]
                data.append(sub)
            data = np.array(data)
            (loss_val, _) = sess.run([loss_, train_], {p_input: data,p_is_training : True})
            loss.append(loss_val)
            print('Retrain-iter %d:' % (i + 1), loss_val)
        pd.Series(loss).plot(title="Loss")
        
        
        # mu & sigma & threshold
        
        para = ReTrainParaHelper(self.vn1_list,self.vn2_list,self.va_list,self.batch_num,self.step_num,self.elem_num)
        mu, sigma = para.mu_and_sigma(sess,input_, output_,p_input, p_is_training)
        threshold = para.get_threshold(mu,sigma,sess,input_, output_,p_input, p_is_training)
        print("Threshold:%.3f"%threshold)
        return mu,sigma,threshold,loss
    
#    def start_from_scratch(self,sess,)
    