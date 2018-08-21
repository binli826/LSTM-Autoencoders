# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 15:53:12 2018

@author: Bin
"""

class Conf(object):
    
    def __init__(self, dataset):
        
        
        if dataset == "power":

            self.batch_num = 8
            self.hidden_num = 15
            self.step_num = 84
            self.elem_num = 1
            self.training_set_size = self.step_num*12
            self.HardCriterion = 5
            self.buffersize = 9# number of batches
        elif dataset == "smtp":
            
            self.batch_num = 8
            self.hidden_num = 15
            self.step_num = 10
            self.elem_num = 34
            self.training_set_size = self.step_num*6000
            self.HardCriterion = 5
            self.buffersize = 50# number of batches
        elif dataset == "http":
            
            self.batch_num = 8
            self.hidden_num = 35
            self.step_num = 30
            self.elem_num = 34
            self.training_set_size = self.step_num*30000
            self.HardCriterion = 5
            self.buffersize = 1000 # number of batches

        elif dataset == "smtphttp":
            self.batch_num = 8
            self.hidden_num = 15
            self.step_num = 10
            self.elem_num = 34
            self.training_set_size = self.step_num*2500
            self.HardCriterion = 5
            self.buffersize = 1500# number of batches

        elif dataset == "forest":
            self.batch_num = 8
            self.hidden_num = 25
            self.step_num = 10
            self.elem_num = 7
            self.training_set_size = self.step_num*10000
            self.HardCriterion = 4
            self.buffersize = 400 # number of batches

        else: 
            print("Wrong dataset name input.")
         
        