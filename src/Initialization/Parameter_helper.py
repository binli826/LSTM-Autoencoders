# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 10:04:22 2018

@author: Bin
"""
import sys
import tensorflow as tf
import numpy as np
import pandas as pd

class Parameter_Helper(object):
    
    def __init__(self, conf):
        self.conf = conf
       
        
    def mu_and_sigma(self,sess,input_, output_,p_input, p_is_training):

        err_vec_list = []
        
        ind = list(np.random.permutation(len(self.conf.vn1_list)))
        while len(ind)>=self.conf.batch_num:
            data = []
            for _ in range(self.conf.batch_num):
                data += [self.conf.vn1_list[ind.pop()]]
            data = np.array(data,dtype=float)
            data = data.reshape((self.conf.batch_num,self.conf.step_num,self.conf.elem_num))

            (_input_, _output_) = sess.run([input_, output_], {p_input: data, p_is_training: False})
            err_vec_list.append(abs(_input_ - _output_))

            

        # new metric
        err_vec_array = np.array(err_vec_list).reshape(-1,self.conf.step_num,self.conf.elem_num)
        mu = np.mean(err_vec_array,axis=0)
        sigma = [] # len(sigma) = step_num, each element has shape [elem_num*elem_num]
#        err_matrix = pd.DataFrame(err_vec_list)
        for i in range(err_vec_array.shape[1]):
            sigma.append(np.cov(err_vec_array[:,i,:].T))
        sigma = np.array(sigma)
        print("Got parameters mu and sigma.")
        
        return mu, sigma
        

        
    def get_threshold(self,mu,sigma,sess,input_, output_,p_input, p_is_training):

            normal_score = []
            for count in range(len(self.conf.vn2_list)//self.conf.batch_num):
                normal_sub = np.array(self.conf.vn2_list[count*self.conf.batch_num:(count+1)*self.conf.batch_num]) 
                (input_n, output_n) = sess.run([input_, output_], {p_input: normal_sub,p_is_training : False})
#                err_n = abs(input_n-output_n).reshape(-1,self.conf.step_num)
#                err_n = err_n.reshape(self.conf.batch_num,-1)
                err_n = abs(input_n-output_n).reshape(-1,self.conf.step_num,self.conf.elem_num)
                
#                for window in range(self.conf.batch_num):
#                   temp = np.dot( (err_n[window] - mu ).reshape(1,-1)  , sigma.T)
#                   s = np.dot(temp,(err_n[window] - mu ))
#                   normal_score.append(s[0])
                   
                for window in range(self.conf.batch_num):
                    win_a = []
                    for t in range(self.conf.step_num):
                        temp = np.dot((err_n[window,t,:] - mu[t,:] ) , sigma[t])
                        s = np.dot(temp,(err_n[window,t,:] - mu[t,:] ).T)
#                        win_a.append(s)
                        normal_score.append(s)
                    
            abnormal_score = []
            '''
            if have enough anomaly data, then calculate anomaly score, and the 
            threshold that achives best f1 score as divide boundary.
            otherwise estimate threshold through normal scores
            '''
            print(len(self.conf.va_list))
            
            if len(self.conf.va_list) < self.conf.batch_num: # not enough anomaly data for a single batch
                threshold = max(normal_score) * 2
                print("Not enough large va set.")
                
            else:
            
                for count in range(len(self.conf.va_list)//self.conf.batch_num):
                    abnormal_sub = np.array(self.conf.va_list[count*self.conf.batch_num:(count+1)*self.conf.batch_num]) 
                    va_lable_list = np.array(self.conf.va_label_list[count*self.conf.batch_num:(count+1)*self.conf.batch_num]) 
                    va_lable_list = va_lable_list.reshape(self.conf.batch_num,self.conf.step_num)
                    
                    (input_a, output_a) = sess.run([input_, output_], {p_input: abnormal_sub,p_is_training : False})
#                    err_a = abs(input_a-output_a).reshape(-1,self.conf.step_num)
#                    err_a = err_a.reshape(self.conf.batch_num,-1)
                    err_a = abs(input_a-output_a).reshape(-1,self.conf.step_num,self.conf.elem_num)
#                    for batch in range(self.conf.batch_num):
#                       temp = np.dot( (err_a[batch] - mu ).reshape(1,-1)  , sigma.T)
#                       s = np.dot(temp,(err_a[batch] - mu ))
#                       abnormal_score.append(s[0])      
                    for window in range(self.conf.batch_num):
                        win_a = []
                        for t in range(self.conf.step_num):
                            temp = np.dot((err_a[window,t,:] - mu[t,:] ) , sigma[t])
                            s = np.dot(temp,(err_a[window,t,:] - mu[t,:] ).T)
                            if va_lable_list[window,t] == "normal":
                                normal_score.append(s)
                            else:
#                                win_a.append(s)
                                abnormal_score.append(s)
                
                upper = np.median(np.array(abnormal_score))
                lower = np.median(np.array(normal_score)) 
                scala = 20
                delta = (upper-lower) / scala
                candidate = lower
                threshold = 0
                result = 0
                
                def evaluate(threshold,normal_score,abnormal_score):
                    beta = 0.5
                    tp = np.array(abnormal_score)[np.array(abnormal_score)>threshold].size
                    fp = len(abnormal_score)-tp
                    fn = np.array(normal_score)[np.array(normal_score)>threshold].size
                    tn = len(normal_score)- fn
                    
                    if tp == 0: return 0
                    
                    P = tp/(tp+fp)
                    R = tp/(tp+fn)
                    fbeta= (1+beta*beta)*P*R/(beta*beta*P+R)
                    return fbeta 
                
                for _ in range(scala):
                    r = evaluate(candidate,normal_score,abnormal_score)
                    if r > result:
                        result = r 
                        threshold = candidate
                    candidate += delta 
            
            print("Threshold: ",threshold)

            return threshold

  
  
