# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 14:08:21 2018

@author: Bin
"""

import numpy as np
from scipy.spatial.distance import mahalanobis

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
            abs_err = abs(_input_ - _output_)
            err_vec_list += [abs_err[i] for i in range(abs_err.shape[0])]
            

        # new metric
        err_vec_array = np.array(err_vec_list).reshape(-1,self.conf.elem_num)
        
        # for univariate data, anomaly score is squared euclidean distance
        # for multivariate data, anomaly score is squared mahalanobis distance

        mu = np.mean(err_vec_array.ravel()) if self.conf.elem_num == 1 else np.mean(err_vec_array,axis=0)
        sigma = np.var(err_vec_array.ravel()) if self.conf.elem_num == 1 else np.cov(err_vec_array.T)

        print("Got parameters mu and sigma.")
        
        return mu, sigma
        

        
    def get_threshold(self,mu,sigma,sess,input_, output_,p_input, p_is_training):

            normal_score = []
            for count in range(len(self.conf.vn2_list)//self.conf.batch_num):
                normal_sub = np.array(self.conf.vn2_list[count*self.conf.batch_num:(count+1)*self.conf.batch_num]) 
                (input_n, output_n) = sess.run([input_, output_], {p_input: normal_sub,p_is_training : False})

                err_n = abs(input_n-output_n).reshape(-1,self.conf.step_num,self.conf.elem_num)
                for window in range(self.conf.batch_num):
                        for t in range(self.conf.step_num):
                            s = mahalanobis(err_n[window,t],mu,sigma)
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
                    err_a = abs(input_a-output_a).reshape(-1,self.conf.step_num,self.conf.elem_num)
                    for window in range(self.conf.batch_num):
                        for t in range(self.conf.step_num):
#                            temp = np.dot((err_a[window,t,:] - mu[t,:] ) , sigma[t])
#                            s = np.dot(temp,(err_a[window,t,:] - mu[t,:] ).T)
                            s = mahalanobis(err_a[window,t],mu,sigma)
                            if va_lable_list[window,t] == "normal":
                                normal_score.append(s)
                            else:
                                abnormal_score.append(s)
                
                upper = np.median(np.array(abnormal_score))
                lower = np.median(np.array(normal_score)) 
                scala = 20 # divide the range(min,max) into 20 parts, find the optimal threshold
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
