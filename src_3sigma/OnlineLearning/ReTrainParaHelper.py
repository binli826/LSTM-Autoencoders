# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 17:21:40 2018

@author: Bin
"""

import numpy as np
import math

class ReTrainParaHelper(object):

    
    def __init__(self, vn1_list,vn2_list,va_list,batch_num,step_num,elem_num,va_label_list):
        self.vn1_list=vn1_list
        self.batch_num=batch_num
        self.step_num=step_num
        self.elem_num=elem_num
        self.vn2_list=vn2_list
        self.va_list=va_list
        self.va_label_list = va_label_list
        
    def mu_and_sigma(self,sess,input_, output_,p_input, p_is_training):

        err_vec_list = []
        
        
        ind = list(np.random.permutation(len(self.vn1_list)))
        while len(ind)>=self.batch_num:
            data = []
            for _ in range(self.batch_num):
                data += [self.vn1_list[ind.pop()]]
            data = np.array(data,dtype=float)
            data = data.reshape((self.batch_num,self.step_num,self.elem_num))

            (_input_, _output_) = sess.run([input_, output_], {p_input: data, p_is_training: False})
            err_vec_list.append(abs(_input_ - _output_))
        
        
#        for _ in range(len(self.vn1_list)//self.batch_num):
#            data =[]
#            for temp in range(self.batch_num):
#                ind = np.random.randint(0,len(self.vn1_list)-1)
#                sub = self.vn1_list[ind]
#                data.append(sub)
#            data = np.array(data,dtype=float)
#            data = data.reshape((self.batch_num,self.step_num,self.elem_num))
#
#            (_input_, _output_) = sess.run([input_, output_], {p_input: data, p_is_training: False})
#            err_vec_list.append(abs(_input_ - _output_))

#        err_vec = np.mean(np.array(err_vec_list),axis=0).reshape(self.batch_num,-1)
#        mu = np.mean(err_vec,axis=0)
#        sigma = np.cov(err_vec.T)
#        print("Got parameters mu and sigma.")
        
        
        err_vec_array = np.array(err_vec_list).reshape(-1,self.step_num,self.elem_num)
        mu = np.mean(err_vec_array,axis=0)
        sigma = []
        for i in range(err_vec_array.shape[1]):
            sigma.append(np.cov(err_vec_array[:,i,:].T))
        sigma = np.array(sigma)
        print("Got parameters mu and sigma.")
        
        return mu, sigma
        

        
    def get_threshold(self,mu,sigma,sess,input_, output_,p_input, p_is_training):

            normal_score = []
            for count in range(len(self.vn2_list)//self.batch_num):
                normal_sub = np.array(self.vn2_list[count*self.batch_num:(count+1)*self.batch_num]) 
                (input_n, output_n) = sess.run([input_, output_], {p_input: normal_sub,p_is_training : False})
                err_n = abs(input_n-output_n).reshape(-1,self.step_num,self.elem_num)
                
                for window in range(self.batch_num):
                    for t in range(self.step_num):
                        temp = np.dot((err_n[window,t,:] - mu[t,:] ) , sigma[t])
                        s = np.dot(temp,(err_n[window,t,:] - mu[t,:] ).T)
                        normal_score.append(s)
                        
                        
                        
                        
                        
#            normal_score = []
#            for count in range(len(self.vn2_list)//self.batch_num):
#                normal_sub = np.array(self.vn2_list[count*self.batch_num:(count+1)*self.batch_num]) 
#                (input_n, output_n) = sess.run([input_, output_], {p_input: normal_sub,p_is_training : False})
#                err_n = abs(input_n-output_n).reshape(-1,self.step_num)
#                err_n = err_n.reshape(self.batch_num,-1)
#                for batch in range(self.batch_num):
#                   temp = np.dot( (err_n[batch] - mu ).reshape(1,-1)  , sigma.T)
#                   s = np.dot(temp,(err_n[batch] - mu ))
#                   normal_score.append(s[0])
                   
            abnormal_score = []
            '''
            if have enough anomaly data, then calculate anomaly score, and the 
            threshold that achives best f1 score as divide boundary.
            otherwise estimate threshold through normal scores
            '''
            if len(self.va_list) < self.batch_num: # not enough anomaly data for a single batch
                threshold = np.mean(np.array(normal_score)) * 2 if np.mean(np.array(normal_score)) * 2<10 else float('nan')
                
                return threshold
            else:
                
#                for count in range(len(self.va_list)//self.batch_num):
#                    abnormal_sub = np.array(self.va_list[count*self.batch_num:(count+1)*self.batch_num]) 
#                    (input_a, output_a) = sess.run([input_, output_], {p_input: abnormal_sub,p_is_training : False})
#                    err_a = abs(input_a-output_a).reshape(-1,self.step_num)
#                    err_a = err_a.reshape(self.batch_num,-1)
#                    for batch in range(self.batch_num):
#                       temp = np.dot( (err_a[batch] - mu ).reshape(1,-1)  , sigma.T)
#                       s = np.dot(temp,(err_a[batch] - mu ))
#                       abnormal_score.append(s[0])      
#                       
                for count in range(len(self.va_list)//self.batch_num):
                    abnormal_sub = np.array(self.va_list[count*self.batch_num:(count+1)*self.batch_num]) 
                    va_lable_list = np.array(self.va_label_list[count*self.batch_num:(count+1)*self.batch_num]) 
                    va_lable_list = va_lable_list.reshape(self.batch_num,self.step_num)
                    
                    (input_a, output_a) = sess.run([input_, output_], {p_input: abnormal_sub,p_is_training : False})
                    err_a = abs(input_a-output_a).reshape(-1,self.step_num,self.elem_num)
                    for window in range(self.batch_num):
                        win_a = []
                        for t in range(self.step_num):
                            temp = np.dot((err_a[window,t,:] - mu[t,:] ) , sigma[t])
                            s = np.dot(temp,(err_a[window,t,:] - mu[t,:] ).T)
                            if va_lable_list[window,t] == "normal":
                                normal_score.append(s)
                            else:
                                abnormal_score.append(s)      
                       
                
                upper = np.median(np.array(abnormal_score))
                lower = np.median(np.array(normal_score)) 
                
                if math.isnan(upper) or math.isnan(lower):
                    print("Bad new threshold, remain original parameters.")
                    return float('nan')
                else:
                    scala = 20
                    delta = (upper-lower) / scala
                    candidate = lower
                    threshold = 0
                    result = 0
                    
                    def evaluate(threshold,normal_score,abnormal_score):
                        beta = 1
                        tp = np.array(abnormal_score)[np.array(abnormal_score)>threshold].size
                        fp = len(abnormal_score)-tp
                        fn = np.array(normal_score)[np.array(normal_score)>threshold].size
                        tn = len(normal_score)- fn
                        if min(tp,fn,fp) != 0:
                            P = tp/(tp+fp)
                            R = tp/(tp+fn)
                            fbeta= (1+beta*beta)*P*R/(beta*beta*P+R)
                        else:
                            fbeta = -1
                        return fbeta 
                    
                    for _ in range(scala):
                        r = evaluate(candidate,normal_score,abnormal_score)
                        if r > result:
                            result = r 
                            threshold = candidate
                        candidate += delta 
                        
                    if result >=0.7: # if evaluation result fbeta >=0.7, then take the new threshold, otherwise giveup
                        print("Threshold: ",threshold)        
                        return threshold
                    else:
                        print("Bad retrain performance, remain original parameters.")
                        return float('nan')

  
  
