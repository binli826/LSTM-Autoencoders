# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 10:04:22 2018

@author: Bin
"""
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
sys.path.insert(0, 'C:/Users/Bin/Desktop/Thesis/code')
from EncDecAD import EncDecAD

class Parameter_Helper(object):
    
    def __init__(self, conf):
        self.conf = conf
        self.is_training = False
        
        
    def mu_and_sigma(self):
        
#        ae = EncDecAD(self.conf.hidden_num, self.conf.p_inputs,self.is_training, self.conf.decode_without_input )
#        ae = self.conf.ae
      
        with tf.Session() as sess:
#            tf.reset_default_graph()
#            saver = tf.train.Saver()
            
            saver = tf.train.import_meta_graph(self.conf.modelmeta) # load trained gragh, but without the trained parameters
            saver.restore(sess,self.conf.modelpath) # reload trained parameters
            graph = tf.get_default_graph()


            _input_ = graph.get_tensor_by_name("input_:0")
            _output_ = graph.get_tensor_by_name("output_:0")
            
            
            print("Model restored.") 
            print('Initialized')
            
            
#            init = tf.global_variables_initializer()
#            sess.run(init)
        
            err_vec_list = []
            for _ in range(len(self.conf.vn1_list)//self.conf.batch_num):
                data =[]
                for temp in range(self.conf.batch_num):
                    ind = np.random.randint(0,len(self.conf.vn1_list)-1)
                    sub = self.conf.vn1_list[ind]
                    data.append(sub)
                data = np.array(data)
#                (input_, output_) = sess.run([ae.input_, ae.output_], {self.conf.p_input: data})
                
                (input_, output_) = sess.run([_input_, _output_], {self.conf.p_input: data})
                err_vec_list.append(abs(input_ - output_))
            err_vec = np.mean(np.array(err_vec_list),axis=0).reshape(self.conf.batch_num,-1)
            mu = np.mean(err_vec,axis=0)
            sigma = np.cov(err_vec.T)
            print("Got parameters mu and sigma.")
            
            return mu, sigma
        

        
    def get_threshold(self,mu,sigma):
        
#        ae = EncDecAD(self.conf.hidden_num, self.conf.p_inputs,self.is_training, self.conf.decode_without_input )
        ae = self.conf.ae
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(self.conf.modelmeta)
            saver.restore(sess,self.conf.modelpath)
            init = tf.global_variables_initializer()
            sess.run(init)
            normal_score = []
            for count in range(len(self.conf.vn2_list)//self.conf.batch_num):
                normal_sub = np.array(self.conf.vn2_list[count*self.conf.batch_num:(count+1)*self.conf.batch_num]) 
                (input_n, output_n) = sess.run([ae.input_, ae.output_], {self.conf.p_input: normal_sub})
                err_n = abs(input_n-output_n).reshape(-1,self.conf.step_num)
                err_n = err_n.reshape(self.conf.batch_num,-1)
                for batch in range(self.conf.batch_num):
                   temp = np.dot( (err_n[batch] - mu ).reshape(1,-1)  , sigma.T)
                   s = np.dot(temp,(err_n[batch] - mu ))
                   normal_score.append(s[0])
                   
            abnormal_score = []
            for count in range(len(self.conf.va_list)//self.conf.batch_num):
                abnormal_sub = np.array(self.conf.va_list[count*self.conf.batch_num:(count+1)*self.conf.batch_num]) 
                (input_a, output_a) = sess.run([ae.input_, ae.output_], {self.conf.p_input: abnormal_sub})
                err_a = abs(input_a-output_a).reshape(-1,self.conf.step_num)
                err_a = err_a.reshape(self.conf.batch_num,-1)
                for batch in range(self.conf.batch_num):
                   temp = np.dot( (err_a[batch] - mu ).reshape(1,-1)  , sigma.T)
                   s = np.dot(temp,(err_a[batch] - mu ))
                   abnormal_score.append(s[0])      
            print('Finished')
            

        
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
            # anomaly score of vn2 and va dataset
            pd.Series(normal_score).plot(figsize=(18,5))
            pd.Series(abnormal_score).plot()
            bar = threshold*np.ones(len(normal_score)+len(abnormal_score))
            pd.Series(bar).plot(label="threshold")

            v_threshold = tf.constant(threshold,name="v_threshold")
            v_mu = tf.constant(mu,name="v_mu")
            v_sigma = tf.constant(sigma,name="v_sigma")

            save_path = saver.save(sess, self.conf.modelpath)
            print("Model saved at: ",self.conf.modelpath)
        
        
        
        
        return threshold
  