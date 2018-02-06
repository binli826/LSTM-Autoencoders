# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:02:59 2018

@author: Bin
"""

import tensorflow as tf
import numpy as np
import pandas as pd


class EncDecAD_Test(object):
    
    def __init__(self, conf):
        self.conf = conf
    
    def test_encdecad(self,beta = 0.5):
        
        with tf.Session() as sess:
            normal_score = []
            n_in = []
            n_out = []
            a_in = []
            a_out = []
            
            saver = tf.train.import_meta_graph(self.conf.modelmeta_p) # load trained gragh, but without the trained parameters
            saver.restore(sess,tf.train.latest_checkpoint(self.conf.modelpath_root))
            graph = tf.get_default_graph()
            
            p_input = graph.get_tensor_by_name("p_input:0")
            p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, self.conf.step_num, 1)] 
            p_is_training = tf.placeholder(tf.bool)

            input_= tf.transpose(tf.stack(p_inputs), [1, 0, 2])    
            output_ = graph.get_tensor_by_name("decoder/output_:0")
            
            tensor_mu = graph.get_tensor_by_name("mu:0")
            tensor_sigma = graph.get_tensor_by_name("sigma:0")
            tensor_threshold = graph.get_tensor_by_name("threshold:0")
            
            mu = sess.run(tensor_mu)
            sigma = sess.run(tensor_sigma)
            threshold = sess.run(tensor_threshold)
            
            for count in range(len(self.conf.tn_list)//self.conf.batch_num):
                normal_sub = np.array(self.conf.tn_list[count*self.conf.batch_num:(count+1)*self.conf.batch_num]) 
                (input_n, output_n) = sess.run([input_, output_], {p_input: normal_sub, p_is_training: False})
                n_in.append(input_n)
                n_out.append(output_n)
                err_n = abs(input_n-output_n).reshape(-1,self.conf.step_num)
                err_n = err_n.reshape(self.conf.batch_num,-1)
                for batch in range(self.conf.batch_num):
                   temp = np.dot( (err_n[batch] - mu ).reshape(1,-1)  , sigma.T)
                   s = np.dot(temp,(err_n[batch] - mu ))
                   normal_score.append(s[0])
                   
            abnormal_score = []
            for count in range(len(self.conf.ta_list)//self.conf.batch_num):
                abnormal_sub = np.array(self.conf.ta_list[count*self.conf.batch_num:(count+1)*self.conf.batch_num]) 
                (input_a, output_a) = sess.run([input_, output_], {p_input: abnormal_sub, p_is_training: False})
                a_in.append(input_a)
                a_out.append(output_a)
                err_a = abs(input_a-output_a).reshape(-1,self.conf.step_num)
                err_a = err_a.reshape(self.conf.batch_num,-1)
                for batch in range(self.conf.batch_num):
                   temp = np.dot( (err_a[batch] - mu ).reshape(1,-1)  , sigma.T)
                   s = np.dot(temp,(err_a[batch] - mu ))
                   abnormal_score.append(s[0])
                     
        
            print('Predict result :')
        
            pd.Series(normal_score).plot(label="normal_score",figsize=(18,5))
            pd.Series(abnormal_score).plot(label="abnormal_score")
            bar = threshold*np.ones(len(normal_score)+len(abnormal_score))
            pd.Series(bar).plot(label="threshold")
            
            
            #targets
            tp = np.array(abnormal_score)[np.array(abnormal_score)>threshold].size
            fp = len(abnormal_score)-tp
            fn = np.array(normal_score)[np.array(normal_score)>threshold].size
            tn = len(normal_score)- fn
            P = tp/(tp+fp)
            R = tp/(tp+fn)
            fbeta= (1+beta*beta)*P*R/(beta*beta*P+R)
            print("tp: %.3f,fp: %.3f,tn: %.3f,fn: %.3f,\nP: %.3f,R: %.3f"%(tp,fp,tn,fn,P,R))
            print("Fbeta: %.3f"%fbeta)
        
            