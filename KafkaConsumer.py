# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:22:47 2018

@author: Bin
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from EncDecAD import EncDecAD
modelpath = "C:/Users/Bin/Desktop/Thesis/tmp/LSTMAutoencoder_kdd99_v1.ckpt.meta"
ckpt_dir = "C:/Users/Bin/Desktop/Thesis/tmp/LSTMAutoencoder_kdd99_v1.ckpt"
paras = pd.Series(pd.read_csv(ckpt_dir + "parameters.csv",header=None)[0])

batch_num  = paras.iloc[0]
hidden_num = paras.iloc[1]
step_num = paras.iloc[2]
elem_num = paras.iloc[3]
threshold  = paras.iloc[4]
mu = 
sigma = 
batch_num = 20
hidden_num = 100
step_num = 20
elem_num = sn.shape[1]
threshold = 0.6
#
#p_input = tf.placeholder(tf.float32, shape=(batch_num, step_num, elem_num))
#p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]
#
#ae = EncDecAD(hidden_num, p_inputs, decode_without_input=False)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(modelpath)
    saver.restore(sess, ckpt_dir)
    print("Model restored.") 
    print('Initialized')

        
    graph = tf.get_default_graph()
    init = tf.global_variables_initializer()
    
    batch_num = graph.get_tensor_by_name("v_batch_num:0")
    hidden_num = graph.get_tensor_by_name("v_hidden_num:0")
    step_num = graph.get_tensor_by_name("v_step_num:0")
    elem_num = graph.get_tensor_by_name("v_elem_num:0")
    #threshold = graph.get_tensor_by_name("v_threshold:0")
#    mu = graph.get_tensor_by_name("v_mu:0")
#    sigma = graph.get_tensor_by_name("v_aigma:0")
    
    sess.run(init)
    batch_num,hidden_num,step_num,elem_num = sess.run([batch_num,hidden_num,step_num,elem_num])
   
    
    
    
    normal_score = []
    n_in = []
    n_out = []
    a_in = []
    a_out = []
    
    for count in range(len(tn_list)//batch_num):
        normal_sub = np.array(tn_list[count*batch_num:(count+1)*batch_num]) 
        (input_n, output_n) = sess.run([ae.input_, ae.output_], {p_input: normal_sub})
        n_in.append(input_n)
        n_out.append(output_n)
        err_n = abs(input_n-output_n).reshape(-1,step_num)
        err_n = err_n.reshape(batch_num,-1)
        for batch in range(batch_num):
           temp = np.dot( (err_n[batch] - mu ).reshape(1,-1)  , sigma.T)
           s = np.dot(temp,(err_n[batch] - mu ))
           normal_score.append(s[0])
           
    abnormal_score = []
    for count in range(len(ta_list)//batch_num):
        abnormal_sub = np.array(ta_list[count*batch_num:(count+1)*batch_num]) 
        (input_a, output_a) = sess.run([ae.input_, ae.output_], {p_input: abnormal_sub})
        a_in.append(input_a)
        a_out.append(output_a)
        err_a = abs(input_a-output_a).reshape(-1,step_num)
        err_a = err_a.reshape(batch_num,-1)
        for batch in range(batch_num):
           temp = np.dot( (err_a[batch] - mu ).reshape(1,-1)  , sigma.T)
           s = np.dot(temp,(err_a[batch] - mu ))
           abnormal_score.append(s[0])
             

    print('Predict result :')

    pd.Series(normal_score).plot(label="normal_score",figsize=(18,5))
    pd.Series(abnormal_score).plot(label="abnormal_score")
    bar = threshold*np.ones(len(normal_score)+len(abnormal_score))
    pd.Series(bar).plot(label="threshold")
    
def evaluation(abnormal_score,normal_score,threshold):
    beta = 0.5
    tp = np.array(abnormal_score)[np.array(abnormal_score)>threshold].size
    fp = len(abnormal_score)-tp
    fn = np.array(normal_score)[np.array(normal_score)>threshold].size
    tn = len(normal_score)- fn
    P = tp/(tp+fp)
    R = tp/(tp+fn)
    fbeta= (1+beta*beta)*P*R/(beta*beta*P+R)
    print(fbeta,tp,fp,tn,fn,P,R)