# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:08:45 2018

@author: Bin
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, 'C:/Users/Bin/Desktop/Thesis/code')
from Conf_EncDecAD_KDD99 import Conf_EncDecAD_KDD99
from EncDecAD import EncDecAD
from Parameter_helper import Parameter_Helper
from EncDecAD_Test import EncDecAD_Test

# Configuration 


conf = Conf_EncDecAD_KDD99()

batch_num = conf.batch_num
hidden_num = conf.hidden_num
step_num = conf.step_num
elem_num = conf.elem_num

iteration = conf.iteration
modelpath_root = conf.modelpath_root
modelpath = conf.modelpath
decode_without_input = conf.decode_without_input


#************#
# Training
#************#

p_input = tf.placeholder(tf.float32, shape=(batch_num, step_num, elem_num),name = "p_input")
p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]

p_is_training = tf.placeholder(tf.bool,name= "is_training_")


ae = EncDecAD(hidden_num, p_inputs, p_is_training , decode_without_input=False)

with tf.Session() as sess:
    saver = tf.train.Saver()
    graph = tf.get_default_graph()
    
    sess.run(tf.global_variables_initializer())
    loss = []
    for i in range(iteration):
        data =[]
        for temp in range(batch_num):
            ind = np.random.randint(0,len(conf.sn_list)-1)
            sub = conf.sn_list[ind]
            data.append(sub)
        data = np.array(data)
        (loss_val, _) = sess.run([ae.loss, ae.train], {p_input: data,p_is_training : True})
        loss.append(loss_val)
        print('iter %d:' % (i + 1), loss_val)
    pd.Series(loss).plot(title="Loss")
    
    # mu & sigma & threshold
    input_= tf.transpose(tf.stack(p_inputs), [1, 0, 2])    
    output_ = graph.get_tensor_by_name("decoder/output_:0")
    para = Parameter_Helper(conf)
    mu, sigma = para.mu_and_sigma(sess,input_, output_,p_input, p_is_training)
    threshold = para.get_threshold(mu,sigma,sess,input_, output_,p_input, p_is_training)
    
    test = EncDecAD_Test(conf)
    test.test_encdecad(sess,input_,output_,p_input,p_is_training,mu,sigma,threshold,beta = 0.5)
    
    
    
    
    c_mu = tf.constant(mu,dtype=tf.float32,name = "mu")
    c_sigma = tf.constant(sigma,dtype=tf.float32,name = "sigma")
    c_threshold = tf.constant(threshold,dtype=tf.float32,name = "threshold")
    print("Saving model to disk...")
    save_path = saver.save(sess, conf.modelpath_p)
    print("Model saved accompany with parameters and threshold in file: %s" % save_path)
    
    