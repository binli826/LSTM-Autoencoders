# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:05:29 2018

@author: Bin
"""

import sys
import tensorflow as tf
import numpy as np
import pandas as pd
sys.path.insert(0, 'C:/Users/Bin/Desktop/Thesis/code')
from functools import reduce
from Conf_EncDecAD_KDD99 import Conf_EncDecAD_KDD99



# Configuration 

data_root = "C:/Users/Bin/Documents/Datasets/KDD99/6_subsets_win/"
conf = Conf_EncDecAD_KDD99(data_root)
#[sn_list, va_list, vn1_list, vn2_list, tn_list, ta_list] = conf.data_list

p_input = conf.p_input
p_inputs = conf.p_inputs



batch_num = conf.batch_num
hidden_num = conf.hidden_num
step_num = conf.step_num
elem_num = conf.elem_num

iteration = conf.iteration
modelpath_root = conf.modelpath_root
modelpath = conf.modelpath
decode_without_input = conf.decode_without_input


 


with tf.Session() as sess:

    saver = tf.train.Saver()

    _enc_cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
    _dec_cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
    inputs = conf.p_inputs
    reverse = True
    decode_without_input = False
    is_training = True
    
    with tf.variable_scope('encoder',reuse = tf.AUTO_REUSE):
        (z_codes, enc_state) = tf.contrib.rnn.static_rnn(_enc_cell, inputs, dtype=tf.float32)
    with tf.variable_scope('decoder',reuse =tf.AUTO_REUSE) as vs:
        dec_weight_ = tf.Variable(tf.truncated_normal([hidden_num,elem_num], dtype=tf.float32),name="dec_weight_")
        dec_bias_ = tf.Variable(tf.constant(0.1,shape=[elem_num],dtype=tf.float32),name="dec_bias_")
        dec_state = enc_state
        dec_input_ = tf.zeros(tf.shape(inputs[0]),dtype=tf.float32)
        dec_outputs = []
        for step in range(len(inputs)):
            if step > 0:
                vs.reuse_variables()
            (dec_input_, dec_state) =_dec_cell(dec_input_, dec_state)
            dec_input_ = tf.matmul(dec_input_, dec_weight_) + dec_bias_
            dec_outputs.append(dec_input_)
        if reverse:
            dec_outputs = dec_outputs[::-1]
        output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])
    
        input_= tf.transpose(tf.stack(inputs), [1, 0, 2])
        output_ = tf.transpose(output_, [0,1, 2])
        loss_ = tf.reduce_mean(tf.square(input_ - output_),name="loss_")
        
        input_ = tf.Variable( tf.transpose(tf.stack(inputs), [1, 0, 2]),dtype=tf.float32,name="input_")
        output_ = tf.Variable( tf.transpose(tf.stack(dec_outputs), [1, 0, 2]),dtype=tf.float32,name="output_")
    
        
#    with tf.variable_scope(tf.get_variable_scope(),reuse=False):    
    train_ = tf.train.AdamOptimizer().minimize(loss_)
    sess.run(tf.global_variables_initializer())

    loss = []
    for i in range(iteration):
        data =[]
        for temp in range(batch_num):
            ind = np.random.randint(0,len(conf.sn_list)-1)
            sub = conf.sn_list[ind]
            data.append(sub)
        data = np.array(data)
        
        (loss_val, _) = sess.run([loss_, train_], {p_input: data})
        loss.append(loss_val)
        print('iter %d:' % (i + 1), loss_val)
    pd.Series(loss).plot(title="Loss")

    save_path = saver.save(sess, modelpath)
    print("Model saved in file: %s" % save_path) 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
