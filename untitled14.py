# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:03:12 2018

@author: Bin
"""


import tensorflow as tf


with tf.Session() as sess:

            
    saver = tf.train.import_meta_graph( "C:/Users/Bin/Desktop/Thesis/tmp/"+"LSTMAutoencoder_kdd99_v1.ckpt.meta")
    saver.restore( sess,"C:/Users/Bin/Desktop/Thesis/tmp/"+"LSTMAutoencoder_kdd99_v1.ckpt")
    graph = tf.get_default_graph()
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    threshold = graph.get_tensor_by_name("v_threshold:0")
    
    print(sess.run(threshold))
    