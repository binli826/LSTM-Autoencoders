# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:17:47 2018

@author: Bin
"""
import tensorflow as tf
import math
class EncDecAD(object):

    def __init__(self, hidden_num, inputs, is_training, optimizer=None, reverse=True, decode_without_input=False):

        self.batch_num = inputs[0].get_shape().as_list()[0]
        self.elem_num = inputs[0].get_shape().as_list()[1]
        
        self._enc_cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
        self._dec_cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
        
        self.is_training = is_training

        with tf.variable_scope('encoder',reuse = tf.AUTO_REUSE):
            (self.z_codes, self.enc_state) = tf.contrib.rnn.static_rnn(self._enc_cell, inputs, dtype=tf.float32)

        with tf.variable_scope('decoder',reuse =tf.AUTO_REUSE) as vs:
         
            dec_weight_ = tf.Variable(tf.truncated_normal([hidden_num,self.elem_num], dtype=tf.float32))
 
            dec_bias_ = tf.Variable(tf.constant(0.1,shape=[self.elem_num],dtype=tf.float32))

           
            dec_state = self.enc_state
            dec_input_ = tf.zeros(tf.shape(inputs[0]),dtype=tf.float32)
            dec_outputs = []
            
            for step in range(len(inputs)):
                if step > 0:
                    vs.reuse_variables()
                (dec_input_, dec_state) =self._dec_cell(dec_input_, dec_state)
                dec_input_ = tf.matmul(dec_input_, dec_weight_) + dec_bias_
                dec_outputs.append(dec_input_)

            if reverse:
                dec_outputs = dec_outputs[::-1]

            self.output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2],name="output_")
                
        
        self.input_ = tf.transpose(tf.stack(inputs), [1, 0, 2],name="input_")
        self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_),name="loss")
        
       
        def check_is_train(ph):
            def t_ (): 
                print("is training")
                return tf.train.AdamOptimizer().minimize(self.loss)
            def f_ (): 
                print("not training")
                return tf.train.AdamOptimizer(1/math.inf).minimize(self.loss)
            is_train = tf.cond(is_training, t_, f_)
            return is_train
        
    
        
        self.train = check_is_train(is_training)

#            if is_train:
#                self.train = tf.train.AdamOptimizer().minimize(self.loss)
#            else:
#                self.trian = self.train
                      