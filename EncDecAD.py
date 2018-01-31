# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:17:47 2018

@author: Bin
"""
import tensorflow as tf

class EncDecAD(object):

    def __init__(self, hidden_num, inputs, is_training, optimizer=None, reverse=True, decode_without_input=False):

        self.batch_num = inputs[0].get_shape().as_list()[0]
        self.elem_num = inputs[0].get_shape().as_list()[1]
        
        self._enc_cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
        self._dec_cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)


        with tf.variable_scope('encoder',reuse = tf.AUTO_REUSE):
            (self.z_codes, self.enc_state) = tf.contrib.rnn.static_rnn(self._enc_cell, inputs, dtype=tf.float32)

        with tf.variable_scope('decoder',reuse =tf.AUTO_REUSE) as vs:
         
            dec_weight_ = tf.Variable(tf.truncated_normal([hidden_num,self.elem_num], dtype=tf.float32))
 
            dec_bias_ = tf.Variable(tf.constant(0.1,shape=[self.elem_num],dtype=tf.float32))

            if decode_without_input:
                dec_inputs = [tf.zeros(tf.shape(inputs[0]),dtype=tf.float32) for _ in range(len(inputs))]
                (dec_outputs, dec_state) = tf.contrib.rnn.static_rnn(self._dec_cell, dec_inputs, initial_state=self.enc_state, dtype=tf.float32)

                if reverse:
                    dec_outputs = dec_outputs[::-1]
                    
                dec_output_ = tf.transpose(tf.stack(dec_outputs), [1, 0,2])
                dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0),[self.batch_num, 1, 1])
                self.output_ =tf.add( tf.matmul(dec_output_, dec_weight_) + dec_bias_,name="output_")
                

            else:
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
                
        with tf.variable_scope('training',reuse = tf.AUTO_REUSE):
            self.input_ = tf.transpose(tf.stack(inputs), [1, 0, 2],name="input_")
            self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_))
            
            if is_training:
                self.train = tf.train.AdamOptimizer().minimize(self.loss)
                      