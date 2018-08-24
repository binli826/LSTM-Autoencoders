# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 14:09:13 2018

@author: Bin
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from conf_init import Configuration
from encdecad import EncDecAD
from parameterHelper import Parameter_Helper

class Initialization_Train(object):
    
    def __init__(self, dataset,dataPath, modelSavePath, training_data_source='file'):
        start_time = time.time()
        conf = Configuration(dataset, dataPath, modelSavePath, training_data_source=training_data_source)
        

        batch_num = conf.batch_num
        hidden_num = conf.hidden_num
        step_num = conf.step_num
        elem_num = conf.elem_num
        
        iteration = conf.iteration
        modelpath_root = conf.modelpath_root
        modelpath = conf.modelpath_p
        decode_without_input = conf.decode_without_input
        
        patience = 20
        patience_cnt = 0
        min_delta = 0.0001
        
        
        #************#
        # Training
        #************#
        
        p_input = tf.placeholder(tf.float32, shape=(batch_num, step_num, elem_num),name = "p_input")
        p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]
        
        p_is_training = tf.placeholder(tf.bool,name= "is_training_")
        
        ae = EncDecAD(hidden_num, p_inputs, p_is_training , decode_without_input=False)
        
        graph = tf.get_default_graph()
        gvars = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = [graph.get_operation_by_name(v.op.name + "/Assign") for v in gvars]
        init_values = [assign_op.inputs[1] for assign_op in assign_ops]    
            
        
        print("Training start.")
        with tf.Session() as sess:
            saver = tf.train.Saver()
            
            
            sess.run(tf.global_variables_initializer())
            input_= tf.transpose(tf.stack(p_inputs), [1, 0, 2])    
            output_ = graph.get_tensor_by_name("decoder/output_:0")

            loss = []
            val_loss = []
            sn_list_length = len(conf.sn_list)
            tn_list_length = len(conf.tn_list)
            
            for i in range(iteration):
                #training set
                snlist = conf.sn_list[:]
                tmp_loss = 0
                for t in range(sn_list_length//batch_num):
                    data =[]
                    for _ in range(batch_num):
                        data.append(snlist.pop())
                    data = np.array(data)
                    (loss_val, _) = sess.run([ae.loss, ae.train], {p_input: data,p_is_training : True})
                    tmp_loss += loss_val
                l = tmp_loss/(sn_list_length//batch_num)
                loss.append(l)
                
                #validation set
                tnlist = conf.tn_list[:]
                tmp_loss_ = 0
                for t in range(tn_list_length//batch_num):
                    testdata = []
                    for _ in range(batch_num):
                        testdata.append(tnlist.pop())
                    testdata = np.array(testdata)
                    (loss_val,ein,aus) = sess.run([ae.loss,input_,output_], {p_input: testdata,p_is_training :False})
                    tmp_loss_ += loss_val
                tl = tmp_loss_/(tn_list_length//batch_num)
                val_loss.append(tl)
                print('Epoch %d: Loss:%.3f, Val_loss:%.3f' %(i, l,tl))
                
                if i == 5:
                    break
                #Early stopping
                if i > 0 and  val_loss[i] < np.array(val_loss[:i]).min():
                    #save_path = saver.save(sess, conf.modelpath_p)
                    gvars_state = sess.run(gvars)
                    
                if i > 0 and val_loss[i-1] - val_loss[i] > min_delta:
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                
                if i>0 and patience_cnt > patience:
                    print("Early stopping at epoch %d\n"%i)
                    feed_dict = {init_value: val for init_value, val in zip(init_values, gvars_state)}
                    sess.run(assign_ops, feed_dict=feed_dict)
                    #saver.restore(sess,tf.train.latest_checkpoint(modelpath_root))
                    #graph = tf.get_default_graph()
                    break
        
            plt.plot(loss,label="Train")
            plt.plot(val_loss,label="val_loss")
            plt.legend()
            plt.show()
            
            
            # mu & sigma & threshold

            para = Parameter_Helper(conf)
            mu, sigma = para.mu_and_sigma(sess,input_, output_,p_input, p_is_training)
            threshold = para.get_threshold(mu,sigma,sess,input_, output_,p_input, p_is_training)
            
#            test = EncDecAD_Test(conf)
#            test.test_encdecad(sess,input_,output_,p_input,p_is_training,mu,sigma,threshold,beta = 0.5)
            
            c_mu = tf.constant(mu,dtype=tf.float32,name = "mu")
            c_sigma = tf.constant(sigma,dtype=tf.float32,name = "sigma")
            c_threshold = tf.constant(threshold,dtype=tf.float32,name = "threshold")
            print("Saving model to disk...")
            save_path = saver.save(sess, conf.modelpath_p)
            print("Model saved accompany with parameters and threshold in file: %s" % save_path)
            
            print("--- Initialization time: %s seconds ---" % (time.time() - start_time))
            
            f = open(conf.log_path,'a')
            f.write("Early stopping at epoch %d\n"%i)
#            f.write("mu:\n",mu,"sigma:\n",sigma,"threshold:\n",threshold)
            f.write("Model saved accompany with parameters and threshold in file: %s" % save_path)
            f.write("--- Initialization time: %s seconds ---" % (time.time() - start_time))
            f.close()