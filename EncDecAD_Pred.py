# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:15:23 2018

@author: Bin
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'C:/Users/Bin/Desktop/Thesis/code')
from Conf_Prediction_KDD99 import Conf_Prediction_KDD99
from sklearn.metrics import confusion_matrix


class EncDecAD_Pred(object):
    
    def __init__(self,):
        self.conf = Conf_Prediction_KDD99()
        
    def prediction(self,dataset,label, beta = 0.5):
        
        with tf.Session() as sess: 
            
            inputs = []
            predictions = []
            anomaly_scores = []
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

            print("LSTMs-Autoencoder Model imported.")
            
            for count in range(dataset.shape[0]//self.conf.batch_num//self.conf.step_num):
                data = np.array(dataset[count*self.conf.batch_num*self.conf.step_num:
                                (count+1)*self.conf.batch_num*self.conf.step_num])
                data = data.reshape((self.conf.batch_num,self.conf.step_num,-1)) #**********#
                (input_n, output_n) = sess.run([input_, output_], {p_input: data, p_is_training: False})
                inputs.append(input_n)
                predictions.append(output_n)
                err_n = abs(input_n-output_n).reshape(-1,self.conf.step_num)
                err_n = err_n.reshape(self.conf.batch_num,-1)
                
                for batch in range(self.conf.batch_num):
                   temp = np.dot( (err_n[batch] - mu ).reshape(1,-1)  , sigma.T)
                   s = np.dot(temp,(err_n[batch] - mu ))
                   anomaly_scores.append(s[0])
            # each anomaly_score represent for the anomalous likelyhood of a window (length == batch_num)
            # so here replicate each score 20 times, to approximate the anomalous likelyhood for each data point
            tmp = []
            for i in range(self.conf.step_num):
                for _ in range(self.conf.batch_num):
                    tmp.append(anomaly_scores[i])
            anomaly_scores = tmp
            
            pred = np.zeros(len(anomaly_scores))
            pred[np.array(anomaly_scores) > threshold] = 1
            print('Predict result :')
            fig, ax = plt.subplots()
            ax.set_ylim(min(min(anomaly_scores),threshold)*0.8,max(max(anomaly_scores),threshold)*1.2)
            anomaly_scores = pd.Series(anomaly_scores)
            plt.scatter(anomaly_scores[anomaly_scores==1].index,anomaly_scores[anomaly_scores==1],c='b')
            plt.scatter(anomaly_scores[anomaly_scores==0].index,anomaly_scores[anomaly_scores==0],c='r')
          
#            pd.Series(anomaly_scores).plot(label="Anomaly_score",figsize=(18,5))
#            pd.Series(abnormal_score).plot(label="abnormal_score")
            bar = threshold*np.ones(anomaly_scores.size)
            pd.Series(bar).plot(label="Threshold")
            plt.legend(loc=2)
            plt.show()
            plt.close(fig)
            
            
            tn, fp, fn, tp = confusion_matrix(list(label), list(pred),labels=[1,0]).ravel() # 0 is positive, 1 is negative
            print("Label sum, Pred sum:\n",sum(label),sum(pred))
#            #targets
#            tp = np.array(abnormal_score)[np.array(abnormal_score)>threshold].size
#            fp = len(abnormal_score)-tp
#            fn = np.array(normal_score)[np.array(normal_score)>threshold].size
#            tn = len(normal_score)- fn
            
            
            P = tp/(tp+fp)
            R = tp/(tp+fn)
            fbeta= (1+beta*beta)*P*R/(beta*beta*P+R)
            print("tp: %.d,fp: %.d,tn: %.d,fn: %.d,\nP: %.3f,R: %.3f"%(tp,fp,tn,fn,P,R))
            print("Fbeta: %.3f"%fbeta)
            