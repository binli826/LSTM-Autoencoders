# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:15:23 2018

@author: Bin
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time

class EncDecAD_Pred(object):
    
    def __init__(self,conf):
        self.conf = conf
        
    def reloadModel(self,sess,):
        
        saver = tf.train.import_meta_graph(self.conf.modelmeta_p) # load trained gragh, but without the trained parameters
        saver.restore(sess,tf.train.latest_checkpoint(self.conf.modelpath_root))
        graph = tf.get_default_graph()
        
        p_input = graph.get_tensor_by_name("p_input:0")
        p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, self.conf.step_num, 1)] 
#        p_is_training = tf.placeholder(tf.bool)
        p_is_training = graph.get_tensor_by_name("is_training_:0")
        
        input_= tf.transpose(tf.stack(p_inputs), [1, 0, 2])    
        output_ = graph.get_tensor_by_name("decoder/output_:0")
        
        tensor_mu = graph.get_tensor_by_name("mu:0")
        tensor_sigma = graph.get_tensor_by_name("sigma:0")
        tensor_threshold = graph.get_tensor_by_name("threshold:0")
        
        loss_ = graph.get_tensor_by_name("decoder/loss:0")
        train_ = graph.get_operation_by_name("cond/train_")
        
        mu = sess.run(tensor_mu)
        sigma = sess.run(tensor_sigma)
        threshold = sess.run(tensor_threshold)
        return input_,output_,p_input,p_is_training,loss_,train_,mu,sigma,threshold
        
    def predict(self,dataset,df_index_,label,sess,input_,output_,p_input,p_is_training,mu,sigma,threshold,beta=0.5):

            anomaly_scores = []
            
            elem_num = dataset.shape[1]
            for count in range(dataset.shape[0]//self.conf.batch_num//self.conf.step_num):
                inputs = []
                predictions = []
                anomaly_scores_sub = []
                data = np.array(dataset[count*self.conf.batch_num*self.conf.step_num:
                                (count+1)*self.conf.batch_num*self.conf.step_num])
                data = data.reshape((self.conf.batch_num,self.conf.step_num,elem_num)) #**********#
                (input_n, output_n) = sess.run([input_, output_], {p_input: data, p_is_training: False})
                inputs.append(input_n)
                predictions.append(output_n)
                err_n = abs(input_n-output_n).reshape(-1,self.conf.step_num)
                err_n = err_n.reshape(self.conf.batch_num,-1)
                
                for batch in range(self.conf.batch_num):
                   temp = np.dot( (err_n[batch] - mu ).reshape(1,-1)  , sigma.T)
                   s = np.dot(temp,(err_n[batch] - mu ))
                   anomaly_scores_sub.append(s[0])
            # each anomaly_score represent for the anomalous likelyhood of a window (length == batch_num)
            # so here replicate each score 20 times, to approximate the anomalous likelyhood for each data point
                tmp = []
                for i in range(self.conf.step_num):
                    for _ in range(self.conf.batch_num):
                        tmp.append(anomaly_scores_sub[i])
                anomaly_scores_sub = tmp
                
                anomaly_scores += anomaly_scores_sub
                
            
            pred = np.zeros(len(anomaly_scores))
            pred[np.array(anomaly_scores) > threshold] = 1
            print('Predict result :')
            fig, ax = plt.subplots(figsize=(18,5))
            ax.set_ylim(min(min(anomaly_scores),threshold)*0.8,max(max(anomaly_scores),threshold)*1.2)
            anomaly_scores = pd.Series(anomaly_scores)
            
            #data within the following boundary will be collected in the retrain buffer
            upper_bound = np.mean([anomaly_scores[anomaly_scores>threshold].median(),threshold]) 
            lower_bound = np.mean([anomaly_scores[anomaly_scores<=threshold].median(),threshold])

            plt.scatter(anomaly_scores.index,anomaly_scores,color="r",label="Anomaly score",s=2)
#            plt.scatter(anomaly_scores.index,anomaly_scores,color="r",label="Anomaly score",s=2)
            bar = threshold*np.ones(anomaly_scores.size)
            up = upper_bound*np.ones(anomaly_scores.size)
            low = lower_bound*np.ones(anomaly_scores.size)
            pd.Series(bar).plot(label="Threshold")
            pd.Series(up).plot(label="Upper bound",c="y")
            pd.Series(low).plot(label="Lower Bound",c="y")
            plt.legend(loc=2)
            plt.ylabel("Anomaly score")
#            plt.xlabel("Indices(Contrains values from index "+str(df_index_[0])+" to "+str(df_index_[-1]))
            plt.xlabel("Indices(Contrains values from index "+str(min(df_index_))+" to "+str(max(df_index_)))
            plt.title("Real-time prediction")
            
            plt.savefig("C:/Users/Bin/Desktop/Thesis/Plotting/2/Predictions/"+str(int(time.time()))+".png")
            plt.show()
            plt.close(fig)
            
            assert len(list(label))==len(list(pred)), "label(%d) and pred(%d) have different size."%(len(list(label)),len(list(pred)))
            tn, fp, fn, tp = confusion_matrix(list(label), list(pred),labels=[1,0]).ravel() # 0 is positive, 1 is negative
            print("Label sum, Pred sum:\n",sum(label),sum(pred))
            
            alarm_accuracy = tn/(fn+tn) if (fn+tn)!=0 else -0.1  
            false_alarm = fn#/(fn+tn) if (fn+tn)!=0 else -0.1
            alarm_recall = tn/(tn+fp) if (tn+fp)!=0 else 1
            results = [alarm_accuracy,false_alarm,alarm_recall,pred]
            print("alarm_accuracy : ",alarm_accuracy)
            print("false_alarm : ",false_alarm)
            print("alarm_recall : ",alarm_recall)
#            print("alarm_accuracy : %d\nfalse_alarm : %d\nalarm_recall : %.f\n"%(alarm_accuracy,false_alarm,alarm_recall))
            ''' 
            P = tp/(tp+fp)
            R = tp/(tp+fn)
            fbeta= (1+beta*beta)*P*R/(beta*beta*P+R)
            print("tp: %.d,fp: %.d,tn: %.d,fn: %.d,\nP: %.3f,R: %.3f"%(tp,fp,tn,fn,P,R))
            print("Fbeta: %.3f"%fbeta)
            '''
            # return hard examples for model retraining
#            a_s_o = [anomaly_scores[self.conf.batch_num*i] for i in range(anomaly_scores.shape[0]//self.conf.batch_num)]
            hard_exaple_window_index = anomaly_scores.between(lower_bound/5,upper_bound*5,inclusive=True)
            
            return hard_exaple_window_index,results
            
    