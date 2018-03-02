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
    
    
    def plotting(self,anomaly_scores,threshold,df_index_,pred,label,buffer_info,class_list,hard_example_window_index,upper_bound,lower_bound,false_alarm_list,anomaly_recall_list):
        
            print('Prediction report :')
            fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,1,figsize=(15,25))
            plt.subplots_adjust( hspace=0.7)
            plt.suptitle("Prediction on KDD99 dataset index between "+str(min(df_index_))+" to "+str(max(df_index_)),fontsize=30)
            ax1.set_title("Prediction")
            ax1.set_ylim(min(min(anomaly_scores),threshold)*0.8,max(max(anomaly_scores),threshold)*1.2)

            a_set_index = anomaly_scores.index * label.as_matrix()
            a_set_index = a_set_index[a_set_index!=0]
            a_set_value = anomaly_scores[a_set_index]
            ax1.scatter(a_set_index,a_set_value,color="red",label="Anomaly score(Grund truth=1)",s=2)
            
            n_set_index = anomaly_scores.index * (1-label.as_matrix())
            n_set_index = n_set_index[n_set_index != 0]
            n_set_value = anomaly_scores[n_set_index]
            ax1.scatter(n_set_index,n_set_value,color="black",label="Anomaly score(Grund truth=0)",s=2)
            
            bar = threshold*np.ones(anomaly_scores.size)
            up = upper_bound*np.ones(anomaly_scores.size)
            low = lower_bound*np.ones(anomaly_scores.size)
            ax1.plot(bar,label="Threshold")
            ax1.plot(up,label="Upper bound",c="y")
            ax1.plot(low,label="Lower Bound",c="y")
#            pd.Series(bar).plot(label="Threshold")
#            pd.Series(up).plot(label="Upper bound",c="y")
#            pd.Series(low).plot(label="Lower Bound",c="y")
            ax1.legend(loc=2)
            ax1.set_ylabel("Anomaly score")
            ax1.set_xlabel("Index (Contrains values from index "+str(min(df_index_))+" to "+str(max(df_index_))+")")
           
            
            ax2.set_title("Dataset used for prediction")
            count = []
            labels = []
            for class_label in class_list.unique():
                count.append(class_list[class_list == class_label].size)
                labels.append(class_label)
            if len(labels) <=10:
                count = count+list(np.zeros(10-len(count)))
                labels = labels+['' for _ in range(10-len(labels))]
                ax2.bar(range(10),count,tick_label=labels,color="blue")
            else:
                ax2.bar(range(len(labels)),count,tick_label=labels,color="blue")
            ax2.set_xlabel("Classes")
            ax2.set_ylabel("Count")
            ax2.set_ylim([0,max(count)*1.3])
            
            rects = ax2.patches
            for rect in rects[:len(class_list.unique())]:
                y_value = rect.get_height()
                x_value = rect.get_x() + rect.get_width() / 2

                ax2.annotate(y_value, (x_value, y_value), xytext=(0, 5), textcoords="offset points", 
                    ha='center', va='bottom')                      

            
            
            ax3.set_title("Hard examples' distribution")
            hard_list = class_list.loc[hard_example_window_index]
            count = []
            labels = []
            for class_label in hard_list.unique():
                count.append(hard_list[hard_list == class_label].size)
                labels.append(class_label)
            if len(labels) <=10:
                count = count+list(np.zeros(10-len(count)))
                labels = labels+['' for _ in range(10-len(labels))]
                ax3.bar(range(10),count,tick_label=labels,color="blue")
            else:
                ax3.bar(range(len(labels)),count,tick_label=labels,color="blue")
            ax3.set_xlabel("Classes")
            ax3.set_ylabel("Count")
            
            count.append(1.)
            ax3.set_ylim([0,max(count)*1.3])
            rects = ax3.patches
            for rect in rects[:len(hard_list.unique())]:
                y_value = rect.get_height()
                x_value = rect.get_x() + rect.get_width() / 2

                ax3.annotate(y_value, (x_value, y_value), xytext=(0, 5), textcoords="offset points", 
                    ha='center', va='bottom')
            
            
            
            ax4.set_title("Buffer data distribution")
            
            buffer_normal = buffer_info[0]
            buffer_anomaly = buffer_info[1]
            
            ax4.bar([0,1],[buffer_normal,buffer_anomaly],tick_label=["Normal","Anomalous"],width=0.4,align='center')
            ax4.set_xlabel("Label")
            ax4.set_ylabel("Count")
            
            ax4.set_ylim([0,max(1,max(buffer_normal,buffer_anomaly)*1.2)])
            
            rects = ax4.patches
            for rect in rects:
                y_value = rect.get_height()
                x_value = rect.get_x() + rect.get_width() / 2

                ax4.annotate(y_value, (x_value, y_value), xytext=(0, 5), textcoords="offset points", 
                    ha='center', va='bottom')
            
            
            ax5.set_title("False alarm")
            fa = pd.DataFrame(false_alarm_list)
            ax5.plot(range(fa.shape[0]),fa.iloc[:,1])
            ax5.plot(fa.shape[0]-1,fa.iloc[-1,1],"X",c='r')
            gap =( fa.shape[0] // 20)+1
            ax5.set_xticklabels([fa.iloc[i,0] for i in range(fa.shape[0]) if i%gap==0],rotation=45)
            ax5.set_xticks([i for i in range(fa.shape[0]) if i%gap==0])
#            ax5.set_xticklabels(fa.iloc[:,0], rotation=45)
#            ax5.set_xticks(range(fa.shape[0]))
            ax5.set_xlabel("Indices")
            ax5.set_ylabel("False alarm count")
            
            
            
            ax6.set_title("Anomaly recall")
            
            ar = pd.DataFrame(anomaly_recall_list)
            ax6.plot(range(ar.shape[0]),ar.iloc[:,1])
            ax6.plot(ar.shape[0]-1,ar.iloc[-1,1],"X",c='r')
            
            gap =( ar.shape[0] // 20)+1
            ax6.set_xticklabels([ar.iloc[i,0] for i in range(ar.shape[0]) if i%gap==0], rotation=45)
            ax6.set_xticks([i for i in range(ar.shape[0]) if i%gap==0])
#            ax6.set_xticklabels(ar.iloc[:,0], rotation=45)
#            ax6.set_xticks(range(ar.shape[0]))
            
            ax6.set_xlabel("Indices")
            ax6.set_ylabel("Anomaly recall")
            ax6.set_ylim([0,1.1])
            
                
            
            statistic = pd.concat((df_index_,anomaly_scores,pd.Series(pred),pd.Series(upper_bound),pd.Series(lower_bound),pd.Series(bar)),axis=1).reset_index(drop=True)
            t = str(int(time.time()))
            statistic.to_csv(self.conf.plot_savepath+"Predictions/"+t+".csv")
            plt.savefig(self.conf.plot_savepath+"Predictions/"+t+".png")
            plt.show()
            plt.close(fig)
            
            
    def predict(self,dataset,df_index_,label,class_list,sess,input_,output_,p_input,p_is_training,mu,sigma,threshold,buffer_info,false_alarm_list,anomaly_recall_list,beta=0.5):
            '''
                para: 
                    dataset: format[f1,...,fn]
            '''
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
            anomaly_scores = pd.Series(anomaly_scores)
            upper_bound = np.mean([anomaly_scores[anomaly_scores>threshold].median(),threshold])*5
            lower_bound = np.mean([anomaly_scores[anomaly_scores<=threshold].median(),threshold])/5
            
            hard_example_window_index = anomaly_scores.between(lower_bound,upper_bound,inclusive=True)
            assert df_index_.size!=0, "prediction dataset index size is 0"
            false_alarm_list.append([str(df_index_[0])+"-"+str(df_index_[df_index_.size-1]),false_alarm])
            anomaly_recall_list.append([str(df_index_[0])+"-"+str(df_index_[df_index_.size-1]),alarm_recall])
            
            self.plotting(anomaly_scores,threshold,df_index_,pred,label,buffer_info,class_list,hard_example_window_index,
                          upper_bound,lower_bound,false_alarm_list,anomaly_recall_list)
            
            return hard_example_window_index,results
            '''
                return format:
                    hard_example_window_index:
                        batch index of examples with anomaly score between lower bound and upper bound
                    results:
                       alarm_accuracy (float[0,1])
                       false_alarm (int, count of false alarms)
                       alarm_recall (float[0,1])
                       pred: array of (0/1)
            '''
    