# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:22:47 2018

@author: Bin
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import queue
from kafka import KafkaConsumer
import time, threading
import sys
import math
import time
import matplotlib.pyplot as plt
sys.path.insert(0, 'C:/Users/Bin/Desktop/Thesis/code')
from EncDecAD_Pred import EncDecAD_Pred
from Conf_Prediction_KDD99 import Conf_Prediction_KDD99
from LocalPreprocessing import LocalPreprocessing
from EncDecAD_ReTrain import EncDecAD_ReTrain

conf = Conf_Prediction_KDD99()
batch_num =conf.batch_num
step_num = conf.step_num
MIN_TEST_BLOCK_NUM = conf.min_test_block_num
MIN_RETRAIN_BLOCK_NUM = conf.min_retrain_block_num
class_label_file = conf.class_label_path
#class_label_file = "C:/Users/Bin/Documents/Datasets/KDD99/classes.txt"
class_list = pd.DataFrame()
kafka_topic = 'kdd99stream'
g_id='test-consumer-group'
servers = ['localhost:9092']
offset = "earliest"

consumer = KafkaConsumer(kafka_topic,
                         group_id=g_id,    # defined in consumer.properties file
                         bootstrap_servers=servers,
                         auto_offset_reset = offset)
consumer.poll()
#go to end of the stream
consumer.seek_to_end()
#consumer.seek_to_beginning()
dataframe = pd.DataFrame()
lock = threading.BoundedSemaphore(value=1)

index = pd.Series()
results_list = []
retrain_apply_indices = []
retrain_index_list = []
threshold_list = []
false_alarm_list = []
anomaly_recall_list = []

# use for store relation between pred and lables
with open(class_label_file) as file:
    line = file.readline()
    class_labels = pd.Series(line.split(","),name="label")
    class_labels = class_labels[class_labels!="normal"].reset_index(drop=True)
class_pred_relation = pd.DataFrame(np.zeros(class_labels.size*2).reshape(-1,2),columns=['False alarm','True alarm'])# two columns for a_as_n  and a_as_a
class_pred_relation = pd.concat((class_labels,class_pred_relation),axis=1)
class_pred_relation.label = class_pred_relation.label.apply(str)

def block_generator2queue(q,stop_event):
    
    while not stop_event.is_set():
#        print("Thread: block_generator2queue\n\n")
        block = []
        try:
            for message in consumer:
                if stop_event.is_set():
                    break
                row = message.value.decode("utf-8") 
                row_tmp = row.split(",")
                row_tmp[-1] = row_tmp[-1].strip("\".")
                block.append(row_tmp)
                if len(block)==batch_num*step_num:
                    df = pd.DataFrame(np.array(block))
                    q.put(df)
                    block.clear()
        finally:
            consumer.close()
                
                
def read_block_from_queue(q,stop_event):
    global dataframe
    
    while not stop_event.is_set():
#        print("Thread: read_block_from_queue\n\n")
        if q.empty() == False:
            b = q.get()
#            if b.shape[1] <50:   #hardcode         
            if dataframe.size == 0:
                dataframe = b
               
            else:    
                df_tmp = pd.concat((dataframe,b),axis=0).reset_index(drop=True)                  
                if df_tmp.shape[1] != b.shape[1] : 
                    print("Warn: Strange dataframe shape, dropped.") 
                else:
                    dataframe = df_tmp
        else :
            time.sleep(0.5)
            
def prediction(stop_event):
    global dataframe
    global class_pred_relation
    global index
    global results_list
    global retrain_apply_indices
    global retrain_index_list 
    global class_list
    global threshold_list
    print("Thread: prediction\n\n")
    pred = EncDecAD_Pred(conf)
    local_preprocessing = LocalPreprocessing(conf.column_name_file ,conf.step_num)
    #  reload model
    sess = tf.Session()
    input_,output_,p_input,p_is_training,loss_,train_,mu,sigma,threshold= pred.reloadModel(sess)
    threshold_list.append([0,threshold])
    p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]
    print("LSTMs-Autoencoder Model reloaded.")
    buffer = [] # for collecting hard examples used for retraining model
    buffer_info = [0,0] # [normal,anomaly]
    while not stop_event.is_set():   
#        with lock:
            lock.acquire()
            if dataframe.index.size < batch_num*step_num*MIN_TEST_BLOCK_NUM:
                sec = 5
                print("Currently not enough data for prediction, ",dataframe.index.size,"/",batch_num*step_num*MIN_TEST_BLOCK_NUM)
                lock.release()
                time.sleep(sec)
                
            else:
                try:
                    print("Local preprocessing...")
                    #After preprocessing, the second to last col is the string class label
                    # and last col is the 0/1 grundtruth (1 stand for anomaly)
                    lpdf = dataframe
                    index = dataframe.iloc[:,0]
                    index = pd.to_numeric(index, errors='coerce')
                    dataframe_preprocessed = local_preprocessing.run(lpdf, for_training = False)    
                    
                    print("Making prediction...")
                    dataframe_preprocessed.reset_index(drop=True,inplace=True)
                    dataset = dataframe_preprocessed.iloc[:,:-2]
                    label = dataframe_preprocessed.iloc[:,-1]
                    class_list = dataframe_preprocessed.iloc[:,-2]
                    # window.size == step_num
                    print("Making prediciton on: ",str(index[0])+"-"+str(index[index.size-1])+'\n')
                    hard_example_window_index, results= pred.predict(dataset,index,label,class_list,sess,input_,output_,p_input,p_is_training,mu,sigma,threshold,buffer_info,false_alarm_list,anomaly_recall_list)
                    
                    # results : [alarm_accuracy,false_alarm,alarm_recall,pred]
#                    false_alarm_list.append([str(index[0])+"-"+str(index[index.size-1]),results[1]])
#                    anomaly_recall_list.append([str(index[0])+"-"+str(index[index.size-1]),results[2]])
                    
                    
                    # store pred & label relation
                    predictions = pd.Series(results[3])
                    for p in range(class_list.size):
                        if class_list[p] !='normal':
                            if predictions[p] ==1: # true alarm
                                 class_pred_relation.loc[class_pred_relation.label ==  class_list[p],'True alarm'] += 1 
                            else: # false alarm
                                 class_pred_relation.loc[class_pred_relation.label ==  class_list[p],'False alarm'] += 1 

                    results[3] = index
                     #[index,alarm_accuracy,false_alarm,alarm_recall,pred]
                    result_df = pd.concat((results[3],pd.Series(results[0]*np.ones(results[3].size)),
                                           pd.Series(results[1]*np.ones(results[3].size)),
                                           pd.Series(results[2]*np.ones(results[3].size))),axis=1)
                    
                    results_list.append(result_df)
#                    result_q.put(result_df)
                    # got hard examples' index from prediction, then using this index to find the UNpreprocessed 
                    #hard examples from the original dataframe 
                    buffer.append(lpdf.loc[hard_example_window_index])
                    sub_lpdf = lpdf.loc[hard_example_window_index]
                    if sub_lpdf.size !=0:
                        buffer_info[0] += sub_lpdf[sub_lpdf.iloc[:,-1]=="normal"].index.size
                        buffer_info[1] += sub_lpdf[sub_lpdf.iloc[:,-1]!="normal"].index.size
#                    print("A df with %d rows is added to Buffer."%lpdf.index.size)
                    buffer_data_len = sum([df_.shape[0] for df_ in buffer])
                    
                    if buffer_data_len >= MIN_RETRAIN_BLOCK_NUM*batch_num:
                            print("Apply for retraining...")
                            if lpdf.loc[hard_example_window_index].size != 0:
                                apply_index = buffer[-1].iloc[-1,0]
                                if int(apply_index) not in retrain_apply_indices:
                                    retrain_apply_indices.append(int(apply_index))
                            data_for_retrain = pd.concat(buffer,axis=0)
                            data_for_retrain.reset_index(drop=True,inplace=True)
                            #retrain dataset shape: (batch_num*step_num*MIN_RETRAIN_BLOCK_NUM,elem_num)
                            data_for_retrain = data_for_retrain.iloc[:data_for_retrain.index.size-data_for_retrain.index.size%batch_num,:]#buffer[0].shape[1]]#.....................
                            sn,vn1,vn2,tn,va,ta,class_labels = local_preprocessing.run(data_for_retrain, for_training = True)
                            
                            if min([x.index.size for x in [sn,vn1,vn2,va]])<batch_num*step_num:
                                
                                print("Not enough normal or anomaly data for retraining, still waiting for more data.")
#                                print("sn(%d), vn1(%d), vn2(%d), va(%d) batches."%(sn.index.size//step_num//batch_num,vn1.index.size//step_num//batch_num,vn2.index.size//step_num//batch_num,va.index.size//step_num//batch_num))
                                print("sn(%d), vn1(%d), vn2(%d), va(%d)."%(sn.index.size,vn1.index.size,vn2.index.size,va.index.size))
                                print("data_for_retrain size: ",data_for_retrain.index.size)
                                print("Retrain Buffer: %d/%d.\n"%(buffer_data_len,MIN_RETRAIN_BLOCK_NUM*batch_num))
                                dataframe = pd.DataFrame()                                
                                continue
                            
                            print("Re-Training Model...")
                            print("sn(%d), vn1(%d), vn2(%d), va(%d) batches."%(sn.index.size//step_num//batch_num,vn1.index.size//step_num//batch_num,vn2.index.size//step_num//batch_num,va.index.size//step_num//batch_num))
                            index_of_data_for_retrain = [i.iloc[:,0] for i in [sn,vn1,vn2,tn,va,ta]]                        
                            retrain_index_list += index_of_data_for_retrain
                            # check out the largest index of retrain data, as the new threshold index after retrain
                            max_index = max(retrain_apply_indices)
                            
                            
                            sn = sn.drop(sn.columns[[0]],axis=1)
                            vn1 = vn1.drop(vn1.columns[[0]],axis=1)
                            vn2 = vn2.drop(vn2.columns[[0]],axis=1)
                            tn = tn.drop(tn.columns[[0]],axis=1)
                            va = va.drop(va.columns[[0]],axis=1)
                            ta = ta.drop(ta.columns[[0]],axis=1)
                            retrain = EncDecAD_ReTrain(conf,sn,vn1,vn2,tn,va,ta)
                            
                            threshold_list.append([max_index,threshold]) ## for plotting
                            mu_new,sigma_new,threshold_new,loss = retrain.continue_training(sess,loss_, train_,p_input,p_inputs,p_is_training,input_,output_)
                            
                            if math.isnan(threshold_new ) == False:
                                mu,sigma,threshold = mu_new,sigma_new,threshold_new
                            buffer.clear()
                            buffer_info = [0,0]
                            threshold_list.append([max_index,threshold])
                            retrain_plotting(class_labels,loss)
                    else: 
                        
                        print("Retrain Buffer: %d/%d.\n"%(buffer_data_len,MIN_RETRAIN_BLOCK_NUM*batch_num))
                        print("Finish prediction.Waiting for next batches of data.")
                    
                finally:
                    dataframe = pd.DataFrame()
                    lock.release()

def retrain_plotting(class_labels,loss):
    global threshold_list
    
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(13,13))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Retrain report")
    
    #ax1: threshold changes
    thresholds = pd.DataFrame(threshold_list)
    ax1.set_title("Threshold changing according to model update")
    ax1.plot(thresholds.iloc[:,0],thresholds.iloc[:,1])
    ax1.plot(thresholds.iloc[-1,0],thresholds.iloc[-1,1],'X',c='r')
    if thresholds.index.size<25:
        r = 'horizontal'
    else:
        r = 'vertical'
    ax1.set_xticklabels(thresholds.iloc[:,0], rotation=r)
    ax1.set_xticks(thresholds.iloc[:,0])
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Threshold")
    
    #ax2: retrain data distribution
    ax2.set_title("Retrain dataset distribution")
    ax2.set_xlabel("Subsets")
    ax2.set_ylabel("Count")
    
#    label_counts = [class_labels.count(l) for l in  pd.Series(class_labels).unique()]
#    ax2.bar(range(len(label_counts)),label_counts)
#    ax2.set_xticklabels(list(pd.Series(class_labels).unique()), rotation='vertical')
   
    count = []
    labels = []
    class_labels = pd.Series(class_labels)
    for class_label in class_labels.unique():
        count.append(class_labels[class_labels==class_label].size)
        labels.append(class_label)
    ax2.bar(range(len(labels)),count,width=0.2)
    if len(count)<10:
        r = 'horizontal'
    else:
        r = 'vertical'
    ax2.set_xticklabels(labels, rotation=r)
    ax2.set_xticks(range(len(labels)))
    
    ax2.set_ylim(max(count)*1.3)
    rects = ax2.patches
    for rect in rects:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        ax2.annotate(y_value, (x_value, y_value), xytext=(0, 5), textcoords="offset points", 
            ha='center', va='bottom')           
        
        
    #ax3: retrain error
    ax3.plot(loss)
    ax3.set_title("Retrain loss")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("SSE")
    
    save_path = conf.plot_savepath
    t = str(int(time.time()))
    plt.savefig(save_path+"Retrain"+t+".png")
    plt.show()
    plt.close()
    
def drawing():             
    global class_pred_relation 
    global index
    global results_list
    global retrain_index_list
    global retrain_apply_indices
    global threshold_list
    save_path = conf.plot_savepath
    
    #threshols changes
    t = str(int(time.time()))
    
    thresholds = pd.DataFrame(threshold_list)
    thresholds.set_index(thresholds.iloc[:,0]).iloc[:,1].plot()
    plt.xticks(thresholds.iloc[:,0],thresholds.iloc[:,0])
    plt.title("Threshold changing according to model update")
    plt.xlabel("Index")
    plt.ylabel("Threshold")
    plt.savefig(save_path+ "Threshold"+t+".png")
    plt.show()
    plt.close()

    # relationship
    class_pred_relation.iloc[:,1:].plot.bar(figsize=(13,6))
    plt.xticks(class_pred_relation.index, class_pred_relation.label, rotation='vertical')
    plt.title("Prediction statistic according to class label")
    plt.xlabel("Anomalous classes")
    plt.ylabel("Count")
    
    
    class_pred_relation.to_csv(save_path+"Prediction_relation"+t+".csv",header=None,index=None)
    plt.savefig(save_path+"Prediction"+t+".png")
    plt.show()
    plt.close()
    

    result = pd.concat(results_list,axis=0).reset_index(drop=True)
    if len(retrain_index_list) !=0:
        retrain_index = pd.concat(retrain_index_list,axis=0).reset_index(drop=True)
    else:
        retrain_index = pd.Series([])

    result = result.set_index(result.iloc[:,0]).iloc[:,1:]
    print("result shape",result.shape)
    result.columns = ['Alarm accuracy','False alarm','Alarm recall']
    
    # anomaly recall
    fig,ax = plt.subplots(1,1)
    fig.set_size_inches(12,6)
#    ax.scatter(retrain_index,np.ones(retrain_index.size)*(-0.2),label="Retrain data",c="y")
#    ind = []
#    for i in range(retrain_index.size-1):
#        if retrain_index[i+1] ==retrain_index[i] + 1:
#            ind.append(i)
#        else :
#            ax.fill_between(retrain_index[ind],y1=np.zeros(len(ind)),y2=np.ones(len(ind)))
#            ind.clear()
    
#    ax.scatter(result.index,result.iloc[:,0],label="Alarm accuracy",c="b")
    ax.scatter(result.index,result.iloc[:,2],label="Anomaly recall",c='g',s=0.1)
    plt.legend()
    plt.title("Anomaly recall")
    plt.xlabel("Index")
    
    t = str(int(time.time()))
    result.iloc[:,1:].to_csv(save_path+"Prediction_performance"+t+".csv",header=None,index=None)

    plt.savefig(save_path+"Recall"+t+".png")
    plt.show()
    plt.close()
    
    #False alarm
    fig,ax = plt.subplots(1,1)
    fig.set_size_inches(12,6)
    ax.scatter(result.index,result.iloc[:,1],label="False alarm",c='r',s=0.1)
    # retrain positions
    lines = [-0.2,1.1]*len(retrain_apply_indices)
    for i in range(len(retrain_apply_indices)):
        ys = [lines[i*2],lines[1+i*2]]
        xs = [retrain_apply_indices[i],retrain_apply_indices[i]]
        ax.plot(xs,ys,c="grey")
    plt.legend()
    plt.title("#False Alarm")
    plt.ylabel("Count")
    plt.xlabel("Index")
    
    plt.savefig(save_path+"FalseAlarm"+t+".png")
    plt.show()
    plt.close()
    
    count = 0
    for x in results_list:
        count += x.index.size
    print("Made prediction on",count,"examples.")
    
def main():
    q = queue.Queue()
#    result_q = queue.Queue()
    stop_event = threading.Event()
    
    write = threading.Thread(target=block_generator2queue, name='WriteThread',args=(q,stop_event,))
    read = threading.Thread(target=read_block_from_queue, name='ReadThread',args=(q,stop_event,))
    predict = threading.Thread(target=prediction, name='Prediction',args=(stop_event,))
#    draw = threading.Thread(target=drawing, name='Plotting',args=(stop_event,))
    
    try:
        
        write.start()
        read.start()
        predict.start()
#        draw.start()
        
        while 1:
            time.sleep(.1)
    except (KeyboardInterrupt,SystemExit): 
        drawing()
        stop_event.set()
        print("Threads closed.")
    
#    result = []
#    while q.empty() == False:
#        result.append(result_q.get())
#    return result
    
        
if __name__=="__main__":
    main()


    

    
