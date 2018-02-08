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
import matplotlib.pyplot as plt
import time, threading
import sys
from sklearn.metrics import confusion_matrix
sys.path.insert(0, 'C:/Users/Bin/Desktop/Thesis/code')
from EncDecAD_Pred import EncDecAD_Pred
from Conf_Prediction_KDD99 import Conf_Prediction_KDD99

batch_num =20
step_num = 20

consumer = KafkaConsumer('kdd99stream',
                         group_id='test-consumer-group',    # defined in consumer.properties file
                         bootstrap_servers=['localhost:9092'],
                         auto_offset_reset = "earliest")
consumer.poll()
#go to end of the stream
consumer.seek_to_end()
dataframe = pd.DataFrame()
lock = threading.Lock()


def block_generator2queue(q,stop_event):
   
    while not stop_event.is_set():
        block = []
        
        for message in consumer:
            if stop_event.is_set():
                break
            row = message.value.decode("utf-8") 
#            list_of_str = row.replace(",null,null,null,null,null,null,null,null","").split(",")
            list_of_str = row.strip(",null,null,null,null,null,null,null,null").split(",")
            list_of_num = [float(n) for n in list_of_str]
            block.append(list_of_num)
            if len(block)==batch_num*step_num:
                df = pd.DataFrame(np.array(block))
                q.put(df)
#                print("Wrote a block to queue.\n")
                block.clear()
                
def read_block_from_queue(q,stop_event):
    global dataframe
    
    while not stop_event.is_set():
        if q.empty() == False:
            b = q.get()
#            print("Read a block from queue.\n")
            if dataframe.size == 0:
                dataframe = b
            else:
                pd.concat((dataframe,b),axis=0)
                
        else :
            time.sleep(0.5)
            
def prediction(stop_event):
    global dataframe
    pred = EncDecAD_Pred()
    conf = Conf_Prediction_KDD99()
    
    #.......................................#
    #  reload model
    sess = tf.Session()
        
    inputs = []
    predictions = []
    anomaly_scores = []
    
    saver = tf.train.import_meta_graph(conf.modelmeta_p) # load trained gragh, but without the trained parameters
    saver.restore(sess,tf.train.latest_checkpoint(conf.modelpath_root))
    graph = tf.get_default_graph()
    
    p_input = graph.get_tensor_by_name("p_input:0")
    p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, conf.step_num, 1)] 
    p_is_training = tf.placeholder(tf.bool)

    input_= tf.transpose(tf.stack(p_inputs), [1, 0, 2])    
    output_ = graph.get_tensor_by_name("decoder/output_:0")
    
    tensor_mu = graph.get_tensor_by_name("mu:0")
    tensor_sigma = graph.get_tensor_by_name("sigma:0")
    tensor_threshold = graph.get_tensor_by_name("threshold:0")
    
    mu = sess.run(tensor_mu)
    sigma = sess.run(tensor_sigma)
    threshold = sess.run(tensor_threshold)

    print("LSTMs-Autoencoder Model reloaded.")
            
    #................................................#
    
    while not stop_event.is_set():
        if dataframe.size == 0:
            sec = 5
            print("Currently not enough data for prediction, wait for %d seconds."%sec)
            time.sleep(sec)
        else:
            lock.acquire()
            try:
                print("Making prediction...")
                #.......................................#
#                pred.prediction(dataframe.iloc[:,:-1],dataframe.iloc[:,-1])
                
                dataset = dataframe.iloc[:,:-1]
                label = dataframe.iloc[:,-1]
                pred.predict(dataset,label,sess,input_,output_,p_input,p_is_training,mu,sigma,threshold)
#                
#                for count in range(dataset.shape[0]//conf.batch_num//conf.step_num):
#                    data = np.array(dataset[count*conf.batch_num*conf.step_num:
#                                    (count+1)*conf.batch_num*conf.step_num])
#                    data = data.reshape((conf.batch_num,conf.step_num,-1)) #**********#
#                    (input_n, output_n) = sess.run([input_, output_], {p_input: data, p_is_training: False})
#                    inputs.append(input_n)
#                    predictions.append(output_n)
#                    err_n = abs(input_n-output_n).reshape(-1,conf.step_num)
#                    err_n = err_n.reshape(conf.batch_num,-1)
#                    
#                    for batch in range(conf.batch_num):
#                       temp = np.dot( (err_n[batch] - mu ).reshape(1,-1)  , sigma.T)
#                       s = np.dot(temp,(err_n[batch] - mu ))
#                       anomaly_scores.append(s[0])
#                # each anomaly_score represent for the anomalous likelyhood of a window (length == batch_num)
#                # so here replicate each score 20 times, to approximate the anomalous likelyhood for each data point
#                tmp = []
#                for i in range(conf.step_num):
#                    for _ in range(conf.batch_num):
#                        tmp.append(anomaly_scores[i])
#                anomaly_scores = tmp
#                
#                pred = np.zeros(len(anomaly_scores))
#                pred[np.array(anomaly_scores) > threshold] = 1
#                evaluation(pred,label,threshold,anomaly_scores)

                #.......................................#
                print("Finish prediction.")
            finally:
                dataframe = pd.DataFrame()
                lock.release()
def evaluation(pred,label,threshold,anomaly_scores,beta=0.5):                
    print('Predict result :')
    fig, ax = plt.subplots()
    ax.set_ylim(min(min(anomaly_scores),threshold)*0.8,max(max(anomaly_scores),threshold)*1.2)
    anomaly_scores = pd.Series(anomaly_scores)
    plt.scatter(anomaly_scores.index,anomaly_scores,color="r",label="Anomaly score")
    bar = threshold*np.ones(anomaly_scores.size)
    pd.Series(bar).plot(label="Threshold")
    plt.legend(loc=2)
    plt.show()
    plt.close(fig)
    tn, fp, fn, tp = confusion_matrix(list(label), list(pred),labels=[1,0]).ravel() # 0 is positive, 1 is negative
    print("Label sum, Pred sum:\n",sum(label),sum(pred))
    P = tp/(tp+fp)
    R = tp/(tp+fn)
    fbeta= (1+beta*beta)*P*R/(beta*beta*P+R)
    print("tp: %.d,fp: %.d,tn: %.d,fn: %.d,\nP: %.3f,R: %.3f"%(tp,fp,tn,fn,P,R))
    print("Fbeta: %.3f"%fbeta)
            
def main():
    q = queue.Queue()
    stop_event = threading.Event()
    
    write = threading.Thread(target=block_generator2queue, name='WriteThread',args=(q,stop_event,))
    read = threading.Thread(target=read_block_from_queue, name='ReadThread',args=(q,stop_event,))
    predict = threading.Thread(target=prediction, name='prediction',args=(stop_event,))
   
    try:
        write.start()
        read.start()
        predict.start()
        
        while 1:
            time.sleep(.1)
    except (KeyboardInterrupt,SystemExit):
        stop_event.set()
        print("Threads closed.")
        
if __name__=="__main__":
    main()

    

    
