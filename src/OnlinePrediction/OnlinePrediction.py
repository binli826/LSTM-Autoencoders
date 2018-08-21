# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 15:50:07 2018

@author: Bin
"""
import argparse

import pandas as pd
import numpy as np
import tensorflow as tf
import time
from sklearn import metrics

from conf_online import Conf
from ProcessingHelper import processingHelper


def parseArguments():
    parser = argparse.ArgumentParser()
    # Positional mandatory arguments
    parser.add_argument("datasetname", help="power/smtp/http/forest", type=str)
    parser.add_argument("dataPath", help="input data path", type=str)
    parser.add_argument("modelPath", help="folder of trained model", type=str)
    
    # Parse arguments
    args = parser.parse_args()
    
    return args

if __name__  == "__main__":

    args = parseArguments()
    datasetname = args.__dict__['datasetname']
    datapath = args.__dict__['dataPath']
    modelpath_root= args.__dict__['modelPath']
    
    # get dataset parameters
    conf = Conf(datasetname)
    batch_num = conf.batch_num 
    hidden_num = conf.hidden_num
    step_num = conf.step_num
    elem_num = conf.elem_num
    init_wins = conf.training_set_size
    
    # load dataset and divede into batches and windows
    names = [str(x) for x in range(elem_num)] +["label"]
    forest = pd.read_csv(datapath,names=names,skiprows=step_num*init_wins)
    
    batches = forest.shape[0]//step_num//batch_num
    
    test_set = forest.iloc[:batches*batch_num*step_num,:-1]
    labels =forest.iloc[:batches*batch_num*step_num,-1]
    ts = test_set.as_matrix().reshape(batches,batch_num,step_num,elem_num)
    test_set_list = [ts[a] for a in range(batches)]
    
    wins = batches * batch_num
    # figure out anomaly windows
    buffer = [labels[i*step_num:(i+1)*step_num] for i in range(0,labels.size//step_num)]
    anomaly_index = []
    count = 0
    
    for buf in buffer:
        if "anomaly" in buf.tolist():
            anomaly_index.append(count)
        else:
            pass
        count +=1
    
    expert = ["normal"]*wins
    for x in anomaly_index:
        expert[x] = "anomaly"
        
        
    # load model
    
    modelmeta_p = modelpath_root + "_"+str(batch_num)+"_"+str(hidden_num)+"_"+str(step_num)+"_para.ckpt.meta"

    sess = tf.Session()
    saver = tf.train.import_meta_graph(modelmeta_p) # load trained gragh, but without the trained parameters
    saver.restore(sess,tf.train.latest_checkpoint(modelpath_root))
    graph = tf.get_default_graph()
    
    p_input = graph.get_tensor_by_name("p_input:0")
    p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)] 
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
    
    
    # online phase
    
    count = 0
    n_buf = []
    a_buf = []
    
    y = []
    output=[]
    err_nbuf = []
    err_abuf = []
    all_scores = []
    
    start_time = time.time()
    helper = processingHelper
    
    
    for l in labels:
        if l == "normal":
            y +=[0]
        else: 
            y +=[1]
    for ids in range(len(test_set_list)):
        
            data = test_set_list[ids]
            if count % 100 == 0:
                print(count,"batches processed.")
            prediction = []
            df = helper.local_preprocessing(data)
            (input_n, output_n) = sess.run([input_, output_], {p_input: df, p_is_training: False})
            err = abs(input_n-output_n).reshape(-1,elem_num)
            scores = helper.scoring(err,mu,sigma) 
            all_scores.append(scores)
            output += [ss.max() for ss in np.array(scores).reshape(batch_num,step_num)]
            pred = [scores[b*step_num:(b+1)*step_num] for b in range(batch_num)] 
            label = [expert[count*batch_num+b] for b in range(batch_num)]
            e = err
            
            for i in range(pd.DataFrame(pred).shape[0]):#loop batch_num
                index = i
                value=pd.DataFrame(pred).iloc[i,:]
                
                if value[value>threshold].size>=conf.HardCriterion: 
                    if label[index] == "anomaly":
                        a_buf += [df[index,x,:] for x in range(step_num)]
                        err_abuf = np.concatenate((err_abuf , e[index*step_num:(index+1)*step_num]),axis=0) if len(err_abuf) != 0 else e[index*step_num:(index+1)*step_num]
                    else:
                        err_nbuf =np.concatenate((err_nbuf , e[index*step_num:(index+1)*step_num]),axis=0) if len(err_nbuf) != 0 else e[index*step_num:(index+1)*step_num]
                        n_buf += [df[index,x,:] for x in range(step_num)]
                else:               
                    if label[index] == "anomaly":             
                        a_buf += [df[index,x,:] for x in range(step_num)]
                        err_abuf = np.concatenate((err_abuf , e[index*step_num:(index+1)*step_num]),axis=0) if len(err_abuf) != 0 else e[index*step_num:(index+1)*step_num]
                    else:
                        err_nbuf = np.concatenate((err_nbuf , e[index*step_num:(index+1)*step_num]),axis=0) if len(err_nbuf) != 0 else e[index*step_num:(index+1)*step_num]
            count +=1
            
            #Check update
            if len(n_buf)>=batch_num*step_num*conf.buffersize and len(a_buf) !=0:
                while (len(a_buf) < batch_num*step_num):
                    a_buf += a_buf
                
                B = len(n_buf) //(batch_num*step_num)
                n_buf = n_buf[:batch_num*step_num*B]
                A = len(a_buf)//(batch_num*step_num)
                a_buf = a_buf[:batch_num*step_num*A]
                
                print("retrain at %d batch"%count)
                loss_list_all=[]
    
                datalist = np.array(n_buf[:batch_num*step_num*(B-1)]).reshape(-1,batch_num,step_num,elem_num)
                validation_list_n = np.array(n_buf[batch_num*step_num*(B-1):]).reshape(-1,batch_num,step_num,elem_num)
                validation_list_a = np.array(a_buf).reshape(-1,batch_num,step_num,elem_num)
                
                patience = 10
                min_delta = 0.005
                lastLoss = np.float('inf')
                patience_cnt = 0
                
                for i in range(300):
                    loss_list=[]
                    for data in datalist:
                        (loss, _) = sess.run([loss_, train_], {p_input: data,p_is_training : True})
                        loss_list.append(loss)
                    loss_list_all.append( np.array(loss_list).mean()) 
                    
                    if i > 0 and lastLoss - loss >min_delta:
                        patience_cnt =0
                    else:
                        patience_cnt += 1
                    lastLoss = loss
                    if patience_cnt > patience:
                        print("Early stopping...")
                        break
                    
                    
                err_nbuf_tmp = np.array(err_nbuf).reshape(-1,elem_num)
                mu,sigma = helper.get_musigma(err_nbuf_tmp,mu,sigma)
    
                print("Parameters updated!")
                pd.Series(loss_list_all).plot(title="Loss")
                n_buf = []
                a_buf = []
                err_buf = []
                err_nbuf = []
                err_abuf = []
            
            
    fpr, tpr, thresholds = metrics.roc_curve(expert, output, pos_label="anomaly")
    auc = metrics.auc(fpr, tpr)
    helper.plot_roc(fpr,tpr,auc)
    print("--- Used time: %s seconds ---" % (time.time() - start_time))