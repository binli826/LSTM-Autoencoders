# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 16:07:31 2018

@author: Bin
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import mahalanobis,euclidean


class processingHelper(object):
    
    def local_preprocessing(batchdata):
        # input batchdata with shape : [batch_num, step_num, elem_num]
        # minmax scaler on window level
        df = pd.DataFrame()
        
        for window in batchdata:
            
            scaler = MinMaxScaler()
            scaler.fit(window)
            new_win = scaler.transform(window)
            df = pd.concat((df, pd.DataFrame(new_win)),axis=0) if df.size!=0 else pd.DataFrame(new_win)
        return df.as_matrix().reshape(batchdata.shape)
    
    def scoring(err,mu,sigma):
        
        scores = []
        for e in err:
            scores.append(mahalanobis(e,mu,sigma))
    
        return scores
    
    def get_musigma(err_nbuf,mu,sigma):       
        
            err_vec_array = np.array(err_nbuf)
            # for multivariate  data, cov, for univariate data, var
            mu = np.mean(err_vec_array.ravel()) if err_nbuf.shape[1] == 1 else np.mean(err_vec_array,axis=0)
            sigma = np.var(err_vec_array.ravel()) if err_nbuf.shape[1] == 1 else np.cov(err_vec_array.T)

            return mu, sigma
        
    def get_threshold(normal_score, abnormal_score):
            upper = np.median(np.array(abnormal_score))
            lower = np.median(np.array(normal_score)) 
            scala = 20
            delta = (upper-lower) / scala
            candidate = lower
            threshold = 0
            result = 0
    
            def evaluate(threshold,normal_score,abnormal_score):
    
                beta = 0.5
                tp = np.array(abnormal_score)[np.array(abnormal_score)>threshold].size
                fp = len(abnormal_score)-tp
                fn = np.array(normal_score)[np.array(normal_score)>threshold].size
                tn = len(normal_score)- fn
    
                if tp == 0: return 0
    
                P = tp/(tp+fp)
                R = tp/(tp+fn)
                fbeta= (1+beta*beta)*P*R/(beta*beta*P+R)
                return fbeta 
    
            for _ in range(scala):
                r = evaluate(candidate,normal_score,abnormal_score)
                if r > result:
                    result = r 
                    threshold = candidate
                candidate += delta 
            return threshold
        
    def plot_roc(fpr,tpr,auc):
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' %auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()