# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:12:59 2018

@author: Bin
"""

from Parameter_helper import Parameter_Helper

from Conf_EncDecAD_KDD99 import Conf_EncDecAD_KDD99

data_root = "C:/Users/Bin/Documents/Datasets/KDD99/6_subsets_win/"
conf = Conf_EncDecAD_KDD99(data_root)


para = Parameter_Helper(conf)

mu, sigma = para.mu_and_sigma()