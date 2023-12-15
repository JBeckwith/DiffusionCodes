#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:48:14 2023

This class relates to statistical tests etc.

@author: jbeckwith
"""
import numpy as np
import scipy as sp


class Statistics_Functions():
    def __init__(self):
        self = self
        return
    
    def rejectoutliers(self, data):
        """ rejectoutliers function
        # rejects outliers from data, does iqr method (i.e. anything below
        lower quartile (25 percent) or above upper quartile (75 percent)
        is rejected)
        # ================INPUTS============= 
        # data is data matrix
        # ================OUTPUT============= 
        # newdata, data matrix """
        iqr = sp.stats.iqr(data)
        q1, q2 = np.percentile(data, q=(25, 75))
        
        nd1 = data[data <= (1.5*iqr)+q2]
        newdata = nd1[nd1 >= q1-(1.5*iqr)]
        return newdata 
    
    def bincalculator(self, data):
        """ bincalculator function
        # reads in data and generates bins according to Freedman-Diaconis rule
        # ================INPUTS============= 
        # data is data to be histogrammed
        # ================OUTPUT============= 
        # bins """
        N = len(data)
        sigma = np.std(data)
    
        binwidth = np.multiply(np.multiply(np.power(N, np.divide(-1,3)), sigma), 3.5)
        bins = np.linspace(np.min(data), np.max(data), int((np.max(data) - np.min(data))/binwidth)+1)
        return bins
    