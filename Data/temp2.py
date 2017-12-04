#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:35:33 2017

@author: christy
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import scipy.io
import scipy.stats
from Wd_exact import *

if __name__ == '__main__':
    path='/home/christy/Documents/CEMRACS/'
    mu_Pontryagin=np.load(path+'flocking_grid/flocking_mu_Pont_t10.npy')
    mu_weak=np.load(path+'flocking_grid//flocking_mu_weak_t10.npy')
    mu_true=np.load(path+'flocking_true/mu_true_t10.npy')

    mu_Pontryagin_end=mu_Pontryagin[len(mu_Pontryagin)-1]
    mu_weak_end=mu_weak[len(mu_weak)-1]
    mu_true_end=mu_true[len(mu_true)-1]
    num_x=len(mu_Pontryagin[0])     
    x_min=-3
    x_max=3
    x_grid=np.linspace(x_min,x_max,num_x)
    
    d1=Wd_exact_R(x_grid,mu_Pontryagin_end,mu_true_end,2)
    print(d1)
    d1=Wd_approx_R(x_grid,mu_Pontryagin_end,mu_true_end,2)
    print(d1)
    d2=Wd_exact_R(x_grid,mu_weak_end,mu_true_end,2)
    print(d2)
    d2=Wd_approx_R(x_grid,mu_weak_end,mu_true_end,2)
    print(d2)
    
    
    savemat(mu_Pontryagin.mat,)