#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:54:57 2017

@author: christy
for comparing flocking
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import time
import scipy.stats

def firstindex(list,target):
    nl=len(list)
    for i in range(nl):
        if list[i]>target:
#            return i
            if i>0:
                return (i-1)
            else:
                return 0
    return (nl-1)



def Wd(mu_1,grid_1,mu_2,grid_2,Nint,p=2):
    CDF_1=np.cumsum(mu_1)
    CDF_2=np.cumsum(mu_2)
    n_1=len(mu_1)
    n_2=len(mu_2)
    u_vec=np.linspace(0,1,Nint)
    du=1.0/Nint
    
    W=0
    for i in range(Nint):
        i1=firstindex(CDF_1,u_vec[i])
        i2=firstindex(CDF_2,u_vec[i])
        dW=du*pow(math.fabs(grid_1[i1]-grid_2[i2]),p)
        #print(dW)
        W+=dW
    W=pow(W,1.0/p)
    return W
    
if __name__ == '__main__':
#    mu_Pontryagin=np.load('./Data/flocking/mu_Pontryagin_num_t_40.npy')
#    mu_weak=np.load('./Data/flocking/mu_weak_num_t_40.npy')
#    mu_true=np.load('./Data/flocking/true_solution_num_t_40.npy')
    
    mu_Pontryagin=np.load('./Data/flocking/mu_Pontryagin_num_t_60.npy')
    mu_weak=np.load('./Data/flocking/mu_weak_num_t_60.npy')
    mu_true=np.load('./Data/flocking/true_solution_num_t_60.npy')

    mu_Pontryagin_end=mu_Pontryagin[len(mu_Pontryagin)-1]
    mu_weak_end=mu_weak[len(mu_weak)-1]
    mu_true_end=mu_true[len(mu_true)-1]

    num_x=len(mu_Pontryagin[0])
    x_min=-3
    x_max=3
    x_grid=np.linspace(x_min,x_max,num_x)
    num_t=len(mu_Pontryagin)
    
    
    
    d1=Wd(mu_Pontryagin_end,x_grid,mu_weak_end,x_grid,1000)
    d2=Wd(mu_Pontryagin_end,x_grid,mu_true_end,x_grid,1000)
    d3=Wd(mu_weak_end,x_grid,mu_true_end,x_grid,1000)
    
    print(d1,d2,d3)
    
    num_bins=int(num_x/30)
    num_x_hist=int(num_x/num_bins)
    delta_x_hist=np.abs(x_max-x_min)/num_x_hist
    x_grid_hist=np.linspace(x_min,x_max,num_x_hist)
    mu_weak_hist=np.zeros((num_t,num_x_hist))
    mu_Pontryagin_hist=np.zeros((num_t,num_x_hist))
    mu_true_hist=np.zeros((num_t,num_x_hist))
    
    for t in range(num_t):
        for i in range(num_x/num_bins):
            mu_weak_hist[t,i]=np.sum(mu_weak[t,num_bins*i:num_bins*(i+1)])
            mu_Pontryagin_hist[t,i]=np.sum(mu_Pontryagin[t,num_bins*i:num_bins*(i+1)])
            mu_true_hist[t,i]=np.sum(mu_true[t,num_bins*i:num_bins*(i+1)])
    for t in range(num_t):       
        d1=Wd(mu_Pontryagin_hist[t],x_grid_hist,mu_weak_hist[t],x_grid_hist,1000)
        d2=Wd(mu_Pontryagin_hist[t],x_grid_hist,mu_true_hist[t],x_grid_hist,1000)
        d3=Wd(mu_weak_hist[t],x_grid_hist,mu_true_hist[t],x_grid,1000)
        print(d1,d2,d3)
