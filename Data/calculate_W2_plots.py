#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:25:29 2017

@author: christy
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
    value_num_t=np.linspace(10,200,20)
    all_d1=np.zeros(20)
    all_d2=np.zeros(20)
    all_d3=np.zeros(20)
    all_d4=np.zeros(20)
    all_d5=np.zeros(20)
    all_d6=np.zeros(20)
    for k in range(20):
        num_t=value_num_t[k]
        num_t=int(num_t)
        mu_Pontryagin=np.load('./Data/from_cluster/trader_mu_Pont_t'+str(num_t)+'.npy')
        mu_weak=np.load('./Data/from_cluster/trader_mu_weak_t'+str(num_t)+'.npy')
        mu_weak_trunc=np.load('./Data/from_cluster/trader_mu_weak_trunc_t'+str(num_t)+'.npy')
        mu_true=np.load('./Data/from_cluster/trader_mu_true_t'+str(num_t)+'.npy')

        mu_Pontryagin_end=mu_Pontryagin[len(mu_Pontryagin)-1]
        mu_weak_end=mu_weak[len(mu_weak)-1]
        mu_weak_trunc_end=mu_weak_trunc[len(mu_weak_trunc)-1]
        mu_true_end=mu_true[len(mu_true)-1]

        num_x=len(mu_Pontryagin[0])
        x_min=-2
        x_max=4
#        x_min=-3
#        x_max=3
        x_grid=np.linspace(x_min,x_max,num_x)
    
        d1=Wd(mu_Pontryagin_end,x_grid,mu_weak_end,x_grid,1000)
        d2=Wd(mu_Pontryagin_end,x_grid,mu_weak_trunc_end,x_grid,1000)
        d3=Wd(mu_weak_end,x_grid,mu_weak_trunc_end,x_grid,1000)
        d4=Wd(mu_Pontryagin_end,x_grid,mu_true_end,x_grid,1000)
        d5=Wd(mu_weak_end,x_grid,mu_true_end,x_grid,1000)
        d6=Wd(mu_weak_trunc_end,x_grid,mu_true_end,x_grid,1000)
    
        print(d1,d2,d3,d4,d5,d6)
        all_d1[k]=d1
        all_d2[k]=d2
        all_d3[k]=d3
        all_d4[k]=d4
        all_d5[k]=d5
        all_d6[k]=d6
    
    T=1
    delta_t_s=(T-0.06)/(value_num_t-1)
    plt.scatter(delta_t_s,all_d1,color='blue')
    plt.scatter(delta_t_s,all_d2,color='red')
    plt.scatter(delta_t_s,all_d3,color='green')
    plt.xlabel('\Delta t')
    plt.ylabel('W2 distance')
    plt.savefig('trader_changing_delta_t.eps')