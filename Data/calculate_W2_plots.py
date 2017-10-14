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
from Wd_exact import *
    
if __name__ == '__main__':
    path='/home/christy/Dropbox/CEMRACS_MFG/cluster_results/trader_grid/'
    path2='/home/christy/Documents/MFG_git/large_data_trader/'
    num_trials=7
    value_num_t=np.linspace(10,130,num_trials)
    all_d1=np.zeros(num_trials)
    all_d2=np.zeros(num_trials)
    all_d3=np.zeros(num_trials)
    all_d4=np.zeros(num_trials)
    all_d5=np.zeros(num_trials)
    all_d6=np.zeros(num_trials)
    all_d7=np.zeros(num_trials)
    all_d8=np.zeros(num_trials)    
    all_d9=np.zeros(num_trials)
    d1=0
    d2=0
    d3=0
    d4=0
    d5=0
    d6=0
    for k in range(num_trials):
        num_t=value_num_t[k]
        num_t=int(num_t)
        mu_Pontryagin=np.load(path+'trader_mu_Pont_t'+str(num_t)+'.npy')
        mu_weak=np.load(path+'trader_mu_weak_t'+str(num_t)+'.npy')
        mu_weak_trunc=np.load(path+'trader_mu_weak_trunc_t'+str(num_t)+'.npy')
        mu_true=np.load(path2+'mu_true_t'+str(num_t)+'.npy')
#        mu_Pontryagin=np.load(path+'flocking_mu_Pont_t'+str(num_t)+'.npy')
#        mu_weak=np.load(path+'flocking_mu_weak_t'+str(num_t)+'_real.npy')
#        mu_true=np.load('mu_true_t'+str(num_t)+'.npy')

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
    
        #d1=Wd_exact_R(x_grid,mu_Pontryagin_end,mu_weak_end,2)
        #d2=Wd_exact_R(x_grid,mu_Pontryagin_end,mu_weak_trunc_end,2)
        #d3=Wd_exact_R(x_grid,mu_weak_end,mu_weak_trunc_end,2)
        d4=Wd_exact_R(x_grid,mu_Pontryagin_end,mu_true_end,2)
        d5=Wd_exact_R(x_grid,mu_weak_end,mu_true_end,2)
        d6=Wd_exact_R(x_grid,mu_weak_trunc_end,mu_true_end,2)
    
        print(d1,d2,d3,d4,d5,d6)
        all_d1[k]=d1
        all_d2[k]=d2
        all_d3[k]=d3
        all_d4[k]=d4
        all_d5[k]=d5
        all_d6[k]=d6

        Y_Pontryagin=np.load(path+'trader_Y_Pont_t'+str(num_t)+'.npy')
        Z_weak=np.load(path+'trader_Z_weak_t'+str(num_t)+'.npy')
        Z_weak_trunc=np.load(path+'trader_Z_weak_trunc_t'+str(num_t)+'.npy')
        
        true_Y=np.load(path2+'trader_Y_solution'+str(num_t)+'.npy')
        MSE_Pont=0
        MSE_weak=0
        MSE_weak_trunc=0
        num_x=len(mu_true[0])
        #for i in range(num_t):
        i=0
        
        square_t_weak=np.power(Z_weak[i]-0.7*true_Y[i],2)
        thing=np.dot(square_t_weak,mu_true[i])
        print(thing)
        
        MSE_Pont+=np.dot((Y_Pontryagin[i]-true_Y[i])**2,mu_true[i]) #/float(num_t)
        MSE_weak+=np.dot((Z_weak[i]/0.7-true_Y[i])**2,mu_true[i]) #/float(num_t)
        MSE_weak_trunc+=np.dot((Z_weak_trunc[i]/0.7-true_Y[i])**2,mu_true[i]) #/float(num_t)
        all_d7[k]=MSE_Pont
        all_d8[k]=MSE_weak
        all_d9[k]=MSE_weak_trunc

        print(MSE_Pont,MSE_weak,MSE_weak_trunc)
    

#    plot1=plt.scatter(value_num_t,all_d4,color='blue')
#    plot2=plt.scatter(value_num_t,all_d5,color='red')
#    plot3=plt.scatter(value_num_t,all_d6,color='green')
#    plt.xlabel('number of time steps')
#    plt.ylabel('W2 distance at time $T$')
#    plt.legend([plot1, plot2,plot3], ['Pontryagin', 'Weak','Weak Truncated'],bbox_to_anchor=(1, 0.65), loc=1, borderaxespad=0.)
#
#    plt.savefig('trader_changing_delta_t_legend.eps')
#    
#    
#    plot1=plt.scatter(value_num_t,all_d7,color='blue')
#    #plot2=plt.scatter(value_num_t,all_d8,color='red')
#    #plot3=plt.scatter(value_num_t,all_d9,color='green')
#    plt.xlabel('number of time steps')
#    plt.ylabel('Average MSE of Controls')
#    plt.legend([plot1], ['Pontryagin'],bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
#    plt.savefig('trader_changing_delta_t_MSE_legend_1.eps')
    
    plot1=plt.scatter(value_num_t,all_d7,color='blue')
    #plot2=plt.scatter(value_num_t,all_d8,color='red')
    plot3=plt.scatter(value_num_t,all_d9,color='green')
    plt.xlabel('number of time steps')
    plt.ylabel('Mean Square Error of Controls')
    plt.legend([plot1,plot3], ['Pontryagin', 'Weak Truncated'],bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
    plt.savefig('trader_changing_delta_t_MSE_legend_3.eps')
