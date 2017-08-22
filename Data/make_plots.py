#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:42:56 2017

@author: Andrea Angiuli, Christy Graves, Houzhi Li
"""

import numpy as np
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
#    rho_values=np.load('./ex 73/tree_example_73_rho_values.npy')
#    tree_Y_0_values_1=np.load('./ex 73/tree_example_73_one_level_changing_rho.npy')
#    tree_Y_0_values_2=np.load('./ex 73/tree_example_73_two_level_changing_rho.npy')
#    tree_Y_0_values_3=np.load('./ex 73/tree_example_73_three_level_changing_rho.npy')
#    grid_Y_0_values=np.load('./ex 73/grid_example_73_changing_rho.npy')
#    #grid_Y_0_values=np.load('grid_example_73_changing_rho_larger_x_domain.npy')
#    
#    num_keep=5
#    for index2 in range(num_keep):
#        for index in range(len(rho_values)):
#            tree_1=plt.scatter(rho_values[index],tree_Y_0_values_1[index][index2],color='blue')
#            tree_2=plt.scatter(rho_values[index],tree_Y_0_values_2[index][index2],color='red')
#            tree_3=plt.scatter(rho_values[index],tree_Y_0_values_3[index][index2],color='green')
#            grid=plt.scatter(rho_values[index],grid_Y_0_values[index][index2],color='black')
#    plt.xlabel('$\\rho$')
#    plt.ylabel('$Y_0$')
#    plt.legend([tree_1, tree_2, tree_3, grid], ['Tree, N=1', 'Tree, N=2', 'Tree, N=3', 'Grid, N=1'],bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
#    plt.savefig('tree_and_grid_example_73_changing_rho_legend.eps')
    
    
    
    
#    sigma_values=np.load('./ex 73/tree_example_73_sigma_values.npy')
#    tree_Y_0_values_one=np.load('./ex 73/tree_example_73_one_level_changing_sigma.npy')
#    num_keep=5
#    num_sigma=20
#    for index in range(num_sigma):
#        for index2 in range(num_keep):
#            tree_1=plt.scatter(sigma_values[index],tree_Y_0_values_one[index][index2],color='blue')
#    plt.xlabel('$\sigma$')
#    plt.ylabel('$Y_0$')
#    plt.legend([tree_1], ['Tree, N=1, $\\rho=2$'],bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
#    plt.savefig('tree_example_73_changing_sigma_legend.eps')


#    rho_values=np.load('./ex 73/tree_example_73_continuation_sigma_rho_values.npy')
#    tree_Y_0_values_1=np.load('./ex 73/tree_example_73_one_level_continuation_sigma_1.npy')
#    tree_Y_0_values_2=np.load('./ex 73/tree_example_73_one_level_continuation_sigma_2.npy')
#    tree_Y_0_values_3=np.load('./ex 73/tree_example_73_one_level_continuation_sigma_3.npy')
#
#    rho_values_2=np.load('./ex 73/tree_example_73_rho_values.npy')
#    tree_Y_0_values=np.load('./ex 73/tree_example_73_one_level_changing_rho.npy')
#
#    num_keep=5
#    for index2 in range(num_keep):
#        for index in range(9):
#            tree_4=plt.scatter(rho_values_2[index],tree_Y_0_values[index][index2],color='black')
#        for index in range(len(rho_values)):
#            tree_1=plt.scatter(rho_values[index],tree_Y_0_values_1[index][index2],color='blue')
#            tree_2=plt.scatter(rho_values[index],tree_Y_0_values_2[index][index2],color='red')
#            tree_3=plt.scatter(rho_values[index],tree_Y_0_values_3[index][index2],color='green')
#    plt.xlabel('$\\rho$')
#    plt.ylabel('$Y_0$')
#    plt.legend([tree_4, tree_1, tree_2, tree_3], ['Tree, N=1', 'Tree, N=1, $\Delta\sigma$=1', 'Tree, N=1, $\Delta\sigma$=0.5', 'Tree, N=1, $\Delta\sigma$=0.1'],bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
#    plt.savefig('tree_example_73_continuation_sigma_legend.eps')
    
#    rho_values=np.load('./ex 73/tree_example_73_rho_values.npy')
#    tree_Y_0_values=np.load('./ex 73/tree_example_73_one_level_changing_rho.npy')
#    rho_values_1=np.load('./ex 73/tree_example_73_rho_values_1.npy')
#    rho_values_2=np.load('./ex 73/tree_example_73_rho_values_01.npy')
#    rho_values_3=np.load('./ex 73/tree_example_73_rho_values_001.npy')
#    tree_Y_0_values_1=np.load('./ex 73/tree_example_73_one_level_continuation_rho_1.npy')
#    tree_Y_0_values_2=np.load('./ex 73/tree_example_73_one_level_continuation_rho_01.npy')
#    tree_Y_0_values_3=np.load('./ex 73/tree_example_73_one_level_continuation_rho_001.npy')
#
#    num_keep=5
#    for index2 in range(num_keep):
#        for index in range(9):
#            tree_4=plt.scatter(rho_values[index],tree_Y_0_values[index][index2],color='black')
#        for index in range(len(rho_values_1)):
#            tree_1=plt.scatter(rho_values_1[index],tree_Y_0_values_1[index][index2],color='blue')
#        for index in range(len(rho_values_2)):
#            tree_2=plt.scatter(rho_values_2[index],tree_Y_0_values_2[index][index2],color='red')
#        for index in range(len(rho_values_3)):
#            tree_3=plt.scatter(rho_values_3[index],tree_Y_0_values_3[index][index2],color='green')
#    plt.xlabel('$\\rho$')
#    plt.ylabel('$Y_0$')
#    plt.legend([tree_4, tree_1, tree_2, tree_3], ['Tree, N=1', 'Tree, N=1, $\Delta\\rho$=0.1', 'Tree, N=1, $\Delta\\rho$=0.01', 'Tree, N=1, $\Delta\\rho$=0.001'],bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
#    plt.savefig('tree_example_73_continuation_rho_legend.eps')
    
#    num_keep=5
#    for index2 in range(num_keep):
#        for index in range(len(rho_values)):
#             grid_72=plt.scatter(rho_values[index],thing[index][index2],color='black')
#    plt.xlabel('rho')
#    plt.ylabel('Y_0')
#    plt.savefig('adaptive_grid_example_72.eps')

#    plt.scatter(x_grid,mu[348]/delta_x)
#    plt.xlabel('x')
#    plt.ylabel('$\mu(x)$')
#    plt.title('Jetlag, PDE Approach, T/2')
#    plt.savefig('PDE_jetlag.eps')

#    plt.scatter(x_grid,-u[348])
#    plt.xlabel('x')
#    plt.ylabel('alpha(T/2,x)')
#    plt.title('Jetlag, PDE Approach, T/2')
#    plt.savefig('PDE_jetlag_feedback.eps')

#    plt.scatter(x_grid,mu[25]/delta_x)
#    plt.xlabel('x')
#    plt.ylabel('$\mu(x)$')
#    plt.title('Jetlag, Grid Algorithm, Pontryagin Approach, T/2')
#    plt.savefig('grid_jetlag_Pontryagin_larger_delta_t.eps')

    
#    plt.scatter(x_grid,-u[348])
#    plt.xlabel('x')
#    plt.ylabel('alpha(T/2,x)')
#    plt.title('Jetlag, Grid Algorithm, Pontryagin Approach, T/2')
#    plt.savefig('grid_jetlag_Pontryagin_feedback.eps')

#    plt.scatter(x_grid,mu[25]/delta_x)
#    plt.xlabel('x')
#    plt.ylabel('$\mu(x)$')
#    plt.title('Jetlag, Grid Algorithm, Weak Approach, T/2')
#    plt.savefig('grid_jetlag_Weak_larger_delta_t_truncation.eps')
    
    
#    plt.scatter(x_grid_hist,mu_Pontryagin_hist[129],color='blue')
#    plt.xlabel('x')
#    plt.ylabel('$\mu(x)$')
#    #plt.title('Flocking, Grid Algorithm, Pontryagin Approach, T/2')
#    #plt.savefig('flocking_Pontryagin_t130.eps')
#
#    plt.scatter(x_grid_hist,mu_weak_hist[129],color='red')
#    plt.xlabel('x')
#    plt.ylabel('$\mu(x)$')
#    #plt.title('Flocking, Grid Algorithm, Weak Approach, T/2')
#    #plt.savefig('flocking_weak_t130.eps')
#
#    plt.scatter(x_grid_hist,mu_true_hist[129],color='black')
#    plt.xlabel('x')
#    plt.ylabel('$\mu(x)$')
#    #plt.title('Flocking, True Solution, T/2')
#    #plt.savefig('flocking_true_t130.eps')
#    
#    plt.savefig('flocking_grid_and_true.eps')
    
   
#    num_t=20
#    path='/home/christy/Dropbox/CEMRACS_MFG/cluster_results/flocking_tree/'
#    X=np.load(path+'flocking_Pont_tree_X_t20.npy')
#    #data1=plt.hist(X[num_t-1],bins=30)
#    counts=data1[0]
#    counts=counts/sum(counts)
#    markers=data1[1]
#    centers=[(markers[i]+markers[i+1])/2.0 for i in range(len(markers)-1)]
#    plt.scatter(centers,counts,color='blue')
#    plt.xlabel('x')
#    plt.ylabel('$\mu(x)$')
#    #plt.title('Flocking, Tree Algorithm, Pontryagin Approach, T/2')
#    #plt.savefig('flocking_Pontryagin_tree_t20.eps')
#    
#    
#    num_t=20
#    path='/home/christy/Dropbox/CEMRACS_MFG/cluster_results/flocking_tree/'
#    X=np.load(path+'flocking_weak_tree_X_t20.npy')
#    #data2=plt.hist(X[num_t-1],bins=30)
#    counts=data2[0]
#    counts=counts/sum(counts)
#    markers=data2[1]
#    centers=[(markers[i]+markers[i+1])/2.0 for i in range(len(markers)-1)]
#    plt.scatter(centers,counts,color='red')
#    plt.xlabel('x')
#    plt.ylabel('$\mu(x)$')
#    #plt.title('Flocking, Tree Algorithm, Pontryagin Approach, T/2')
#    #plt.savefig('flocking_weak_tree_t20.eps')
#    
#
#    mu_true_hist=scipy.stats.norm(mean_mu, variance_mu).pdf(centers)
#    plt.scatter(centers,mu_true_hist,color='black')
#    plt.xlabel('x')
#    plt.ylabel('$\mu(x)$')
#    #plt.title('Flocking, True Solution, T/2')
#    #plt.savefig('flocking_true_10_bins.eps')
#    
#    plt.savefig('flocking_tree_and_true_t20_30_bins.eps')


#    #data=plt.hist(X[5])
#    counts=data[0]
#    counts=counts/sum(counts)
#    markers=data[1]
#    centers=[(markers[i]+markers[i+1])/2.0 for i in range(len(markers)-1)]
#    plt.scatter(centers,counts)
#    plt.xlabel('x')
#    plt.ylabel('$\mu(x)$')
#    plt.title('Flocking, Tree Algorithm, Weak Approach, T/2')
#    plt.savefig('flocking_weak_tree_10_bins.eps')

#    rho_values=np.load('./trader/trader_continuation_time_3_levels_c_x_values.npy')
#    grid_Y_0_values=np.load('./trader/trader_continuation_time_3_levels_changing_c_x.npy')
#    
#    num_keep=5
#    for index2 in range(num_keep):
#        for index in range(len(rho_values)):
#            grid=plt.scatter(rho_values[index],grid_Y_0_values[index][index2],color='blue')
#    plt.xlabel('c_x')
#    plt.ylabel('Y_0')
#    #plt.title('Continuation in time : sigma = 0.7, rho = 1.5, $c_x /in [3,12]$, h_{bar}=2, c_g=0.3')
#    plt.savefig('trader_continuation_time_3_levels_changing_c_x.eps')


#    thing1=np.load('./ex_1/ex_1_log_errors.npy')
#    thing2=np.load('./ex_1/ex_1_log_errors_2.npy')
#    log_errors=np.concatenate((thing1,thing2))
#    thing3=np.load('ex_1/ex_1_log_num_t.npy')
#    thing4=np.load('ex_1/ex_1_log_num_t_2.npy')
#    log_num_t=np.concatenate((thing3,thing4))
#    plt.scatter(log_num_t,log_errors)
#    plt.xlabel('log number of time steps')
#    plt.ylabel('log error in Y_0')
#    plt.savefig('ex_1_grid_3.eps')

    c_x_values=np.load('./trader/trader_continuation_time_3_levels_c_x_values.npy')
    all_Y_0_1=np.load('./trader/trader_continuation_time_1_levels_changing_c_x.npy')
    all_Y_0_2=np.load('./trader/trader_continuation_time_3_levels_changing_c_x.npy')
    
    for index2 in range(len(c_x_values)):
        for index in range(5):
            thing1=plt.scatter(c_x_values[index2],all_Y_0_1[index2][index],color='blue')
            thing2=plt.scatter(c_x_values[index2],all_Y_0_2[index2][index],color='green')
    plt.xlabel('$c_X$')
    plt.ylabel('$Y_0$')
    plt.legend([thing1, thing2], ['Grid Pontryagin N=1','Grid Pontryagin N=3'],bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)

    plt.savefig('trader_continuation_time_3_levels_changing_c_x.eps')