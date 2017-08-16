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
#    rho_values=np.load('tree_example_72_rho_values.npy')
#    tree_Y_0_values_1=np.load('tree_example_72_one_level_changing_rho.npy')
#    tree_Y_0_values_2=np.load('tree_example_72_two_level_changing_rho.npy')
#    grid_Y_0_values=np.load('grid_example_72_changing_rho_larger_x_domain.npy')
#    
#    num_keep=5
#    for index2 in range(num_keep):
#        for index in range(len(rho_values)):
#            tree_1=plt.scatter(rho_values[index],tree_Y_0_values_1[index][index2],color='blue')
#            tree_2=plt.scatter(rho_values[index],tree_Y_0_values_2[index][index2],color='red')
#            grid=plt.scatter(rho_values[index],grid_Y_0_values[index][index2],color='black')
#    plt.xlabel('rho')
#    plt.ylabel('Y_0')
#    plt.savefig('tree_and_grid_example_72_changing_rho_larger_x_domain.eps')
    
    
    
#    sigma_values=np.load('tree_example_73_E_sigma_values.npy')
#    tree_Y_0_values_one=np.load('tree_example_73_E_one_level_changing_sigma.npy')
#    num_keep=5
#    num_sigma=20
#    for index in range(num_sigma):
#        for index2 in range(num_keep):
#            tree_1=plt.scatter(sigma_values[index],tree_Y_0_values_one[index][index2],color='blue')
#            #grid=plt.scatter(sigma_values[index],grid_Y_0_values[index][index2],color='black')
#    plt.xlabel('sigma')
#    plt.ylabel('Y_0')
#    #plt.legend([tree_1, tree_2, tree_3, grid], ['Tree 1 Level', 'Tree 2 Levels', 'Tree 3 Levels', 'Grid'])
#    plt.savefig('tree_example_73_E_changing_sigma.eps')


#    rho_values=np.load('tree_example_73_continuation_sigma_rho_values.npy')
#    tree_Y_0_values_1=np.load('tree_example_73_one_level_continuation_sigma_1.npy')
#    tree_Y_0_values_2=np.load('tree_example_73_one_level_continuation_sigma_2.npy')
#    tree_Y_0_values_3=np.load('tree_example_73_one_level_continuation_sigma_3.npy')
#
#    rho_values_2=np.load('tree_example_73_rho_values.npy')
#    tree_Y_0_values=np.load('tree_example_73_one_level_changing_rho.npy')
    
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
    
    
#    plt.scatter(x_grid_hist,mu_Pontryagin_hist[20])
#    plt.xlabel('x')
#    plt.ylabel('$\mu(x)$')
#    plt.title('Flocking, Grid Algorithm, Pontryagin Approach, T/2')
#    plt.savefig('flocking_Pontryagin_grid_10_bins.eps')

#    plt.scatter(x_grid_hist,mu_weak_hist[20])
#    plt.xlabel('x')
#    plt.ylabel('$\mu(x)$')
#    plt.title('Flocking, Grid Algorithm, Weak Approach, T/2')
#    plt.savefig('flocking_weak_grid_10_bins.eps')

    plt.scatter(x_grid_hist,mu_true_hist[20])
    plt.xlabel('x')
    plt.ylabel('$\mu(x)$')
    plt.title('Flocking, True Solution, T/2')
    plt.savefig('flocking_true_solution_10_bins_more_accurate.eps')