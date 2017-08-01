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
    #rho_values=np.load('tree_example_73_E_rho_values.npy')
    #tree_Y_0_values_one=np.load('tree_example_73_E_one_level_changing_rho.npy')
    #grid_Y_0_values=np.load('grid_example_72_changing_rho.npy')
    
    #num_keep=5
    #num_rho=20
    #for index in range(num_rho):
    #    for index2 in range(num_keep):
    #        tree_1=plt.scatter(rho_values[index],tree_Y_0_values_one[index][index2],color='blue')
    #        #tree_2=plt.scatter(rho_values[index],tree_Y_0_values_two[index][index2],color='red')
    #        #tree_3=plt.scatter(rho_values[index],tree_Y_0_values_three[index][index2],color='green')
    #        #grid=plt.scatter(rho_values[index],grid_Y_0_values[index][index2],color='black')
    #plt.xlabel('rho')
    #plt.ylabel('Y_0')
    ##plt.legend([tree_1, tree_2, tree_3, grid], ['Tree 1 Level', 'Tree 2 Levels', 'Tree 3 Levels', 'Grid'])
    ##plt.savefig('tree_and_grid_example_72_changing_rho.eps')
    #plt.savefig('tree_example_73_E_changing_rho.eps')
    
    
    sigma_values=np.load('tree_example_73_E_sigma_values.npy')
    tree_Y_0_values_one=np.load('tree_example_73_E_one_level_changing_sigma.npy')
    num_keep=5
    num_sigma=20
    for index in range(num_sigma):
        for index2 in range(num_keep):
            tree_1=plt.scatter(sigma_values[index],tree_Y_0_values_one[index][index2],color='blue')
            #grid=plt.scatter(sigma_values[index],grid_Y_0_values[index][index2],color='black')
    plt.xlabel('sigma')
    plt.ylabel('Y_0')
    #plt.legend([tree_1, tree_2, tree_3, grid], ['Tree 1 Level', 'Tree 2 Levels', 'Tree 3 Levels', 'Grid'])
    plt.savefig('tree_example_73_E_changing_sigma.eps')