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
#    rho_values_1=np.load('tree_example_73_rho_values_1.npy')
#    rho_values_01=np.load('tree_example_73_rho_values_01.npy')
#    rho_values_001=np.load('tree_example_73_rho_values_001.npy')
#    tree_Y_0_values_1=np.load('tree_example_73_one_level_continuation_rho_1.npy')
#    tree_Y_0_values_01=np.load('tree_example_73_one_level_continuation_rho_01.npy')
#    tree_Y_0_values_001=np.load('tree_example_73_one_level_continuation_rho_001.npy')
#    #grid_Y_0_values=np.load('grid_example_72_changing_rho.npy')
#    rho_values=np.load('tree_example_73_rho_values.npy')
#    tree_Y_0_values=np.load('tree_example_73_one_level_changing_rho.npy')
#    
#    num_keep=5
#    for index2 in range(num_keep):
#        for index in range(len(rho_values_1)):
#            tree_1=plt.scatter(rho_values_1[index],tree_Y_0_values_1[index][index2],color='blue')
#        #for index in range(len(rho_values_01)):
#        #    tree_2=plt.scatter(rho_values_01[index],tree_Y_0_values_01[index][index2],color='red')
#        #for index in range(len(rho_values_001)):
#        #    tree_3=plt.scatter(rho_values_001[index],tree_Y_0_values_001[index][index2],color='green')
#        for index in range(9):
#            tree_3=plt.scatter(rho_values[index],tree_Y_0_values[index][index2],color='black')
#            #tree_2=plt.scatter(rho_values[index],tree_Y_0_values_two[index][index2],color='red')
#            #tree_3=plt.scatter(rho_values[index],tree_Y_0_values_three[index][index2],color='green')
#            #grid=plt.scatter(rho_values[index],grid_Y_0_values[index][index2],color='black')
#    plt.xlabel('rho')
#    plt.ylabel('Y_0')
#    #plt.legend([tree_1, tree_2, tree_3, grid], ['Tree 1 Level', 'Tree 2 Levels', 'Tree 3 Levels', 'Grid'])
#    #plt.savefig('tree_and_grid_example_72_changing_rho.eps')
#    #plt.savefig('tree_example_73_E_changing_rho.eps')
#    plt.savefig('tree_example_73_continuation_rho_no_green_no_red.eps')
    
    
    
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


    rho_values=np.load('tree_example_73_continuation_sigma_rho_values.npy')

    tree_Y_0_values_1=np.load('tree_example_73_one_level_continuation_sigma_1.npy')
    tree_Y_0_values_2=np.load('tree_example_73_one_level_continuation_sigma_2.npy')
    tree_Y_0_values_3=np.load('tree_example_73_one_level_continuation_sigma_3.npy')

    rho_values_2=np.load('tree_example_73_rho_values.npy')
    tree_Y_0_values=np.load('tree_example_73_one_level_changing_rho.npy')
    
    num_keep=5
    for index2 in range(num_keep):
        for index in range(len(rho_values)):
            tree_1=plt.scatter(rho_values[index],tree_Y_0_values_1[index][index2],color='blue')
            tree_1=plt.scatter(rho_values[index],tree_Y_0_values_2[index][index2],color='red')
            tree_1=plt.scatter(rho_values[index],tree_Y_0_values_3[index][index2],color='green')
        for index in range(9):
            tree_3=plt.scatter(rho_values_2[index],tree_Y_0_values[index][index2],color='black')
    plt.xlabel('rho')
    plt.ylabel('Y_0')
    plt.savefig('tree_example_73_continuation_sigma.eps')