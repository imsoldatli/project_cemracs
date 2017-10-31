#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:17:50 2017

@author: christy
"""

import numpy as np
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
                    ###########Please don't alter anything here!!
                    ####### make new plots below
    
#### Example 1
#    ### log errors Example 1 tree level with one level
#    log_errors=np.load('./ex_1/ex_1_log_errors_tree_one_level.npy')
#    log_num_t=np.load('ex_1/ex_1_log_num_t_tree_one_level.npy')
#    plt.scatter(log_num_t,log_errors)
#    plt.xlabel('log(number of time steps)')
#    plt.ylabel('log(error in $Y_0$)')
#    plt.savefig('paper_ex_1_tree_one_level.eps')

#    ### log errors Example 1 grid
#    log_errors=np.load('./ex_1/ex_1_log_errors.npy')
#    log_num_t=np.load('ex_1/ex_1_log_num_t.npy')
#    plt.scatter(log_num_t,log_errors)
#    plt.xlabel('log(number of time steps)')
#    plt.ylabel('log(error in $Y_0$)')
#    plt.savefig('paper_ex_1_grid.eps')

#### Example 2
#    ### Example 2, tree with 1, 2, 3 levels, and grid, changing rho
#    rho_values=np.load('./ex 72/tree_example_72_rho_values.npy')
#    tree_Y_0_values_1=np.load('./ex 72/tree_example_72_one_level_changing_rho.npy')
#    tree_Y_0_values_2=np.load('./ex 72/tree_example_72_two_level_changing_rho.npy')
#    grid_Y_0_values=np.load('./ex 72/grid_example_72_changing_rho_larger_x_domain.npy')
#
#    num_keep=5
#    for index2 in range(num_keep):
#        for index in range(len(rho_values)):
#            tree_1=plt.scatter(rho_values[index],tree_Y_0_values_1[index][index2],color='blue', marker="o")
#            tree_2=plt.scatter(rho_values[index],tree_Y_0_values_2[index][index2],color='red',marker='s')
#            grid=plt.scatter(rho_values[index],grid_Y_0_values[index][index2],color='black',marker='*')
#    plt.xlabel('$\\rho$')
#    plt.ylabel('$Y_0$')
#    plt.legend([tree_1, tree_2, grid], ['Tree, N=1', 'Tree, N=2', 'Grid, N=1'],bbox_to_anchor=(0.3, 1), loc=2, borderaxespad=0.)
#    plt.savefig('paper_tree_and_grid_example_72_changing_rho_larger_x_domain.eps')

#    ### Example 2, tree with 1, changing sigma (rho=5)
#    sigma_values=np.load('./ex 72/tree_example_72_sigma_values.npy')
#    tree_Y_0_values_one=np.load('./ex 72/tree_example_72_one_level_changing_sigma.npy')
#    num_keep=5
#    num_sigma=20
#    for index in range(num_sigma):
#        for index2 in range(num_keep):
#            tree_1=plt.scatter(sigma_values[index],tree_Y_0_values_one[index][index2],color='blue',marker='o')
#    plt.xlabel('$\sigma$')
#    plt.ylabel('$Y_0$')
#    plt.savefig('paper_tree_example_72_changing_sigma.eps')
    
#    ### Example 2, tree with 1, changing sigma (rho=3.5)
#    sigma_values=np.load('./ex 72/tree_example_72_sigma_values.npy')
#    tree_Y_0_values_one=np.load('./ex 72/tree_example_72_one_level_changing_sigma_rho_3_5.npy')
#    num_keep=5
#    num_sigma=20
#    for index in range(num_sigma):
#        for index2 in range(num_keep):
#            tree_1=plt.scatter(sigma_values[index],tree_Y_0_values_one[index][index2],color='blue',marker='o')
#    plt.xlabel('$\sigma$')
#    plt.ylabel('$Y_0$')
#    plt.savefig('paper_tree_example_72_changing_sigma_rho_3_5.eps')
    
#    ### Example 2, tree with 1, changing sigma (rho=4)
#    sigma_values=np.load('./ex 72/tree_example_72_sigma_values.npy')
#    tree_Y_0_values_one=np.load('./ex 72/tree_example_72_one_level_changing_sigma_rho_4.npy')
#    num_keep=5
#    num_sigma=20
#    for index in range(num_sigma):
#        for index2 in range(num_keep):
#            tree_1=plt.scatter(sigma_values[index],tree_Y_0_values_one[index][index2],color='blue',marker='o')
#    plt.xlabel('$\sigma$')
#    plt.ylabel('$Y_0$')
#    plt.savefig('paper_tree_example_72_changing_sigma_rho_4.eps')

#### Example 3
#    ### Example 3, tree with 1, 2, 3 levels, and grid, changing rho
#    rho_values=np.load('./ex 73/tree_example_73_rho_values.npy')
#    tree_Y_0_values_1=np.load('./ex 73/tree_example_73_one_level_changing_rho.npy')
#    tree_Y_0_values_2=np.load('./ex 73/tree_example_73_two_level_changing_rho.npy')
#    grid_Y_0_values=np.load('./ex 73/grid_example_73_changing_rho.npy')
#
#    num_keep=5
#    for index2 in range(num_keep):
#        for index in range(len(rho_values)):
#            tree_1=plt.scatter(rho_values[index],tree_Y_0_values_1[index][index2],color='blue', marker="o")
#            tree_2=plt.scatter(rho_values[index],tree_Y_0_values_2[index][index2],color='red',marker='s')
#            grid=plt.scatter(rho_values[index],grid_Y_0_values[index][index2],color='black',marker='*')
#    plt.xlabel('$\\rho$')
#    plt.ylabel('$Y_0$')
#    plt.legend([tree_1, tree_2, grid], ['Tree, N=1', 'Tree, N=2', 'Grid, N=1'],bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
#    plt.savefig('paper_tree_and_grid_example_73_changing_rho.eps')

#    ### Example 3, tree with 1, changing sigma (rho=2)
#    sigma_values=np.load('./ex 73/tree_example_73_sigma_values.npy')
#    tree_Y_0_values_one=np.load('./ex 73/tree_example_73_one_level_changing_sigma.npy')
#    num_keep=5
#    num_sigma=20
#    for index in range(num_sigma):
#        for index2 in range(num_keep):
#            tree_1=plt.scatter(sigma_values[index],tree_Y_0_values_one[index][index2],color='blue',marker='o')
#    plt.xlabel('$\sigma$')
#    plt.ylabel('$Y_0$')
#    plt.savefig('paper_tree_example_73_changing_sigma.eps')


#    ### Example 3, tree continuation in rho
#    rho_values=np.load('./ex 73/tree_example_73_rho_values.npy')
#    tree_Y_0_values=np.load('./ex 73/tree_example_73_one_level_changing_rho.npy')
#    rho_values_1=np.load('./ex 73/tree_example_73_rho_values_1.npy')
#    rho_values_2=np.load('./ex 73/tree_example_73_rho_values_01.npy')
#    rho_values_3=np.load('./ex 73/tree_example_73_rho_values_001.npy')
#    tree_Y_0_values_1=np.load('./ex 73/tree_example_73_one_level_continuation_rho_1.npy')
#    tree_Y_0_values_2=np.load('./ex 73/tree_example_73_one_level_continuation_rho_01.npy')
#    tree_Y_0_values_3=np.load('./ex 73/tree_example_73_one_level_continuation_rho_001.npy')
#    num_keep=5
#    for index2 in range(num_keep):
#        for index in range(9):
#            tree_4=plt.scatter(rho_values[index],tree_Y_0_values[index][index2],color='black',marker='*')
#        for index in range(len(rho_values_1)):
#            if index%1==0:
#                tree_1=plt.scatter(rho_values_1[index],tree_Y_0_values_1[index][index2],color='blue',marker='o')
#        for index in range(len(rho_values_2)):
#            if index%10==0:
#                tree_2=plt.scatter(rho_values_2[index],tree_Y_0_values_2[index][index2],color='red',marker='s')
#        for index in range(len(rho_values_3)):
#            if index%100==0:
#                tree_3=plt.scatter(rho_values_3[index],tree_Y_0_values_3[index][index2],color='green',marker='^')
#    plt.xlabel('$\\rho$')
#    plt.ylabel('$Y_0$')
#    plt.legend([tree_4, tree_1, tree_2, tree_3], ['Tree, N=1', 'Tree, N=1, $\Delta\\rho$=0.1', 'Tree, N=1, $\Delta\\rho$=0.01', 'Tree, N=1, $\Delta\\rho$=0.001'],bbox_to_anchor=(0, 0), loc=3, borderaxespad=0.)
#    plt.savefig('paper_tree_example_73_continuation_rho.eps')

#    ### Example 3, tree continuation in rho
#    rho_values=np.load('./ex 73/tree_example_73_continuation_sigma_rho_values.npy')
#    rho_values_2=np.load('./ex 73/tree_example_73_rho_values.npy')
#    tree_Y_0_values=np.load('./ex 73/tree_example_73_one_level_changing_rho.npy')
#    tree_Y_0_values_1=np.load('./ex 73/tree_example_73_one_level_continuation_sigma_1.npy')
#    tree_Y_0_values_2=np.load('./ex 73/tree_example_73_one_level_continuation_sigma_2.npy')
#    tree_Y_0_values_3=np.load('./ex 73/tree_example_73_one_level_continuation_sigma_3.npy')
#    num_keep=5
#    for index2 in range(num_keep):
#        for index in range(9):
#            tree_4=plt.scatter(rho_values_2[index],tree_Y_0_values[index][index2],color='black',marker='*')
#        for index in range(len(rho_values)):
#            if index%1==0:
#                tree_1=plt.scatter(rho_values[index],tree_Y_0_values_1[index][index2],color='blue',marker='o')
#        for index in range(len(rho_values)):
#            if index%10==0:
#                tree_2=plt.scatter(rho_values[index],tree_Y_0_values_2[index][index2],color='red',marker='s')
#        for index in range(len(rho_values)):
#            if index%100==0:
#                tree_3=plt.scatter(rho_values[index],tree_Y_0_values_3[index][index2],color='green',marker='^')
#    plt.xlabel('$\\rho$')
#    plt.ylabel('$Y_0$')
#    plt.legend([tree_4, tree_1, tree_2, tree_3], ['Tree, N=1', 'Tree, N=1, $\Delta\sigma$=1', 'Tree, N=1, $\Delta\sigma$=0.5', 'Tree, N=1, $\Delta\sigma$=0.1'],bbox_to_anchor=(0, 0), loc=3, borderaxespad=0.)
#    plt.savefig('paper_tree_example_73_continuation_sigma.eps')

#### Example 4
#    ### Example 4, tree with 1
#    rho_values=np.load('./ex 73/tree_example_73_E_rho_values.npy')
#    tree_Y_0_values_1=np.load('./ex 73/tree_example_73_E_one_level_changing_rho.npy')
#    num_keep=5
#    for index2 in range(num_keep):
#        for index in range(len(rho_values)):
#            tree_1=plt.scatter(rho_values[index],tree_Y_0_values_1[index][index2],color='blue', marker="o")
#    plt.xlabel('$\\rho$')
#    plt.ylabel('$Y_0$')
#    plt.savefig('paper_tree_example_73_E_changing_rho.eps')

#    ### Example 4, tree with 1, changing sigma (rho=?)
#    sigma_values=np.load('./ex 73/tree_example_73_E_sigma_values.npy')
#    tree_Y_0_values_one=np.load('./ex 73/tree_example_73_E_one_level_changing_sigma.npy')
#    num_keep=5
#    num_sigma=20
#    for index in range(num_sigma):
#        for index2 in range(num_keep):
#            tree_1=plt.scatter(sigma_values[index],tree_Y_0_values_one[index][index2],color='blue',marker='o')
#    plt.xlabel('$\sigma$')
#    plt.ylabel('$Y_0$')
#    plt.savefig('paper_tree_example_73_E_changing_sigma.eps')