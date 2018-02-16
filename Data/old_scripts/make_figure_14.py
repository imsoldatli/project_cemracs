#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 12:40:54 2018

@author: christy
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from Wd_exact import *

if __name__ == '__main__':
    rho_values=np.linspace(1,12,12)
    Y_0_values_1=np.load('trader_continuation_time_1_levels_changing_c_x.npy')
    Y_0_values_2=np.load('trader_continuation_time_2_levels_changing_c_x.npy')
    Y_0_values_3=np.load('trader_continuation_time_3_levels_changing_c_x.npy')
    true_Y_0_values=np.load('trader_true_Y_0_fig_14.npy')

    num_keep=5
    for index2 in range(num_keep):
        for index in range(len(rho_values)):
            plot_1=plt.scatter(rho_values[index],Y_0_values_1[index][index2],color='blue', marker="o")
            plot_2=plt.scatter(rho_values[index],Y_0_values_2[index][index2],color='red',marker='s')
            plot_3=plt.scatter(rho_values[index],Y_0_values_3[index][index2],color='green',marker='^')
            plot_4=plt.scatter(rho_values[index],true_Y_0_values[index],color='black',marker='*')
    plt.xlabel('$c_X$')
    plt.ylabel('$Y_0$')
    plt.legend([plot_1, plot_1, plot_1, plot_4], ['Grid, Pontryagin, $N_l=1$', 'Grid, Pontryagin, $N_l=2$','Grid, Pontryagin, $N_l=3$', 'True'],bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
    plt.savefig('paper_trader_grid_continuation_t12_new_Y_0.eps')