#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:17:50 2017

@author: christy
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from Wd_exact import *

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
#    log_errors=np.load('./ex_1/ex_1_grid_log_errors.npy')
#    log_num_t=np.load('./ex_1/ex_1_grid_log_num_t.npy')
#    plt.scatter(log_num_t,log_errors)
#    plt.xlabel('log(number of time steps)')
#    plt.ylabel('log(error in $Y_0$)')
#    plt.savefig('paper_ex_1_grid.eps')

##### Example 2
#    ### Example 2, tree with 1, 2, 3 levels, and grid, changing rho
#    rho_values=np.load('./ex 72/tree_example_72_rho_values.npy')
#    tree_Y_0_values_1=np.load('./ex 72/tree_example_72_one_level_changing_rho.npy')
#    tree_Y_0_values_2=np.load('./ex 72/tree_example_72_two_level_changing_rho.npy')
#    grid_Y_0_values=np.load('./ex 72/grid_example_72_changing_rho_larger_x_domain.npy')
#    num_keep=5
#    for index2 in range(num_keep):
#        for index in range(len(rho_values)):
#            tree_1=plt.scatter(rho_values[index],tree_Y_0_values_1[index][index2],color='blue', marker="o")
#            tree_2=plt.scatter(rho_values[index],tree_Y_0_values_2[index][index2],color='red',marker='s')
#            grid=plt.scatter(rho_values[index],grid_Y_0_values[index][index2],color='black',marker='*')
#    plt.xlabel('$\\rho$')
#    plt.ylabel('$Y_0$')
#    plt.legend([tree_1, tree_2, grid], ['Tree, $N_l=1$', 'Tree, $N_l=2$', 'Grid, $N_l=1$'],bbox_to_anchor=(0.3, 1), loc=2, borderaxespad=0.)
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

#    ### Example 2, grid with 1, changing sigma (rho=5)
#    sigma_values=np.load('./ex 72/grid_example_72_sigma_values.npy')
#    grid_Y_0_values_one=np.load('./ex 72/grid_example_72_one_level_changing_sigma.npy')
#    num_keep=5
#    num_sigma=20
#    for index in range(num_sigma):
#        for index2 in range(num_keep):
#            grid_1=plt.scatter(sigma_values[index],grid_Y_0_values_one[index][index2],color='black',marker='*')
#    plt.xlabel('$\sigma$')
#    plt.ylabel('$Y_0$')
#    plt.savefig('paper_grid_example_72_changing_sigma.eps')
    
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

###### Example 3
#    ### Example 3, tree with 1, 2, 3 levels, and grid, changing rho
#    rho_values=np.load('./ex 73/tree_example_73_rho_values.npy')
#    tree_Y_0_values_1=np.load('./ex 73/tree_example_73_one_level_changing_rho.npy')
#    tree_Y_0_values_2=np.load('./ex 73/tree_example_73_two_level_changing_rho.npy')
#    tree_Y_0_values_3=np.load('./ex 73/tree_example_73_three_level_changing_rho.npy')
#    grid_Y_0_values=np.load('./ex 73/grid_example_73_changing_rho.npy')
#    num_keep=5
#    for index2 in range(num_keep):
#        for index in range(len(rho_values)):
#            tree_1=plt.scatter(rho_values[index],tree_Y_0_values_1[index][index2],color='blue', marker="o")
#            tree_2=plt.scatter(rho_values[index],tree_Y_0_values_2[index][index2],color='red',marker='s')
#            tree_3=plt.scatter(rho_values[index],tree_Y_0_values_3[index][index2],color='green',marker='^')
#            grid=plt.scatter(rho_values[index],grid_Y_0_values[index][index2],color='black',marker='*')
#    plt.xlabel('$\\rho$')
#    plt.ylabel('$Y_0$')
#    plt.legend([tree_1, tree_2, tree_3, grid], ['Tree, $N_l=1$', 'Tree, $N_l=2$','Tree, $N_l=3$', 'Grid, $N_l=1$'],bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
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
#        for index in range(7):
#            tree_4=plt.scatter(rho_values[index],tree_Y_0_values[index][index2],color='black',marker='*')
#        for index in range(18): #len(rho_values_1)):
#            if index%1==0:
#                tree_1=plt.scatter(rho_values_1[index],tree_Y_0_values_1[index][index2],color='blue',marker='o')
#        for index in range(180): #len(rho_values_2)):
#            if index%10==0:
#                tree_2=plt.scatter(rho_values_2[index],tree_Y_0_values_2[index][index2],color='red',marker='s')
#        for index in range(1800): #len(rho_values_3)):
#            if index%100==0:
#                tree_3=plt.scatter(rho_values_3[index],tree_Y_0_values_3[index][index2],color='green',marker='^')
#    plt.xlabel('$\\rho$')
#    plt.ylabel('$Y_0$')
#    plt.legend([tree_4, tree_1, tree_2, tree_3], ['Tree, $N_l=1$', 'Tree, $N_l=1$, $\Delta\\rho$=0.1', 'Tree, $N_l=1$, $\Delta\\rho$=0.01', 'Tree, $N_l=1$, $\Delta\\rho$=0.001'],bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
#    plt.savefig('paper_tree_example_73_continuation_rho.eps')

#    ### Example 3, tree continuation in sigma
#    rho_values=np.load('./ex 73/tree_example_73_continuation_sigma_rho_values.npy')
#    rho_values_2=np.load('./ex 73/tree_example_73_rho_values.npy')
#    tree_Y_0_values=np.load('./ex 73/tree_example_73_one_level_changing_rho.npy')
#    tree_Y_0_values_1=np.load('./ex 73/tree_example_73_one_level_continuation_sigma_1.npy')
#    tree_Y_0_values_2=np.load('./ex 73/tree_example_73_one_level_continuation_sigma_2.npy')
#    tree_Y_0_values_3=np.load('./ex 73/tree_example_73_one_level_continuation_sigma_3.npy')
#    num_keep=5
#    for index2 in range(num_keep):
#        for index in range(7):
#            tree_4=plt.scatter(rho_values_2[index],tree_Y_0_values[index][index2],color='black',marker='*')
#        for index in range(len(rho_values)):
#            #if index%1==0:
#                tree_1=plt.scatter(rho_values[index],tree_Y_0_values_1[index][index2],color='blue',marker='o')
#        for index in range(len(rho_values)):
#            #if index%10==0:
#                tree_2=plt.scatter(rho_values[index],tree_Y_0_values_2[index][index2],color='red',marker='s')
#        for index in range(len(rho_values)):
#            #if index%100==0:
#                tree_3=plt.scatter(rho_values[index],tree_Y_0_values_3[index][index2],color='green',marker='^')
#    plt.xlabel('$\\rho$')
#    plt.ylabel('$Y_0$')
#    plt.legend([tree_4, tree_1, tree_2, tree_3], ['Tree, $N_l=1$', 'Tree, $N_l=1$, $\Delta\sigma$=1', 'Tree, $N_l=1$, $\Delta\sigma$=0.5', 'Tree, $N_l=1$, $\Delta\sigma$=0.1'],bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
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

#### Example 5 (flocking)
#    ### Example 5 (flocking), mu(x) grid_Pont, grid_weak, true, histogram
#    mu_Pontryagin=np.load('./flocking/flocking_grid_Pont/flocking_mu_Pont_t130.npy')
#    mu_weak=np.load('./flocking/flocking_grid_weak/feb_flocking_mu_weak_t130.npy')
#    mu_true=np.load('./flocking/flocking_true/feb_flocking_mu_true_t130.npy')
#    mu_Pontryagin_end=mu_Pontryagin[len(mu_Pontryagin)-1]
#    mu_weak_end=mu_weak[len(mu_weak)-1]
#    mu_true_end=mu_true #[len(mu_true)-1]
#    num_x=len(mu_Pontryagin[0])
#    x_min=-3
#    x_max=3
#    x_grid=np.linspace(x_min,x_max,num_x)
#    num_t=len(mu_Pontryagin)
#    num_bins=int(num_x/30)
#    num_x_hist=int(num_x/num_bins)
#    delta_x_hist=np.abs(x_max-x_min)/float(num_x_hist)
#    x_grid_hist=np.linspace(x_min,x_max,num_x_hist)
#    mu_weak_hist=np.zeros((num_t,num_x_hist))
#    mu_Pontryagin_hist=np.zeros((num_t,num_x_hist))
#    mu_true_hist=np.zeros((num_t,num_x_hist))
#    for t in range(129,num_t):
#        for i in range(num_x/num_bins):
#            mu_weak_hist[t,i]=np.sum(mu_weak[t,num_bins*i:num_bins*(i+1)])
#            mu_Pontryagin_hist[t,i]=np.sum(mu_Pontryagin[t,num_bins*i:num_bins*(i+1)])
#            mu_true_hist[t,i]=np.sum(mu_true[num_bins*i:num_bins*(i+1)])
#    thing1=mu_Pontryagin_hist[129]/(delta_x_hist*sum(mu_Pontryagin_hist[129]))
#    thing2=mu_weak_hist[129]/(delta_x_hist*sum(mu_weak_hist[129]))
#    thing3=mu_true_hist[129]/(delta_x_hist*sum(mu_true_hist[129]))
#    plot2=plt.scatter(x_grid_hist,thing2,color='red',marker='s')
#    plot1=plt.scatter(x_grid_hist,thing1,color='blue',marker='o')
#    plot3=plt.scatter(x_grid_hist,thing3,color='black',marker='*')
#    plt.xlabel('x')
#    plt.ylabel('$\mu(x)$')
#    plt.legend([plot1,plot2,plot3], ['Grid, Pontryagin', 'Grid, Weak', 'True'],bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
#    plt.savefig('paper_flocking_grid_and_true_t130.eps')

#    ### Example 5 (flocking), mu(x) tree_Pont, tree_weak, true, histogram
#    num_t=20
#    X=np.load('./flocking/flocking_tree/flocking_Pont_tree_X_t20.npy')
#    #data1=plt.hist(X[num_t-1],bins=10)
#    counts=data1[0]
#    counts=counts/sum(counts)
#    markers=data1[1]
#    centers=[(markers[i]+markers[i+1])/2.0 for i in range(len(markers)-1)]
#    print([centers[i+1]-centers[i] for i in range(9)])
#    delta_x=centers[1]-centers[0]
#    plot1=plt.scatter(centers,counts/(delta_x),color='blue',marker='o')
#    plt.xlabel('x')
#    plt.ylabel('$\mu(x)$')
#    X=np.load('./flocking/flocking_tree/flocking_weak_tree_X_t20.npy')
#    #data2=plt.hist(X[num_t-1],bins=10)
#    counts=data2[0]
#    counts=counts/sum(counts)
#    markers=data2[1]
#    centers=[(markers[i]+markers[i+1])/2.0 for i in range(len(markers)-1)]
#    delta_x=centers[1]-centers[0]
#    print([centers[i+1]-centers[i] for i in range(9)])
#    plot2=plt.scatter(centers,counts/delta_x,color='red',marker='s')
#    mean_mu=0
#    variance_mu=0.75709099384536194
#    mu_true_hist=scipy.stats.norm(mean_mu, math.sqrt(variance_mu)).pdf(centers)
#    plt.scatter(centers,mu_true_hist/(delta_x*sum(mu_true_hist)),color='black',marker='*')
#    plt.legend([plot1,plot2,plot3], ['Tree, Pontryagin', 'Tree, Weak', 'True'],bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
#    plt.savefig('paper_flocking_tree_and_true_t20.eps')

#    ### Example 5 (flocking), W2 between true and grid_Pont, and true and grid_weak
#    num_trials=7
#    value_num_t=np.linspace(10,130,num_trials)
#    all_d1=np.zeros(num_trials)
#    all_d2=np.zeros(num_trials)
#    d1=0
#    d2=0
#    for k in range(num_trials):
#        print(k)
#        num_t=value_num_t[k]
#        num_t=int(num_t)
#        mu_Pontryagin=np.load('./flocking/flocking_grid_Pont/flocking_mu_Pont_t'+str(num_t)+'.npy')
#        mu_weak=np.load('./flocking/flocking_grid_weak/feb_flocking_mu_weak_t'+str(num_t)+'.npy')
#        mu_true=np.load('./flocking/flocking_true/feb_flocking_mu_true_t'+str(num_t)+'.npy')
#        mu_Pontryagin_end=mu_Pontryagin[len(mu_Pontryagin)-1]
#        mu_weak_end=mu_weak[len(mu_weak)-1]
#        mu_true_end=mu_true #[len(mu_true)-1]
#        num_x=len(mu_Pontryagin[0])
#        x_min=-3
#        x_max=3
#        x_grid=np.linspace(x_min,x_max,num_x)
#        #d1=Wd_exact_R(x_grid,mu_Pontryagin_end,mu_true_end,2)
#        #print(d1)
#        d1=Wd_approx_R(x_grid,mu_Pontryagin_end,mu_true_end,2)
#        print(d1)
#        #d2=Wd_exact_R(x_grid,mu_weak_end,mu_true_end,2)
#        #print(d2)
#        d2=Wd_approx_R(x_grid,mu_weak_end,mu_true_end,2)
#        print(d2)
#        all_d1[k]=d1
#        all_d2[k]=d2
#    #np.save('flocking_W2_pont',all_d1)
#    #np.save('flocking_W2_weak',all_d2)

#    num_trials=7
#    value_num_t=np.linspace(10,130,num_trials)
#    all_d1=np.load('./flocking/flocking_W2_pont.npy')
#    all_d2=np.load('./flocking/flocking_W2_weak.npy')
#    plot1=plt.scatter(value_num_t,all_d1,color='blue',marker='o')
#    plot2=plt.scatter(value_num_t,all_d2,color='red',marker='s')
#    plt.xlabel('number of time steps')
#    plt.ylabel('$W_2$ distance from true solution at time $T$')
#    plt.legend([plot1, plot2], ['Pontryagin', 'Weak',],bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
#    plt.savefig('paper_flocking_changing_delta_t.eps')

#    plot1=plt.scatter(np.log(value_num_t),np.log(all_d1),color='blue',marker='o')
#    plot2=plt.scatter(np.log(value_num_t),np.log(all_d2),color='red',marker='s')
#    plt.xlabel('log(number of time steps)')
#    plt.ylabel('log($W_2$ distance from true solution at time $T$)')
#    plt.legend([plot1, plot2], ['Pontryagin', 'Weak',],bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
#    plt.savefig('paper_flocking_changing_delta_t_log_log.eps')

#### Example 6 (trader)
#    ### Example 6 (trader), W2 between true and grid_Pont, and true and grid_weak, and true and grid_weak_truncated
#    num_trials=7
#    value_num_t=np.linspace(10,130,num_trials)
#    all_d1=np.zeros(num_trials)
#    all_d2=np.zeros(num_trials)
#    all_d3=np.zeros(num_trials)
#    d1=0
#    d2=0
#    d3=0
#    for k in range(num_trials):
#        print(k)
#        num_t=value_num_t[k]
#        num_t=int(num_t)
#        mu_Pontryagin=np.load('./trader/trader_grid_Pont/feb_trader_mu_Pont_t'+str(num_t)+'.npy')
#        mu_weak=np.load('./trader/trader_grid_weak/feb_trader_mu_weak_t'+str(num_t)+'.npy')
#        #mu_weak_trunc=
#        mu_true=np.load('./trader/trader_true/feb_trader_mu_true_t'+str(num_t)+'.npy')
#        mu_Pontryagin_end=mu_Pontryagin[len(mu_Pontryagin)-1]
#        mu_weak_end=mu_weak[len(mu_weak)-1]
#        #mu_weak_trunc_end=mu_weak_trunc[len(mu_weak_trunc)-1]
#        mu_true_end=mu_true[len(mu_true)-1]
#        num_x=len(mu_Pontryagin[0])
#        x_min=-2
#        x_max=4
#        x_grid=np.linspace(x_min,x_max,num_x)
#        #d1=Wd_exact_R(x_grid,mu_Pontryagin_end,mu_true_end,2)
#        #print(d1)
#        d1=Wd_approx_R(x_grid,mu_Pontryagin_end,mu_true_end,2)
#        print(d1)
#        #d2=Wd_exact_R(x_grid,mu_weak_end,mu_true_end,2)
#        #print(d2)
#        d2=Wd_approx_R(x_grid,mu_weak_end,mu_true_end,2)
#        print(d2)
#        all_d1[k]=d1
#        all_d2[k]=d2
#        #d3=Wd_exact_R(x_grid,mu_weak_trunc_end,mu_true_end,2)
#        #print(d3)
#        #d3=Wd_approx_R(x_grid,mu_weak_trunc_end,mu_true_end,2)
#        #print(d3)
#        #all_d3[k]=d3
#    #np.save('trader_W2_pont',all_d1)
#    #np.save('trader_W2_weak',all_d2)

#    num_trials=7
#    value_num_t=np.linspace(10,130,num_trials)
#    all_d1=np.load('./trader/trader_W2_pont.npy')
#    all_d2=np.load('./trader/trader_W2_weak.npy')
#    plot1=plt.scatter(value_num_t,all_d1,color='blue',marker='o')
#    plot2=plt.scatter(value_num_t,all_d2,color='red',marker='s')
#    plt.xlabel('number of time steps')
#    plt.ylabel('$W_2$ distance from true solution at time $T$')
#    plt.legend([plot1, plot2], ['Pontryagin', 'Weak'],bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
#    plt.savefig('paper_trader_changing_delta_t_new.eps')

#    plot1=plt.scatter(np.log(value_num_t),np.log(all_d1),color='blue',marker='o')
#    plot2=plt.scatter(np.log(value_num_t),np.log(all_d2),color='red',marker='s')
#    #plot3=plt.scatter(value_num_t,all_d3,color='green',marker='^')
#    plt.xlabel('log(number of time steps)')
#    plt.ylabel('log($W_2$ distance from true solution at time $T$)')
#    plt.legend([plot1, plot2], ['Pontryagin', 'Weak'],bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
#    plt.savefig('paper_trader_changing_delta_t_log_log.eps')

#     ### Example 6 (trader), mu(x) grid_Pont, grid_weak, true, histogram
#    mu_Pontryagin=np.load('./trader/trader_grid_Pont/feb_trader_mu_Pont_t130.npy')
#    mu_weak=np.load('./trader/trader_grid_weak/feb_trader_mu_weak_t130.npy')
#    #mu_weak_trunc=
#    mu_true=np.load('./trader/trader_true/feb_trader_mu_true_t130.npy')
#    mu_Pontryagin_end=mu_Pontryagin[len(mu_Pontryagin)-1]
#    mu_weak_end=mu_weak[len(mu_weak)-1]
#    #mu_weak_trunc_end=mu_weak_trunc[len(mu_weak_trunc)-1]
#    mu_true_end=mu_true[len(mu_true)-1]
#    num_x=len(mu_Pontryagin[0])
#    x_min=-2
#    x_max=4
#    x_grid=np.linspace(x_min,x_max,num_x)
#    num_t=len(mu_Pontryagin)
#    num_bins=int(num_x/20) #***
#    num_x_hist=int(num_x/num_bins)
#    delta_x_hist=np.abs(x_max-x_min)/float(num_x_hist)
#    x_grid_hist=np.linspace(x_min,x_max,num_x_hist)
#    mu_weak_hist=np.zeros((num_t,num_x_hist))
#    #mu_weak_trunc_hist=np.zeros((num_t,num_x_hist))
#    mu_Pontryagin_hist=np.zeros((num_t,num_x_hist))
#    mu_true_hist=np.zeros((num_t,num_x_hist))
#    for t in range(num_t):
#        for i in range(num_x/num_bins):
#            mu_weak_hist[t,i]=np.sum(mu_weak[t,num_bins*i:num_bins*(i+1)])
#            #mu_weak_trunc_hist[t,i]=np.sum(mu_weak_trunc[t,num_bins*i:num_bins*(i+1)])
#            mu_Pontryagin_hist[t,i]=np.sum(mu_Pontryagin[t,num_bins*i:num_bins*(i+1)])
#            mu_true_hist[t,i]=np.sum(mu_true[t,num_bins*i:num_bins*(i+1)])
#    thing1=mu_Pontryagin_hist[129]/(delta_x_hist*sum(mu_Pontryagin_hist[129]))
#    thing2=mu_weak_hist[129]/(delta_x_hist*sum(mu_weak_hist[129]))
#    #thing3=mu_weak_trunc_hist[129]/(delta_x_hist*sum(mu_weak_trunc_hist[129]))
#    thing4=mu_true_hist[129]/(delta_x_hist*sum(mu_true_hist[129]))
#    plot2=plt.scatter(x_grid_hist,thing2,color='red',marker='s')
#    plot1=plt.scatter(x_grid_hist,thing1,color='blue',marker='o')
#    #plot3=plt.scatter(x_grid_hist,thing3,color='green',marker='^')
#    plot4=plt.scatter(x_grid_hist,thing4,color='black',marker='*')
#    plt.xlabel('x')
#    plt.ylabel('$\mu(x)$')
#    plt.legend([plot1,plot2,plot4], ['Pontryagin', 'Weak', 'True'],bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
#    plt.savefig('paper_trader_grid_and_true_t130_new.eps')

#    ### Example 6 (trader) bifurcation in c_x with true values of Y_0
#    rho_values=np.linspace(1,12,12)
#    Y_0_values_1=np.load('./trader/trader_continuation_time_1_levels_changing_c_x.npy')
#    Y_0_values_2=np.load('./trader/trader_continuation_time_2_levels_changing_c_x.npy')
#    Y_0_values_3=np.load('./trader/trader_continuation_time_3_levels_changing_c_x.npy')
#    true_Y_0_values=np.load('./trader/trader_true_Y_0_fig_14.npy')
#    num_keep=5
#    for index2 in range(num_keep):
#        for index in range(len(rho_values)):
#            plot_1=plt.scatter(rho_values[index],Y_0_values_1[index][index2],color='blue', marker="o")
#            #plot_2=plt.scatter(rho_values[index],Y_0_values_2[index][index2],color='red',marker='s')
#            #plot_3=plt.scatter(rho_values[index],Y_0_values_3[index][index2],color='green',marker='^')
#            plot_4=plt.scatter(rho_values[index],true_Y_0_values[index],color='black',marker='*')
#    plt.xlabel('$c_X$')
#    plt.ylabel('$Y_0$')
#    plt.legend([plot_1, plot_2, plot_3, plot_4], ['Grid, Pontryagin, $N_l=1$', 'Grid, Pontryagin, $N_l=2$','Grid, Pontryagin, $N_l=3$', 'True'],bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
#    plt.savefig('paper_trader_grid_continuation_t12_new_Y_0.eps')

#    ### Example 6 (trader) error in control
#    trader_true_Y_0_fig_15=np.load('./trader/trader_true_Y_0_fig_15.npy')
#    trader_Y_0_Pont=np.load('./trader/trader_Y_0_Pont.npy')
#    trader_Z_0_weak=np.load('./trader/trader_Z_0_weak.npy')
#    num_trials=7
#    value_num_t=np.linspace(10,130,num_trials)
#    num_keep=1
#    for index2 in range(num_keep):
#        for index in range(len(value_num_t)):
#            diff_Pont=math.log(0.3*abs(trader_Y_0_Pont[index][index2]-trader_true_Y_0_fig_15))
#            diff_weak=math.log(0.3*abs(trader_Z_0_weak[index][index2]/0.7-trader_true_Y_0_fig_15))
#            plot1=plt.scatter(math.log(value_num_t[index]),diff_Pont,color='blue', marker="o")
#            plot2=plt.scatter(math.log(value_num_t[index]),diff_weak,color='red',marker='s')
#    plt.xlabel('log(number of time steps)')
#    plt.ylabel('log(error in $\\alpha_0$)')
#    plt.legend([plot1,plot2], ['Pontryagin', 'Weak'],bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
#    plt.savefig('paper_trader_error_control_log_log.eps')
