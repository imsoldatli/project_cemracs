#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 09:22:46 2017

@author: christy

%compare Y_t and Z_t/sigma and eta_t along the grid
"""
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    path='/home/christy/Dropbox/CEMRACS_MFG/cluster_results/trader_grid/'
    Y=np.load(path+'trader_Y_Pont_t130.npy')
    Z=np.load(path+'trader_Z_weak_t130.npy')/sigma
    eta_t=np.load('./Data/trader/trader_eta_t130.npy')
    
#    path='/home/christy/Dropbox/CEMRACS_MFG/cluster_results/flocking_grid/'
#    Y=np.load(path+'flocking_Y_Pont_t130.npy')
#    Z=np.load(path+'flocking_Z_weak_t130.npy')/sigma
#    eta_t=np.load('./Data/flocking/flocking_eta_t130.npy')
    
#    n=0
#    plt.scatter(x_grid,Y[n],color='blue')
#    plt.scatter(x_grid,Z[n],color='red')
#    plt.scatter(x_grid,eta_t[0][n]*x_grid+0.55049040851287234,color='black')
#    plt.savefig('trader_eta_Y_Z_t130_0.eps')
#    #plt.savefig('flocking_eta_Y_Z_t130_0')
    
    n=0
    plt.plot(x_grid,Y[n],color='blue')
    plt.plot(x_grid,Z[n],color='red')
    plt.plot(x_grid,eta_t[0][n]*x_grid+0.55049040851287234,color='black')
    plt.savefig('trader_eta_Y_Z_t130_0.eps')
    #plt.savefig('flocking_eta_Y_Z_t130_0')