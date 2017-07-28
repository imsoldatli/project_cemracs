#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:37:27 2017

@author: christy
"""

import numpy as np
import math



if __name__ == '__main__':
    global J
    J=10
    global num_keep
    num_keep=5
    global T
    T=1.0
    global num_t
    num_t=10
    global delta_t
    delta_t=T/(num_t-1)
    global t_grid
    t_grid=linspace(0,T,num_t)
    global x_min
    x_min=-5
    global x_max
    x_max=5
    global num_x
    num_x=10
    global delta_x
    delta_x=(x_max-x_min)/(num_x-1)
    global x_grid
    x_grid=linspace(x_min,x_max,num_x)
    global sigma
    sigma=1

    mu_0=np.zeros((num_x))
    mu_0[num_x/2]=1.0
    mu=np.zeros((num_t,num_x))
    for k in range(num_t):
        for j in range(num_x):
            mu[k][j]=mu_0[j]
    u=np.zeros((num_t,num_x))
    v=np.zeros((num_t,num_x))
    
    for j in range(J):
        [u,v]=backwards(mu,u,v,b,f,g)
        [mu]=forwards(u,v,mu_0,b)