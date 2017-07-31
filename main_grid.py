#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:37:27 2017

@author: christy
"""
import numpy as np
import math
import matplotlib.pyplot as plt

def b_example_1(i,j,mu_0,u,v):
    num_initial=len(X[0])
    Y_mean=0
    for k in range(len(Y[i])):
        num_per_initial=len(Y[i])/num_initial
        index=int(math.floor(k/num_per_initial))
        Y_mean+=Y[i][k]*X_initial_probs[index]/num_per_initial
    return -rho*Y_mean
    
def b_example_72(i,j,mu_0,u,v):
    return rho*math.cos(Y[i][j])
    
def b_example_73(i,j,mu_0,u,v):
    return -rho*Y[i][j]
    
def f_example_1(i,j,X,Y,Z,X_initial_probs):
    return a*Y[i][j]

def f_example_72(i,j,X,Y,Z,X_initial_probs):
    return 0

def f_example_73(i,j,X,Y,Z,X_initial_probs):
    num_initial=len(X[0])
    X_mean=0
    for k in range(len(X[i])):
        num_per_initial=len(X[i])/num_initial
        index=int(math.floor(k/num_per_initial))
        X_mean+=X[i][k]*X_initial_probs[index]/num_per_initial
    return -math.atan(X_mean)

def g_example_1(x):
    return x
    
def g_example_72(x):
    return math.sin(x)
    
def g_example_73(x):
    return math.atan(x)

def pi(x,x_min,x_max,delta_x):

    low=float(x-x_min)//delta_x

    if low>=num_x-1:

        x_index=num_x-1

    elif low<1:

        x_index=0

    elif (x-x_min-low*delta_x)<(x_min+(low+1)*delta_x-x):

        x_index=low
    else:

        x_index=low+1

    return(int(x_index))
    
def forward(u,v,mu_0):

    mu=np.zeros((num_t,num_x))
    mu[0,:]=mu_0

    for i in range(num_t-1): #t_i
       for j in range(num_x): #x_j

           low=x_grid[j]+b(i,j,mu_0,u,v)*delta_t-sigma*sqrt(delta_t)
           low_index=pi(low,x_min,x_max,delta_x)
           mu[i+1,low_index]+=mu[i,j]*0.5

           up=x_grid[j]+b(i,j,mu_0,u,v)*delta_t+sigma*sqrt(delta_t)
           up_index=pi(up,x_min,x_max,delta_x)
           mu[i+1,up_index]+=mu[i,j]*0.5


    print(mu)
    print('the sum on each row is', mu.sum(axis=1))
    return mu

if __name__ == '__main__':
    global b
    b=b_example_1
    global f
    f=f_example_1
    global g
    g=g_example_1
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
    t_grid=np.linspace(0,T,num_t)
    global x_min
    x_min=-5
    global x_max
    x_max=5
    global num_x
    num_x=10
    global delta_x
    delta_x=float(x_max-x_min)/(num_x-1)
    global x_grid
    x_grid=np.linspace(x_min,x_max,num_x)
    global sigma
    sigma=1

    mu_0=np.zeros((num_x))
    mu_0[int(num_x/2)]=1.0
    mu=np.zeros((num_t,num_x))
    for k in range(num_t):
        for j in range(num_x):
            mu[k][j]=mu_0[j]
    u=np.zeros((num_t,num_x))
    v=np.zeros((num_t,num_x))
    
    for j in range(J):
     #   [u,v]=backward(mu,u,v)
        [mu]=forward(u,v,mu_0)
