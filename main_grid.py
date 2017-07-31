#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:37:27 2017

@author: christy
"""
import numpy as np
import math
import matplotlib.pyplot as plt

def b_example_1(i,j,mu,u,v):
    Y_mean=0
    for k in range(num_x):
        Y_mean+=u[i][k]*mu[i][k]
    return -rho*Y_mean
    
def b_example_72(i,j,mu,u,v):
    return rho*math.cos(u[i][j])
    
def b_example_73(i,j,mu,u,v):
    return -rho*u[i][j]
    
def f_example_1(i,j,mu,u,v):
    return a*u[i][j]

def f_example_72(i,j,mu,u,v):
    return 0

def f_example_73(i,j,mu,u,v):
    X_mean=0
    for k in range(num_x):
        X_mean+=x_grid[k]*mu[i][j]
    return -math.atan(X_mean)

def g_example_1(x):
    return x
    
def g_example_72(x):
    return math.sin(x)
    
def g_example_73(x):
    return math.atan(x)

def pi(x):

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

           low=x_grid[j]+b(i,j,mu,u,v)*delta_t-sigma*math.sqrt(delta_t)
           low_index=pi(low)
           mu[i+1,low_index]+=mu[i,j]*0.5

           up=x_grid[j]+b(i,j,mu,u,v)*delta_t+sigma*math.sqrt(delta_t)
           up_index=pi(up)
           mu[i+1,up_index]+=mu[i,j]*0.5


    #print(mu)
    #print('the sum on each row is', mu.sum(axis=1))
    return mu

def backward(mu,u_old,v_old):
    
    u = np.zeros((num_t,num_x))
    v = np.zeros((num_t,num_x))
    
    for j in range(num_x):
        u[num_t-1][j] = y_grid[pi(g(x_grid[num_t-1][j]),y_min,y_max,delta_y)]
        v[num_t-1][j] = v_old[num_t-1][j]
    

    for i in reversed(range(num_t-1)):
        for j in range(num_x):
            
            j_down = pi((x_grid[j] + b(i, j, mu, u_old, v_old) * delta_t - sigma * sqrt(delta_t)), x_min, x_max, delta_x)

#            j_down = pi((x_grid[j] + b(i+1,j,mu,u,v)*delta_t - sigma*sqrt(delta_t)), x_min,x_max,delta_x)
            
            j_up = pi((x_grid[j] + b(i, j, mu, u_old, v_old) * delta_t + sigma * sqrt(delta_t)), x_min, x_max, delta_x)

#            j_up = pi((x_grid[j] + b(i+1,j,mu,u,v)*delta_t + sigma*sqrt(delta_t)), x_min,x_max,delta_x)
            
            u[i][j] = (u[i+1][j_down] + u[i+1][j_up])/2.0 + delta_t*f(i,j,mu,u_old,v_old)
           
#            u[i][j] = y_grid[pi(((u[i+1][j_down] + delta_t*f(i+1,j_down,mu,u,v) + u[i+1][j_up] + delta_t*f(i+1,j_up,mu,u,v))/2.0), y_min, y_max, delta_y)]
            
            v[i][j] = 1.0/sqrt(delta_t) * (u[i+1][j_up] - u[i+1][j_down])

    return [u,v]

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
    
    global a
    a=0.25
    global rho
    rho=0.1

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
        mu=forward(u,v,mu_0)
