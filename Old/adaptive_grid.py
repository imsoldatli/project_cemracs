#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:37:27 2017
@author: Andrea Angiuli, Christy Graves, Houzhi Li
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def b_dummy(i,j,mu,u,v,rho):
    return 0

def b_example_1(i,j,mu,u,v,rho):
    Y_mean=np.dot(u[i],mu[i])
    return -rho*Y_mean
    
def b_example_72(i,j,mu,u,v,rho):
    return rho*np.cos(u[i][j])
    
def b_example_73(i,j,mu,u,v,rho):
    return -rho*u[i][j]
    
def f_example_1(i,j,mu,u,v):
    return a*u[i][j]

def f_example_72(i,j,mu,u,v):
    return 0

def f_example_73(i,j,mu,u,v):
    X_mean=np.dot(x_grid,mu[i])
    return -np.arctan(X_mean)

def g_example_1(x):
    return x
    
def g_example_72(x):
    return np.sin(x)
    
def g_example_73(x):
    return np.arctan(x)

def pi(x):

    low=int((x-x_min)/delta_x)

    if low>=num_x-1:

        x_index=num_x-1

    elif low<0:

        x_index=0

    elif (x-x_min-low*delta_x)<(x_min+(low+1)*delta_x-x):
        
        x_index=low
    else:

        x_index=low+1

    return(x_index)
    
def forward(u,v,mu_0):

    mu=np.zeros((num_t,num_x))
    mu[0,:]=mu_0

    for i in range(num_t-1): #t_i
       for j in range(num_x): #x_j

           low=x_grid[j]+b(i,j,mu,u,v,rho)*delta_t-sigma*math.sqrt(delta_t)
           low_index=pi(low)
           mu[i+1,low_index]+=mu[i,j]*0.5

           up=x_grid[j]+b(i,j,mu,u,v,rho)*delta_t+sigma*math.sqrt(delta_t)
           up_index=pi(up)
           mu[i+1,up_index]+=mu[i,j]*0.5


    #print(mu)
    #print('the sum on each row is', mu.sum(axis=1))
    return mu

def backward(mu,u_old,v_old):
    
    u = np.zeros((num_t,num_x))
    v = np.zeros((num_t,num_x))
        
    u[num_t-1,:] = g(x_grid)
    v[num_t-1,:] = v_old[num_t-1,:] # Not sure if this is right, but doesn't matter for example 1

    for i in reversed(range(num_t-1)):
        for j in range(num_x):
            if i==num_t-2:
                j_down = (x_grid[j] + b(i, j, mu, u_old, v_old, rho) * delta_t - sigma * np.sqrt(delta_t))
            
                j_up = (x_grid[j] + b(i, j, mu, u_old, v_old, rho) * delta_t + sigma * np.sqrt(delta_t))
            
                u[i][j] = (g(j_down) + g(j_up))/2.0 + delta_t*f(i,j,mu,u_old,v_old)
                v[i][j] = 1.0/np.sqrt(delta_t) * (g(j_up) - g(j_down))
            else:
                j_down = pi(x_grid[j] + b(i, j, mu, u_old, v_old, rho) * delta_t - sigma * np.sqrt(delta_t))

                #j_down = pi((x_grid[j] + b(i+1,j,mu,u,v, rho)*delta_t - sigma*np.sqrt(delta_t)))
            
                j_up = pi(x_grid[j] + b(i, j, mu, u_old, v_old, rho) * delta_t + sigma * np.sqrt(delta_t))

                #j_up = pi((x_grid[j] + b(i+1,j,mu,u,v,rho)*delta_t + sigma*np.sqrt(delta_t)))
            
                u[i][j] = (u[i+1][j_down] + u[i+1][j_up])/2.0 + delta_t*f(i,j,mu,u_old,v_old)
                
                #u[i][j] = (u[i+1][j_down] + u[i+1][j_up])/2.0 + delta_t*f(i+1,j,mu,u,v)
                
                v[i][j] = 1.0/np.sqrt(delta_t) * (u[i+1][j_up] - u[i+1][j_down])

    return [u,v]


if __name__ == '__main__':
    global b
    b=b_example_73
    global f
    f=f_example_73
    global g
    g=g_example_73
    global J
    J=10
    global num_keep
    num_keep=5
    global T
    T=1.0    
    global a
    a=0.25
    
    x_min_goal=0
    x_max_goal=4.0
    x_center=(x_min_goal+x_max_goal)/2.0
    
    num_rho=20
    rho_values=np.linspace(2,9,num_rho)
    num_sigma=1
    sigma_values=np.linspace(0.5,10,num_sigma)
    all_Y_0_values=np.zeros((num_rho,num_keep))
    #all_Y_0_values=np.zeros((num_sigma,num_keep))
    for index in range(num_rho):
    #for index in range(num_sigma):
        index2=0
        global rho
        rho=rho_values[index]
        #rho=2.0
        
        global sigma 
        #sigma=sigma_values[index]
        sigma=1
        
        global num_t
#        num_t=int(20*rho/rho_values[0])
#        num_t=int(20/math.sqrt(rho/rho_values[0]))
        num_t=30
        global delta_t
        delta_t=T/(num_t-1)
        global t_grid
        t_grid=np.linspace(0,T,num_t)
        global delta_x

#        delta_x=delta_t**2
        delta_x=(rho+sigma)*delta_t**2
#        delta_x=delta_t
        global num_x
        num_x=int((x_max_goal-x_min_goal)/(delta_x))+1
        if num_x%2==0:
            num_x+=1
        global x_grid
        x_grid=np.linspace(x_center-(num_x-1)/2*delta_x,x_center+(num_x-1)/2*delta_x,num_x)
        global x_min
        x_min=x_grid[0]
        global x_max
        x_max=x_grid[num_x-1]
        print(num_t,delta_x,rho,num_x,x_min)
    
        

    
        mu_0=np.zeros((num_x))
        mu_0[int(num_x/2)]=1.0
        mu=np.zeros((num_t,num_x))
        for k in range(num_t):
            mu[k]=mu_0
        u=np.zeros((num_t,num_x))
        v=np.zeros((num_t,num_x))
    
        for j in range(J):
            [u,v]=backward(mu,u,v)
            mu=forward(u,v,mu_0)
            if j>J-num_keep-1:
                all_Y_0_values[index][index2]=u[0][int(num_x/2)]
                index2+=1
        #print all_Y_0_values[index]
        #print np.dot(mu[num_t-1],g(x_grid))
        
    np.save('adaptive_grid_example_72_changing_rho',all_Y_0_values)
