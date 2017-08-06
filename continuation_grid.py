#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:41:29 2017

@author: lihouzhi
"""


import numpy as np
import math
import matplotlib.pyplot as plt
import scipy

def b_dummy(i,j,mu,u,v):
    return 0

def b_example_1(i,j,mu,u,v):
    Y_mean=np.dot(u[i],mu[i])
    return -rho*Y_mean
    
def b_example_72(i,j,mu,u,v):
    return rho*np.cos(u[i][j])
    
def b_example_73(i,j,mu,u,v):
    return -rho*u[i][j]

def b_jet_lag_weak(i,j,mu,u,v):
    return omega_0-omega_S-1.0/(R*sigma)*v[i][j]

def b_jet_lag_Pontryagin(i,j,mu,u,v):
    return omega_0-omega_S-1.0/R*u[i][j]
    
def f_example_1(i,j,mu,u,v):
    return a*u[i][j]

def f_example_72(i,j,mu,u,v):
    return 0

def f_example_73(i,j,mu,u,v):
    X_mean=np.dot(x_grid,mu[i])
    return -np.arctan(X_mean)
    
def f_jet_lag_weak(i,j,mu,u,v):
    value1=1.0/sigma*(omega_0-omega_S)*v[i][j]-1.0/(2*R*sigma**2)*(v[i][j])**2
    c_bar=np.dot(0.5*(np.sin((x_grid[j]-x_grid)/2.0))**2,mu[i])
    value2=K*c_bar
    c_sun=0.5*(np.sin((p-x_grid[j])/2.0))**2
    value3=F*c_sun
    return value1+value2+value3
    
def f_jet_lag_Pontryagin(i,j,mu,u,v):
    partial_c_bar=np.dot(0.5*np.sin((x_grid-x_grid[j])/2.0)*np.cos((x_grid-x_grid[j])/2.0),mu[i])
    value1=-K*partial_c_bar
    partial_c_sun=0.5*np.sin((x_grid[j]-p)/2.0)*np.cos((x_grid[j]-p)/2.0)
    value2=-F*partial_c_sun
    return value1+value2

def g_example_1(x):
    return x
    
def g_example_72(x):
    return np.sin(x)
    
def g_example_73(x):
    return np.arctan(x)
    
def g_jet_lag(x):
    return 0

def pi(x):
    
    if periodic_2_pi:
        x=x%(2*np.pi)

    low=int((x-x_min)/delta_x)

    if low>=num_x-1:
        if periodic_2_pi:
            if (x-x_grid[num_x-1])<(2*np.pi-x):
                x_index=num_x-1
            else:
                x_index=0
        else:
            x_index=num_x-1

    elif low<0:
        x_index=0

    elif (x-x_min-low*delta_x)<(x_min+(low+1)*delta_x-x):
        
        x_index=low
    else:

        x_index=low+1

    return(x_index)

def lin_int(x_min,x_max,y_min,y_max,x_get):
    if x_get>=x_max:
        return y_max
    elif x_get<=x_min:
        return y_min
    else:
        return y_min+(y_max-y_min)/(x_max-x_min)*(x_get-x_min)
    
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
        
    u[num_t-1,:] = g(x_grid)
    v[num_t-1,:] = v_old[num_t-1,:]

    for i in reversed(range(num_t-1)):
        for j in range(num_x):
            if i==num_t-2:
                j_down = (x_grid[j] + b(i, j, mu, u_old, v_old) * delta_t - sigma * np.sqrt(delta_t))
            
                j_up = (x_grid[j] + b(i, j, mu, u_old, v_old) * delta_t + sigma * np.sqrt(delta_t))
            
                u[i][j] = (g(j_down) + g(j_up))/2.0 + delta_t*f(i,j,mu,u_old,v_old)
                v[i][j] = 1.0/np.sqrt(delta_t) * (g(j_up) - g(j_down))
            else:
                j_down = pi(x_grid[j] + b(i, j, mu, u_old, v_old) * delta_t - sigma * np.sqrt(delta_t))

                #j_down = pi((x_grid[j] + b(i+1,j,mu,u,v)*delta_t - sigma*np.sqrt(delta_t)))
            
                j_up = pi(x_grid[j] + b(i, j, mu, u_old, v_old) * delta_t + sigma * np.sqrt(delta_t))

                #j_up = pi((x_grid[j] + b(i+1,j,mu,u,v)*delta_t + sigma*np.sqrt(delta_t)))
            
                u[i][j] = (u[i+1][j_down] + u[i+1][j_up])/2.0 + delta_t*f(i,j,mu,u_old,v_old)
                
                #u[i][j] = (u[i+1][j_down] + u[i+1][j_up])/2.0 + delta_t*f(i+1,j,mu,u,v)
                
                v[i][j] = 1.0/np.sqrt(delta_t) * (u[i+1][j_up] - u[i+1][j_down])

    return [u,v]

def backward(mu,u_old,v_old):
    
    u = np.zeros((num_t,num_x))
    v = np.zeros((num_t,num_x))
        
    u[num_t-1,:] = g(x_grid)
    v[num_t-1,:] = v_old[num_t-1,:]

    for i in reversed(range(num_t-1)):
        for j in range(num_x):
            x_down = x_grid[j] + b(i, j, mu, u_old, v_old) * delta_t - sigma * np.sqrt(delta_t)
                
            x_up = x_grid[j] + b(i, j, mu, u_old, v_old) * delta_t + sigma * np.sqrt(delta_t)
                
            j_down = pi(x_down)

            j_up = pi(x_up)

            if i==num_t-2:
            
                u[i][j] = (g(x_grid[j_down]) + g(x_grid[j_up]))/2.0 + delta_t*f(i,j,mu,u_old,v_old)
                
                v[i][j] = 1.0/np.sqrt(delta_t) * (g(x_grid[j_up]) - g(x_grid[j_down]))

            else:

                if x_down>x_grid[j_down]:
                    if j_down<num_x-1:
                        u_down= lin_int(x_grid[j_down],x_grid[j_down+1],u[i+1][j_down],u[i+1][j_down+1],x_down)
                    else:
                        u_down=u[i+1][j_down]
                else:
                    if j_down>0:
                        u_down= lin_int(x_grid[j_down],x_grid[j_down-1],u[i+1][j_down],u[i+1][j_down-1],x_down)
                    else:
                        u_down=u[i+1][j_down]
                        
                if x_up>x_grid[j_up]:
                    if j_up<num_x-1:
                        u_up= lin_int(x_grid[j_up],x_grid[j_up+1],u[i+1][j_up],u[i+1][j_up+1],x_up)
                    else:
                        u_up=u[i+1][j_up]
                else:
                    if j_up>0:
                        u_up= lin_int(x_grid[j_up],x_grid[j_up-1],u[i+1][j_up],u[i+1][j_up-1],x_up)
                    else:
                        u_up=u[i+1][j_up]

#                u_up = u[i+1][j_up]
#                u_down = u[i+1][j_down]
                
                
                u[i][j] = (u_down + u_up)/2.0 + delta_t*f(i,j,mu,u_old,v_old)
                
                v[i][j] = 1.0/np.sqrt(delta_t) * (u_up - u_down)

    return [u,v]

if __name__ == '__main__':
    global b
    b=b_example_72
    global f
    f=f_example_72
    global g
    g=g_example_72
    global periodic_2_pi
#    periodic_2_pi=True
    periodic_2_pi=False
    global J
    J=10
    global num_keep
    num_keep=5
    global T
#    T=24.0*10
    T=1.0
    global num_t
#    num_t=int(T)*5+1
    num_t=21
    global delta_t
    delta_t=T/(num_t-1)
    global t_grid
    t_grid=np.linspace(0,T,num_t)
    global delta_x
    delta_x=delta_t**2
    global num_x
    global x_grid
    global x_min
    global x_max
    if periodic_2_pi:
        num_x=int((2*np.pi)/(delta_x))+1
        delta_x=2*np.pi/num_x
        x_grid=np.linspace(0,2*np.pi-delta_x,num_x)
    else:
        x_min_goal=-3
        x_max_goal=3
        x_center=(x_min_goal+x_max_goal)/2.0
        num_x=int((x_max_goal-x_min_goal)/(delta_x))+1
        if num_x%2==0:
            num_x+=1
        x_grid=np.linspace(x_center-(num_x-1)/2*delta_x,x_center+(num_x-1)/2*delta_x,num_x)
        
    x_min=x_grid[0]
    x_max=x_grid[num_x-1]
    
    global a
    a=0.25
    global R
    R=1
    global K
    K=0.01
    global F
    F=0.01
    global omega_0
    omega_0=2*np.pi/24.5
    global omega_S
    omega_S=2*np.pi/24
    global p
    p=(3.0/12.0)*np.pi
    
    num_rho=20
    rho_values=np.linspace(0.5,10,num_rho)
    num_sigma=20
    sigma_values=np.linspace(0.5,10,num_sigma)
    all_Y_0_values=np.zeros((num_rho,num_keep))
    #all_Y_0_values=np.zeros((num_sigma,num_keep))
    
#    mu_0=np.zeros((num_x))
#    if periodic_2_pi:
#        mu_0=scipy.io.loadmat('mu_initial_reference_set_158.mat')['mu_initial']
#        #mu_0[0]=1
#    else:
#        mu_0[int(num_x/2)]=1.0
#    mu=np.zeros((num_t,num_x))
#    for k in range(num_t):
#        mu[k]=mu_0
#    u=np.zeros((num_t,num_x))
#    v=np.zeros((num_t,num_x))
    
    for index in range(num_rho):
    #for index in range(num_sigma):
        index2=0
        global rho
        rho=rho_values[index]
#        rho=2.0
        global sigma
        #sigma=sigma_values[index]
        sigma=1.0
    
        mu_0=np.zeros((num_x))
        if periodic_2_pi:
            mu_0=scipy.io.loadmat('mu_initial_reference_set_158.mat')['mu_initial']
            #mu_0[0]=1
        else:
            mu_0[int(num_x/2)]=1.0
        print(x_grid[int(num_x/2)])
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
        print ('rho=',rho)
        print all_Y_0_values[index]
        #for index2 in range(num_keep):
            #plt.scatter(rho,Y_0_values[index2])
            #plt.scatter(sigma,Y_0_values[index2])
    #plt.savefig('two_level_changing_rho_example_72.eps')
    #plt.savefig('one_level_example_73_change_sigma.eps')
    #np.save('grid_example_72_rho_values',rho_values)
    #np.save('grid_example_72_changing_rho',all_Y_0_values)
#    np.save('mu_jet_lag',mu)

