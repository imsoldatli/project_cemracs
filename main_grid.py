#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:37:27 2017
    
@author: Andrea Angiuli, Christy Graves, Houzhi Li

This code implements the grid algorithm described in
Delarue, Menozzi to solve FBSDEs of McKean Vlasov type.

The FBSDEs to be solved are the following:
    dX_t=b(X,Y,Z,Law(X),Law(Y),Law(Z))dt+sigma dW_t
    X_0=x_0
    dY_t=-f(X,Y,Z,Law(X),Law(Y),Law(Z)) dt+Z_t dW_t
    Y_t=g(X_T,Law(X_T))
    
Possible future extensions include time dependency of b and f, sigma
non constant, and multidimensional state space.

"""

from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import time


#Define functions b, f, and g for a variety of problems:

def b_dummy(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    return 0

def b_example_1(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    #Y_mean=np.dot(u[i],mu[i])
    #Y_mean=Y_mean_all[i]
    return -rho*Y_mean

def f_example_1(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    return a*u[i][j]

def g_example_1(x):
    return x

def b_example_72(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    return rho*np.cos(u[i][j])

def f_example_72(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    return 0

def g_example_72(x):
    return np.sin(x)

def b_example_73(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    return -rho*u[i][j]

def f_example_73(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
#    X_mean=np.dot(x_grid,mu[i])
    #X_mean=X_mean_all[i]
    return -np.arctan(X_mean)

def g_example_73(x):
    return np.arctan(x)

def b_jet_lag_weak(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    return omega_0-omega_S-1.0/(R*sigma)*v[i][j]

def f_jet_lag_weak(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    value1=1.0/sigma*(omega_0-omega_S)*v[i][j]-1.0/(2*R*sigma**2)*(v[i][j])**2

#    mu_pad=np.zeros((3*num_x-2))
#    mu_pad[0:num_x]=mu[i][:]
#    c_bar_2=np.fft.ifft(np.fft.fft(mu_pad)*fft_h_pad)
#    c_bar=c_bar_2[num_x-1:2*num_x-1][j]
#    c_bar=np.real(c_bar)
    #c_bar=np.dot(0.5*(np.sin((x_grid[j]-x_grid)/2.0))**2,mu[i])
    c_bar=convolution[j]
    value2=K*c_bar
    c_sun=0.5*(np.sin((p-x_grid[j])/2.0))**2
    value3=F*c_sun
    return value1+value2+value3

def g_jet_lag(x):
    return 0
    
def get_h_jet_lag_weak():
    temp=[0.5*np.sin((i)*delta_x/2.0)**2 for i in range(num_x)]
    temp=np.asanyarray(temp)
    temp2=[temp[num_x-i-1] for i in range(num_x)]
    temp2=np.asanyarray(temp2)
    h_array=np.concatenate((temp2,temp[1:len(temp)]))
    return h_array

def b_jet_lag_Pontryagin(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    return omega_0-omega_S-1.0/R*u[i][j]

def f_jet_lag_Pontryagin(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
#    mu_pad=np.zeros((3*num_x-2))
#    mu_pad[0:num_x]=mu[i][:]
#    partial_c_bar_2=np.fft.ifft(np.fft.fft(mu_pad)*fft_h_pad)
#    partial_c_bar=partial_c_bar_2[num_x-1:2*num_x-1][j]
#    partial_c_bar=np.real(partial_c_bar)
    #partial_c_bar=np.dot(0.5*np.sin((x_grid[j]-x_grid)/2.0)*np.cos((x_grid[j]-x_grid)/2.0),mu[i])
    partial_c_bar=convolution[j]
    value1=K*partial_c_bar
    partial_c_sun=0.5*np.sin((x_grid[j]-p)/2.0)*np.cos((x_grid[j]-p)/2.0)
    value2=F*partial_c_sun
    return value1+value2
    
def get_h_jet_lag_Pontryagin():
    temp=[0.5*np.sin(i*delta_x/2.0)*np.cos(i*delta_x/2.0) for i in range(num_x)]
    temp=np.asanyarray(temp)
    temp2=[-temp[num_x-i-1] for i in range(num_x)]
    temp2=np.asanyarray(temp2)
    h_array=np.concatenate((temp2,temp[1:len(temp)]))
    return h_array

def b_trader_Pontryagin(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    return -rho*u[i][j] #rho=1/c_alpha

def f_trader_Pontryagin(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    #Y_mean=np.dot(u[i],mu[i])
    #Y_mean=Y_mean_all[i]
    return c_x*x_grid[j]+h_bar*rho*Y_mean

def g_trader_Pontryagin(x):
    return c_g *x

def b_trader_weak(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    return -rho*v[i][j]/sigma #rho=1/c_alpha

def f_trader_weak(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    #Z_mean=np.dot(v[i],mu[i])
    #Z_mean=Z_mean_all[i]
    return 0.5*c_x*x_grid[j]**2+x_grid[j]*h_bar*rho*Z_mean/sigma-rho*0.5*v[i][j]**2/sigma**2

def g_trader_weak(x):
    return c_g*0.5*x**2

def b_trader_weak_trunc(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    return -rho*v[i][j]/sigma #rho=1/c_alpha

def f_trader_weak_trunc(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    #Z_mean=np.dot(v[i],mu[i])
    #Z_mean=Z_mean_all[i]
    value=0
    if v[i,j]<bounds[0,i]:
        value=0.5*c_x*x_grid[j]**2+x_grid[j]*h_bar*rho*Z_mean/sigma-rho*0.5*bounds[0,i]**2/sigma**2
    elif v[i,j] > bounds[1,i]:
        value= 0.5*c_x*x_grid[j]**2+x_grid[j]*h_bar*rho*Z_mean/sigma-rho*0.5*bounds[1,i]**2/sigma**2
    else:
        value=0.5*c_x*x_grid[j]**2+x_grid[j]*h_bar*rho*Z_mean/sigma-rho*0.5*v[i][j]**2/sigma**2
    return(value)

def g_trader_weak_trunc(x):
    return c_g*0.5*x**2

def b_trader_solution(i,j,mu,u,v):
    x_mean=np.dot(x_grid,mu[i])
    return -rho*(eta[0,i]*x_grid[j]+(eta_bar[0,i]-eta[0,i])*x_mean)

def f_trader_solution(i,j,mu,u,v):
    return 0

def g_trader_solution(x):
    return 0
    
def b_flocking_Pontryagin(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    return -u[i][j]

def f_flocking_Pontryagin(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    #X_mean=np.dot(x_grid,mu[i])
    #X_mean=X_mean_all[i]
    return x_grid[j]-X_mean

def g_flocking(x):
    return 0

def b_flocking_weak(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    return -v[i][j]/sigma

def f_flocking_weak(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution):
    #X_mean=np.dot(x_grid,mu[i])
    #X_mean=X_mean_all[i]
    return -1.0/(2*sigma**2)*(v[i][j])**2+0.5*(x_grid[j]-X_mean)**2

#project the value x onto the nearest value in x_grid
def pi_old(x):
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
    
def pi_old_2(x):
    if periodic_2_pi:
        x=x%(2*np.pi)
    index=int(round(((x-x_min)/delta_x)))
    
    if periodic_2_pi:
        index=index%num_x
    else:
        index=min(num_x-1,index)
    index=max(0,index)

    return index
    
def pi(x):
    if periodic_2_pi:
        x=x%(2*np.pi)
        index=int(round(((x-x_min)/delta_x)))
        index=index%num_x
    else:
        index=int(round(((x-x_min)/delta_x)))
        index=min(num_x-1,index)
        index=max(0,index)
    return index

#used to linearly interpolate u(x) using u on x_grid
def lin_int(x_m,x_M,y_m,y_M,x_get):
    if x_get>=x_M:
        return y_M
    elif x_get<=x_m:
        return y_m
    else:
        return y_m+(y_M-y_m)/(x_M-x_m)*(x_get-x_m)

#use mu_0, u, v, to go forward in mu
def forward(u,v,mu_0):
    mu=np.zeros((num_t,num_x))
    mu[0,:]=mu_0
    
    for i in range(num_t-1): #t_i
        convolution=0
        X_mean=0
        Y_mean=0
        Z_mean=0
        if problem=='jetlag_weak' or problem=='jetlag_Pontryagin':
            convolution=np.zeros((num_x))
            mu_pad=np.zeros((3*num_x-2))
            mu_pad[0:num_x]=mu[i][:]
            conv_2=np.fft.ifft(np.fft.fft(mu_pad)*fft_h_pad)
            conv=conv_2[num_x-1:2*num_x-1]
            convolution=np.real(conv)
        elif problem=='ex_73' or problem=='flocking_Pontryagin' or problem=='flocking_weak':
            X_mean=np.dot(x_grid,mu[i])
        elif problem=='ex_1' or problem=='trader_Pontryagin':
            Y_mean=np.dot(u[i],mu[i])
        elif problem=='trader_weak':
            Z_mean=np.dot(v[i],mu[i])

        for j in range(num_x): #x_j
            low=x_grid[j]+b(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution)*delta_t-sigma*sqrt_delta_t
            low_index=pi(low)
            mu[i+1,low_index]+=mu[i,j]*0.5
                        
            up=x_grid[j]+b(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution)*delta_t+sigma*sqrt_delta_t
            up_index=pi(up)
            mu[i+1,up_index]+=mu[i,j]*0.5
    return mu

#use mu, u_old, v_old to go backwards in u and v
def backward(mu,u_old,v_old):    
    u = np.zeros((num_t,num_x))
    v = np.zeros((num_t,num_x))
    
    u[num_t-1,:] = g(x_grid)
    v[num_t-1,:] = v_old[num_t-1,:]
    if problem=='trader_weak_trunc':
        for i in reversed(range(num_t-1)):
            convolution=0
            X_mean=0
            Y_mean=0
            Z_mean=0
            if problem=='jetlag_weak' or problem=='jetlag_Pontryagin':
                convolution=np.zeros((num_x))
                mu_pad=np.zeros((3*num_x-2))
                mu_pad[0:num_x]=mu[i][:]
                conv_2=np.fft.ifft(np.fft.fft(mu_pad)*fft_h_pad)
                conv=conv_2[num_x-1:2*num_x-1]
                convolution=np.real(conv)
            elif problem=='ex_73' or problem=='flocking_Pontryagin' or problem=='flocking_weak':
                X_mean=np.dot(x_grid,mu[i])
            elif problem=='ex_1' or problem=='trader_Pontryagin':
                Y_mean=np.dot(u_old[i],mu[i])
            elif problem=='trader_weak':
                Z_mean=np.dot(v_old[i],mu[i])
            for j in range(num_x):
                x_down = x_grid[j] + b(i, j, mu, u_old, v_old,X_mean,Y_mean,Z_mean,convolution) * delta_t - sigma * sqrt_delta_t

                x_up = x_grid[j] + b(i, j, mu, u_old, v_old,X_mean,Y_mean,Z_mean,convolution) * delta_t + sigma * sqrt_delta_t

                j_down = pi(x_down)

                j_up = pi(x_up)

                if i==num_t-2:

                    u[i][j] = (g(x_down) + g(x_up))/2.0 + delta_t*f(i,j,mu,u_old,v_old,X_mean,Y_mean,Z_mean,convolution)

                    v[i][j] = 1.0/sqrt_delta_t * (g(x_up) - g(x_down))

                    if v[i,j]<bounds[0,i]:
                        v[i,j]=bounds[0,i]
                    elif v[i,j]>bounds[1,i]:
                        v[i,j]=bounds[1,i]
                else:
                    u_up = u[i+1][j_up]
                    u_down = u[i+1][j_down]
    
                    u[i][j] = (u_down + u_up)/2.0 + delta_t*f(i,j,mu,u_old,v_old,X_mean,Y_mean,Z_mean,convolution)
    
                    v[i][j] = 1.0/sqrt_delta_t * (u_up - u_down)
                    if v[i,j]<bounds[0,i]:
                        v[i,j]=bounds[0,i]
                    elif v[i,j]>bounds[1,i]:
                        v[i,j]=bounds[1,i]

    else:
        for i in reversed(range(num_t-1)):
            convolution=0
            X_mean=0
            Y_mean=0
            Z_mean=0
            if problem=='jetlag_weak' or problem=='jetlag_Pontryagin':
                convolution=np.zeros((num_x))
                mu_pad=np.zeros((3*num_x-2))
                mu_pad[0:num_x]=mu[i][:]
                conv_2=np.fft.ifft(np.fft.fft(mu_pad)*fft_h_pad)
                conv=conv_2[num_x-1:2*num_x-1]
                convolution=np.real(conv)
            elif problem=='ex_73' or problem=='flocking_Pontryagin' or problem=='flocking_weak':
                X_mean=np.dot(x_grid,mu[i])
            elif problem=='ex_1' or problem=='trader_Pontryagin':
                Y_mean=np.dot(u_old[i],mu[i])
            elif problem=='trader_weak':
                Z_mean=np.dot(v_old[i],mu[i])
            for j in range(num_x):
                x_down = x_grid[j] + b(i, j, mu, u_old, v_old,X_mean,Y_mean,Z_mean,convolution) * delta_t - sigma * sqrt_delta_t

                x_up = x_grid[j] + b(i, j, mu, u_old, v_old,X_mean,Y_mean,Z_mean,convolution) * delta_t + sigma * sqrt_delta_t

                j_down = pi(x_down)

                j_up = pi(x_up)

#                if i==num_t-2:
#
#                    u[i][j] = (g(x_down) + g(x_up))/2.0 + delta_t*f(i,j,mu,u_old,v_old,X_mean,Y_mean,Z_mean,convolution)
#
#                    v[i][j] = 1.0/sqrt_delta_t * (g(x_up) - g(x_down))
#
#                else:
                if linear_int:
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
                else:

                    u_up = u[i+1][j_up]
                    u_down = u[i+1][j_down]

                u[i][j] = (u_down + u_up)/2.0 + delta_t*f(i,j,mu,u_old,v_old,X_mean,Y_mean,Z_mean,convolution)

                v[i][j] = 1.0/sqrt_delta_t * (u_up - u_down)

    return [u,v]


def pi_lv(x,x_grid_lv):
    num_x_lv=len(x_grid_lv)
    x_min_lv=x_grid_lv[0]

    if periodic_2_pi:
        x=x%(2*np.pi)
        index=int(round(((x-x_min_lv)/delta_x)))
        index=index%num_x_lv
    else:
        index=int(round(((x-x_min_lv)/delta_x)))
        index=min(num_x_lv-1,index)
        index=max(0,index)
    return index
    

# Suppose that x_grid_1 is included in x_grid_2 or 2 included in 1
# Define the law given by mu_1 on x_grid_2
def transform_grid(x_grid_1,mu_1,x_grid_2):
    num_x_1=len(x_grid_1)
    num_x_2=len(x_grid_2)
    mu_2=np.zeros(num_x_2)
    
    if num_x_1<num_x_2:
        diff_0=int((x_grid_1[0]-x_grid_2[0])/delta_x)
        mu_2[diff_0:diff_0+num_x_1]=mu_1
        
    else:
        diff_0=int((x_grid_2[0]-x_grid_1[0])/delta_x)
        mu_2=mu_1[diff_0:diff_0+num_x_2]
        
    return mu_2


def forward_lv(u,v,x_grid_lv,mu_0):
    num_x_lv=len(x_grid_lv)

    mu=np.zeros((num_t,num_x_lv))
    mu[0,:]=mu_0
    
    for i in range(num_t-1): #t_i
        convolution=0
        X_mean=0
        Y_mean=0
        Z_mean=0
        if problem=='jetlag_weak' or problem=='jetlag_Pontryagin':
            convolution=np.zeros((num_x_lv))
            mu_pad=np.zeros((3*num_x_lv-2))
            mu_pad[0:num_x_lv]=mu[i][:]
            conv_2=np.fft.ifft(np.fft.fft(mu_pad)*fft_h_pad)
            conv=conv_2[num_x_lv-1:2*num_x_lv-1]
            convolution=np.real(conv)
        elif problem=='ex_73' or problem=='flocking_Pontryagin' or problem=='flocking_weak':
            X_mean=np.dot(x_grid_lv,mu[i])
        elif problem=='ex_1' or problem=='trader_Pontryagin':
            Y_mean=np.dot(u[i],mu[i])
        elif problem=='trader_weak':
            Z_mean=np.dot(v[i],mu[i])
        for j in range(num_x_lv): #x_j
            
            low=x_grid_lv[j]+b(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution)*delta_t-sigma*sqrt_delta_t
            low_index=pi_lv(low,x_grid_lv)
            mu[i+1,low_index]+=mu[i,j]*0.5
                        
            up=x_grid_lv[j]+b(i,j,mu,u,v,X_mean,Y_mean,Z_mean,convolution)*delta_t+sigma*sqrt_delta_t
            up_index=pi_lv(up,x_grid_lv)
            mu[i+1,up_index]+=mu[i,j]*0.5
    return mu


def backward_lv(mu,u_old,v_old,x_grid_lv,Y_terminal):     
    num_x_lv=len(x_grid_lv)
    u=np.zeros((num_t,num_x_lv))
    v=np.zeros((num_t,num_x_lv))
    
    u[num_t-1,:]=Y_terminal
    v[num_t-1,:]=v_old[num_t-1,:]
    
    for i in reversed(range(num_t-1)):
        convolution=0
        X_mean=0
        Y_mean=0
        Z_mean=0
        if problem=='jetlag_weak' or problem=='jetlag_Pontryagin':
            convolution=np.zeros((num_x_lv))
            mu_pad=np.zeros((3*num_x_lv-2))
            mu_pad[0:num_x_lv]=mu[i][:]
            conv_2=np.fft.ifft(np.fft.fft(mu_pad)*fft_h_pad)
            conv=conv_2[num_x_lv-1:2*num_x_lv-1]
            convolution=np.real(conv)
        elif problem=='ex_73' or problem=='flocking_Pontryagin' or problem=='flocking_weak':
            X_mean=np.dot(x_grid_lv,mu[i])
        elif problem=='ex_1' or problem=='trader_Pontryagin':
            Y_mean=np.dot(u_old[i],mu[i])
        elif problem=='trader_weak':
            Z_mean=np.dot(v_old[i],mu[i])
        for j in range(num_x_lv):
            
            x_down=x_grid_lv[j]+b(i,j,mu,u_old,v_old,X_mean,Y_mean,Z_mean,convolution)*delta_t-sigma*sqrt_delta_t
            j_down=pi_lv(x_down,x_grid_lv)
            
            x_up=x_grid_lv[j]+b(i,j,mu,u_old,v_old,X_mean,Y_mean,Z_mean,convolution)*delta_t+sigma*sqrt_delta_t
            j_up=pi_lv(x_up,x_grid_lv)

            if linear_int:
                if x_down>x_grid_lv[j_down]:
                    if j_down<num_x_lv-1:
                        u_down= lin_int(x_grid_lv[j_down],x_grid_lv[j_down+1],u[i+1][j_down],u[i+1][j_down+1],x_down)
                    else:
                        u_down=u[i+1][j_down]
                else:
                    if j_down>0:
                        u_down= lin_int(x_grid_lv[j_down],x_grid_lv[j_down-1],u[i+1][j_down],u[i+1][j_down-1],x_down)
                    else:
                        u_down=u[i+1][j_down]
                
                if x_up>x_grid_lv[j_up]:
                    if j_up<num_x_lv-1:
                        u_up= lin_int(x_grid_lv[j_up],x_grid_lv[j_up+1],u[i+1][j_up],u[i+1][j_up+1],x_up)
                    else:
                        u_up=u[i+1][j_up]
                else:
                    if j_up>0:
                        u_up= lin_int(x_grid_lv[j_up],x_grid_lv[j_up-1],u[i+1][j_up],u[i+1][j_up-1],x_up)
                    else:
                        u_up=u[i+1][j_up]
            else:
            
                u_up = u[i+1][j_up]
                u_down = u[i+1][j_down]
            
            u[i][j] = (u_down + u_up)/2.0 + delta_t*f(i,j,mu,u_old,v_old,X_mean,Y_mean,Z_mean,convolution)
            
            v[i][j] = 1.0/sqrt_delta_t*(u_up-u_down)

    return [u,v]

def solver_grid(level,mu_0,X_grids):
    #print('level: ',level)
#    num_x=len(mu)
    if level==num_level:
        #print('break')
        Y_terminal=g(X_grids[level-1])
        return Y_terminal
    
    num_x_lv=len(X_grids[level])
    u=np.zeros((num_t,num_x_lv))
    v=np.zeros((num_t,num_x_lv))
    mu=np.zeros((num_t,num_x_lv))
    for k in range(num_t):
        mu[k]=mu_0
    if level==num_level-1:
        Y_terminal=g(X_grids[level])
    else:
        Y_terminal=np.zeros(num_x_lv)


    #Y_terminal=np.zeros(num_x_lv)
    
    for j in range(J_1):
        [u,v]=backward_lv(mu,u,v,X_grids[level],Y_terminal)
        mu=forward_lv(u,v,X_grids[level],mu_0)
    u=np.zeros((num_t,num_x_lv))
    v=np.zeros((num_t,num_x_lv))
    
    if level==num_level-1:
        Y_terminal=g(X_grids[level])
    else:
        Y_terminal=np.zeros(num_x_lv)

    if level==0:
        all_Y_0_values=np.zeros((num_keep))
        index=0
    
        
    for j in range(J_2):
        mu_next=mu[num_t-1,:]
        if level<num_level-1:
            mu_next=transform_grid(X_grids[level],mu_next,X_grids[level+1])
        #print('loop in level: ',level)
        #print('j=',j)
        #print(g(mu_next))
        Y_terminal=solver_grid(level+1,mu_next,X_grids)
        #print(Y_terminal)
        #print('back in level: ',level)
        if level<num_level-1:
            Y_terminal=transform_grid(X_grids[level+1],Y_terminal,X_grids[level])
        for j2 in range(J_1):
            [u,v]=backward_lv(mu,u,v,X_grids[level],Y_terminal)
            mu=forward_lv(u,v,X_grids[level],mu_0)
        thing1=u
        if level==0 and j>J_2-num_keep-1:
            all_Y_0_values[index]=np.dot(u[0],mu_0)
            index+=1
            
    if level==0:
        return [u[0,:],mu,u,v,all_Y_0_values,thing1]
    return u[0,:]



if __name__ == '__main__':
    start_time=time.time()


    global problem

    problem='ex_1'
    #possible values in order of appearance: jetlag(_Pontryagin,_weak),
    #trader(_Pontryagin,_weak,_weak_truncation), ex_1, ex_72, ex_73, flocking(_Pontryagin,_weak)

    global execution
    execution='continuation_in_time'
    # possible values in order of appearance:
    # ordinary, changing_sigma, changing_rho, adaptive, solution_trader,
    #true_start, continuation_in_time
    
    global linear_int
    linear_int=False

    global b
    global f
    global g
    global periodic_2_pi #if True, use periodic domain [0,2pi)
    global J #number of Picard iterations
    global num_keep #number of last Picard iterations to print and save
    global T #finite time horizon
    global num_t #number of time points (one more than the number of time steps)
    
    global delta_t
    global sqrt_delta_t
    global t_grid
    global delta_x
    global x_min #used to set the size of x_grid
    global x_max #used to set the size of x_grid
    global x_grid
    
    global sigma #diffusion coefficient
    
    #parameters that are specific to certain problems
    global rho
    global a
    global R
    global K
    global F
    global omega_0
    global omega_S
    global p
    global c_x
    global h_bar
    global c_g
    global ftt_h_pad

    #set the above variables depending on 'problem'
    if problem =='jetlag_Pontryagin':
        b=b_jet_lag_Pontryagin
        f=f_jet_lag_Pontryagin
        g=g_jet_lag
        periodic_2_pi=True
        J=25
        num_keep=5
        T=24.0*1
        #num_t=int(T)*5+1
        num_t=697
        delta_t=T/(num_t-1)
        t_grid=np.linspace(0,T,num_t)
        #delta_x=delta_t**2
        #num_x=int((2*np.pi)/(delta_x))+1
        num_x=158
        delta_x=2*np.pi/num_x
        x_grid=np.linspace(0,2*np.pi-delta_x,num_x)
        
        x_min=x_grid[0]
        x_max=x_grid[num_x-1]
        
        # Varible Jet Lag
        R=1
        K=0.01
        F=0.01
        omega_0=2*np.pi/24.5
        omega_S=2*np.pi/24
        p=(9.0/12.0)*np.pi
        sigma=0.1
        h_array=get_h_jet_lag_Pontryagin()
        h_pad=np.zeros((3*num_x-2))
        h_pad[0:2*num_x-1]=h_array
        fft_h_pad=np.fft.fft(h_pad)
    elif problem =='jetlag_weak':
        sigma=0.1
        b=b_jet_lag_weak
        f=f_jet_lag_weak
        g=g_jet_lag
        periodic_2_pi=True
        J=25
        num_keep=5
        T=24.0*1
        #num_t=int(T)*5+1
        num_t=697
        delta_t=T/(num_t-1)
        t_grid=np.linspace(0,T,num_t)
        #delta_x=delta_t**2
        #num_x=int((2*np.pi)/(delta_x))+1
        num_x=158
        delta_x=2*np.pi/num_x
        x_grid=np.linspace(0,2*np.pi-delta_x,num_x)
        
        x_min=x_grid[0]
        x_max=x_grid[num_x-1]
        
        # Varible Jet Lag
        R=1
        K=0.01
        F=0.01
        omega_0=2*np.pi/24.5
        omega_S=2*np.pi/24
        p=(9.0/12.0)*np.pi
        h_array=get_h_jet_lag_weak()
        h_pad=np.zeros((3*num_x-2))
        h_pad[0:2*num_x-1]=h_array
        fft_h_pad=np.fft.fft(h_pad)
    elif problem=='ex_1':
        b=b_example_1
        f=f_example_1
        g=g_example_1
        periodic_2_pi=False
        J=25
        num_keep=5
        T=1
        num_t=12
        delta_t=T/(num_t-1)
        t_grid=np.linspace(0,T,num_t)
        delta_x=delta_t**(2)
        x_min=-1
        x_max=5
        num_x=int((x_max-x_min)/delta_x+1)
        x_grid=np.linspace(x_min,x_max,num_x)
        sigma=1
        rho=0.1
        a=0.25
    elif problem=='ex_72':
        b=b_example_72
        f=f_example_72
        g=g_example_72
        periodic_2_pi=False
        J=10
        num_keep=5
        T=1
        num_t=12
        delta_t=T/(num_t-1)
        t_grid=np.linspace(0,T,num_t)
        delta_x=delta_t**(2)
        x_min=-3
        x_max=3
        num_x=int((x_max-x_min)/delta_x)+1
        x_grid=np.linspace(x_min,x_max,num_x)
        sigma=1
        rho=2
    elif problem=='ex_73':
        b=b_example_73
        f=f_example_73
        g=g_example_73
        periodic_2_pi=False
        J=25
        num_keep=5
        T=1
        num_t=12
        delta_t=T/(num_t-1)
        t_grid=np.linspace(0,T,num_t)
        delta_x=delta_t**(2)
        x_min=-1
        x_max=5
        num_x=int((x_max-x_min)/delta_x)+1
        x_grid=np.linspace(x_min,x_max,num_x)
        sigma=1
        rho=1
    elif problem=='trader_Pontryagin':
        sigma=0.7
        rho=1
        c_x=4
        h_bar=2
        c_g=0.3
        # sigma=0.7
        # rho=0.03
        # c_x=1
        # h_bar=10
        # c_g=0.3
        # sigma=0.7
        # rho=0.07
        # c_x=0.1
        # h_bar=0.01
        # c_g=1
        b=b_trader_Pontryagin
        f=f_trader_Pontryagin
        g=g_trader_Pontryagin
        periodic_2_pi=False
        J=25
        num_keep=5
        T=1
        num_t=12
        delta_t=(T-0.06)/(num_t-1)
        t_grid=np.linspace(0.06,T,num_t)
        delta_x=delta_t**(2)
        x_min=-2
        x_max=4
        num_x=int((x_max-x_min)/delta_x+1)
        x_grid=np.linspace(x_min,x_max,num_x)
# Variable trader
# convergence for rho=0.1
    elif problem=='trader_weak':
        sigma=0.7
        rho=1
        c_x=4
        h_bar=2
        c_g=0.3
        b=b_trader_weak
        f=f_trader_weak
        g=g_trader_weak
        periodic_2_pi=False
        J=25
        num_keep=5
        T=1
        num_t=20
        delta_t=(T-0.06)/(num_t-1)
        t_grid=np.linspace(0.06,T,num_t)
        delta_x=delta_t**(2)
        x_min=-2
        x_max=4
        num_x=int((x_max-x_min)/delta_x+1)
        x_grid=np.linspace(x_min,x_max,num_x)

    elif problem=='trader_weak_truncation':
        sigma=0.7
        rho=1
        c_x=2
        h_bar=2
        c_g=0.3
        b=b_trader_weak_trunc
        f=f_trader_weak_trunc
        g=g_trader_weak_trunc
        periodic_2_pi=False
        J=25
        num_keep=5
        T=1
        num_t=20
        delta_t=(T-0.06)/(num_t-1)
        t_grid=np.linspace(0.06,T,num_t)
        delta_x=delta_t**(2)
        x_min=-2
        x_max=4
        num_x=int((x_max-x_min)/delta_x+1)
        x_grid=np.linspace(x_min,x_max,num_x)
        global bounds
        bounds=np.load('./Data/trader/value_y_Pont_to_trunc_z_weak.npy')
    # Variable trader


    elif problem=='trader_solution':
        sigma=0.7
        rho=1
        c_x=0.7
        h_bar=2
        c_g=0.3
        b=b_trader_solution
        f=f_trader_solution
        g=g_trader_solution
        periodic_2_pi=False
        J=25
        num_keep=5
        T=1
        num_t=20
        delta_t=(T-0.06)/(num_t-1)
        t_grid=np.linspace(0.06,T,num_t)
        delta_x=delta_t**(2)
        x_min=-2
        x_max=4
        num_x=int((x_max-x_min)/delta_x+1)
        x_grid=np.linspace(x_min,x_max,num_x)
        A=-h_bar*rho*0.5
        B=rho
        C=c_x
        R=A**2+B*C

        delta_up=-A+np.sqrt(R)
        delta_down=-A-np.sqrt(R)
        delta_delta=delta_up-delta_down

        eta_bar=np.zeros((1,num_t))
        eta=np.zeros((1,num_t))
        ratio=np.sqrt(c_x*rho)
        ratio2=ratio/rho
        for t in range(num_t):
            eta_bar[0,t]=-C*(np.exp(delta_delta*(T-t))-1)-c_g*(delta_up*np.exp(delta_delta*(T-t))-delta_down)/(((delta_down*np.exp(delta_delta*(T-t))-delta_up))-c_g*B*(np.exp(delta_delta*(T-t))-1))
            eta[0,t]=-ratio2*(ratio2-c_g-(ratio2+c_g)*np.exp(2*ratio*(T-t)))/(ratio2-c_g+(ratio2+c_g)*np.exp(2*ratio*(T-t)))


    elif problem=='flocking_Pontryagin':
        b=b_flocking_Pontryagin
        f=f_flocking_Pontryagin
        g=g_flocking
        periodic_2_pi=False
        J=25
        num_keep=5
        T=1
        num_t=20
        delta_t=T/(num_t-1)
        t_grid=np.linspace(0,T,num_t)
        delta_x=delta_t**(2)
        x_min=-3
        x_max=3
        num_x=int((x_max-x_min)/delta_x+1)
        x_grid=np.linspace(x_min,x_max,num_x)
        sigma=1
        
    elif problem=='flocking_weak':
        b=b_flocking_weak
        f=f_flocking_weak
        g=g_flocking
        periodic_2_pi=False
        J=25
        num_keep=5
        T=1
        num_t=20
        delta_t=T/(num_t-1)
        t_grid=np.linspace(0,T,num_t)
        delta_x=delta_t**(2)
        x_min=-3
        x_max=3
        num_x=int((x_max-x_min)/delta_x+1)
        x_grid=np.linspace(x_min,x_max,num_x)
        sigma=1
    sqrt_delta_t=np.sqrt(delta_t)

    if execution=='ordinary':

        mu_0=np.zeros((num_x))
        if periodic_2_pi:
            mu_0=np.load('mu_initial_reference_set_158.npy')
            #mu_0=scipy.io.loadmat('mu_initial_reference_set_158.mat')['mu_initial']
            #mu_0=[mu_0[int(i/6)] for i in range(num_x)]
            #mu_0=mu_0/np.sum(mu_0)
            #mu_0[0]=1
        else:
            mu_0[int(num_x/2)]=1.0
        
        mu=np.zeros((num_t,num_x))
        for k in range(num_t):
            mu[k]=mu_0
        u=np.zeros((num_t,num_x))
        v=np.zeros((num_t,num_x))
        index2=0
        all_Y_0_values=np.zeros((1,num_keep))
        
        for j in range(1):
            [u,v]=backward(mu,u,v)
            mu=forward(u,v,mu_0)
        u=np.zeros((num_t,num_x))
        v=np.zeros((num_t,num_x))
        
        
        for j in range(J):
            [u,v]=backward(mu,u,v)            
            mu=forward(u,v,mu_0)
            thing2=u
            if j>J-num_keep-1:
                all_Y_0_values[0][index2]=np.dot(u[0],mu[0])
                index2+=1
        print all_Y_0_values[0]



        if problem=='trader_Pontryagin':
            bounds=np.zeros((2,num_t))
            for t in range(num_t):
                bounds[0,t]=min(u[t])
                bounds[1,t]=max(u[t])
            np.save('./Data/trader/mu_Pont_t20.npy',mu)
            np.save('./Data/trader/value_y_Pont_to_trunc_z_weak.npy',bounds)
        elif problem=='trader_weak':
            np.save('./Data/trader/mu_weak_t20.npy',mu)
        elif problem=='trader_weak_truncation':
            np.save('./Data/trader/mu_weak_trunc_t20.npy',mu)




        ############## evaluating mu_u, mu_v

        # mu_u = np.zeros((num_t,num_x))
        # mu_v = np.zeros((num_t,num_x))
        #
        # mu_u[num_t-1,:] = mu[num_t-1,:]
        # mu_v[num_t-1,:] = mu[num_t-1,:]
        #
        # for i in reversed(range(num_t-1)):
        #     for j in range(num_x):
        #
        #         j_down = pi(x_grid[j] + b(i, j, mu, u, v) * delta_t - sigma * sqrt_delta_t
        #         j_up = pi(x_grid[j] + b(i, j, mu, u, v) * delta_t + sigma * sqrt_delta_t
        #         mu_u[i][j] = mu[i+1][j_down]+mu[i+1][j_up]
        #         mu_v[i][j] = mu[i+1][j_down]+mu[i+1][j_up]
        # test=np.zeros((num_t,num_x))
        #print(mu_u[num_t//2-1][:])





    ###############
    elif execution=='changing_sigma':
        num_sigma=10
        sigma_values=np.linspace(0.5,10,num_sigma)
        all_Y_0_values=np.zeros((num_sigma,num_keep))
        for index in range(num_sigma):
            index2=0
            sigma=sigma_values[index]
            mu_0=np.zeros((num_x))
            if periodic_2_pi:
                mu_0=scipy.io.loadmat('mu_initial_reference_set_158.mat')['mu_initial']
            #mu_0[0]=1
            else:
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
                    all_Y_0_values[index][index2]=np.dot(u[0],mu[0])
                    index2+=1
        print all_Y_0_values[index]
    
    elif execution=='changing_rho':
        num_rho=15
        rho_values=np.linspace(2.5,5,num_rho)

        all_Y_0_values=np.zeros((num_rho,num_keep))
        value_x=num_keep*[1]
        plot_cx = plt.figure()
        for index in range(num_rho):
            index2=0
            #rho=rho_values[index]
            rho=rho_values[index]
            
            
            mu_0=np.zeros((num_x))
            if periodic_2_pi:
                mu_0=scipy.io.loadmat('mu_initial_reference_set_158.mat')['mu_initial']
            #mu_0[0]=1
            else:
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
                    all_Y_0_values[index][index2]=np.dot(u[0],mu[0])
                    index2+=1
            print all_Y_0_values[index]

            plot_cx=plt.plot(np.multiply(rho,value_x),all_Y_0_values[index],'o')
        plt.title('$sigma = 0.7$, $rho \in [1,10]$, $c_x = 2, $h_bar=2$, $c_g=0.3$')
        plt.show()

        #plt.savefig('./Data/trader/grid_trader_pontryagin_changing_cx.eps')
        #np.save('./Data/trader/grid_trader_pontryagin_rho_larger_x_domain.npy',all_Y_0_values)



    elif execution=='adaptive':
        x_min_0=-3
        x_max_0=3
        num_rho=20
        rho_values=np.linspace(0.5,10.0,num_rho)
        num_t_0=12
        delta_t_0=T/(num_t_0-1)
        delta_x=delta_t_0**2
        
        all_Y_0_values=np.zeros((num_rho,num_keep))
        #all_Y_0_values=np.zeros((num_sigma,num_keep))
        for index in range(num_rho):
            #for index in range(num_sigma):
            index2=0
            rho=rho_values[index]
            
#            num_t=int(num_t_0*(rho/rho_values[0]))
            num_t=12
            delta_t= T/num_t
            x_min_goal=x_min_0*math.sqrt(rho/rho_values[0])*2
            x_max_goal=x_max_0*math.sqrt(rho/rho_values[0])*2
            x_center=(x_min_goal+x_max_goal)/2.0
            num_x=int((x_max_goal-x_min_goal)/(delta_x))+1
            if num_x%2==0:
                num_x+=1
            x_grid=np.linspace(x_center-(num_x-1)/2*delta_x,x_center+(num_x-1)/2*delta_x,num_x)
            x_min=x_grid[0]
            x_max=x_grid[num_x-1]
            
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
                    all_Y_0_values[index][index2]=np.dot(u[0],mu[0])
                    index2+=1
            print('rho=',rho)
            print(all_Y_0_values[index])

    elif execution=='trader_solution':
        mu_0=np.zeros((num_x))
        mu_0[int(num_x/2)]=1.0

        mu=np.zeros((num_t,num_x))
        for k in range(num_t):
            mu[k]=mu_0
        u=np.zeros((num_t,num_x))
        v=np.zeros((num_t,num_x))
        mu=forward(u,v,mu_0)

        np.save('./Data/trader/trader_solution.npy',mu)
    elif execution=='true_start': # only for some problems

        if periodic_2_pi:
            mu=np.load('mu_reference_set_158.npy')*delta_x
            u=np.load('u_reference_set_158.npy')
        elif problem=='trader_weak' or problem=='trader_Pontryagin':
            mu=true_solution=np.load('./Data/trader/trader_solution.npy')
        mu_0=mu[0]
        u=np.zeros((num_t,num_x))
        v=np.zeros((num_t,num_x))
        index2=0
        all_Y_0_values=np.zeros((1,num_keep))
        for j in range(J):
            [u,v]=backward(mu,u,v)
            mu=forward(u,v,mu_0)
            if j>J-num_keep-1:
                all_Y_0_values[0][index2]=np.dot(u[0],mu[0])
                index2+=1
        print all_Y_0_values[0]




        #np.save('./Data/trader/mu_trader_true_start_t20.npy',mu)
            
            
    elif execution=='continuation_in_time':
        global J_1,J_2
        J_1=1
        J_2=25
        global num_level
        
        num_level=2

        num_t=12
        delta_t=T/num_level/(num_t-1)
        sqrt_delta_t=math.sqrt(delta_t)
        delta_x=(delta_t*num_level)**2
        #delta_x=delta_t**(2)
        x_min=-1
        x_max=5
        num_x=int((x_max-x_min)/delta_x)+1
        x_grid=np.linspace(x_min,x_max,num_x)

        # x_grid for each level, should be an increasing sequence
        X_grids=[]
        for i in range(num_level):
            X_grids.append(x_grid)
        mu_0=np.zeros((num_x))
        mu_0[int(num_x/2)]=1.0

              

#        x_0=2
#        # x_grid for each level, should be an increasing sequence
#        X_grids=[]
#        num_x_vec=np.zeros(num_level)
#        b_m=rho
#        for i in range(num_level):
#            T_lv=T/num_level*(i+1)
#            x_min=x_0-b_m*T_lv-3*sigma*math.sqrt(T_lv)
#            x_max=x_0+b_m*T_lv+3*sigma*math.sqrt(T_lv)
#            num_x=int((x_max-x_min)/delta_x)+1
#            num_x_vec[i]=num_x
#            x_grid=np.linspace(x_min,x_max,num_x)
#            X_grids.append(x_grid)
#        mu_0=np.zeros(int(num_x_vec[0]))
#        mu_0[int(num_x_vec[0]/2)]=1.0

        
        [u_0,mu,u,v,all_Y_0_values,thing1]=solver_grid(0,mu_0,X_grids)
        print(all_Y_0_values)
    end_time=time.time()
    print('Time elapsted:',end_time-start_time)
