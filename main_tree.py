#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 09:24:22 2017
@author: Andrea Angiuli, Christy Graves, Houzhi Li


This code implements the continuation in time tree algorithm described in
Chassagneux, Crisan, Delarue to solve FBSDEs of McKean Vlasov type.

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

#Define functions b, f, and g for a variety of problems:

def b_example_1(i,j,X,Y,Z,X_initial_probs):
    num_initial=len(X[0])
    Y_mean=0
    num_per_initial=len(Y[i])/num_initial
    for k in range(len(Y[i])):
        index=int(math.floor(k/num_per_initial))
        Y_mean+=Y[i][k]*X_initial_probs[index]/num_per_initial
    return -rho*Y_mean

def f_example_1(i,j,X,Y,Z,X_initial_probs):
    return a*Y[i][j]

def g_example_1(index,xi_vals,xi_probs):
    x=xi_vals[index]
    return x

def b_example_72(i,j,X,Y,Z,X_initial_probs):
    return rho*np.cos(Y[i][j])

def f_example_72(i,j,X,Y,Z,X_initial_probs):
    return 0

def g_example_72(index,xi_vals,xi_probs):
    x=xi_vals[index]
    return np.sin(x)

def b_example_73(i,j,X,Y,Z,X_initial_probs):
    return -rho*Y[i][j]

def f_example_73(i,j,X,Y,Z,X_initial_probs):
    num_initial=len(X[0])
    X_mean=0
    num_per_initial=len(X[i])/num_initial
    for k in range(len(X[i])):
        index=int(math.floor(k/num_per_initial))
        X_mean+=X[i][k]*X_initial_probs[index]/num_per_initial
    return -np.arctan(X_mean)

def g_example_73(index,xi_vals,xi_probs):
    x=xi_vals[index]
    return np.arctan(x)

def b_example_73_E(i,j,X,Y,Z,X_initial_probs):
    num_initial=len(X[0])
    Y_mean=0
    num_per_initial=len(Y[i])/num_initial
    for k in range(len(Y[i])):
        index=int(math.floor(k/num_per_initial))
        Y_mean+=Y[i][k]*X_initial_probs[index]/num_per_initial
    return -rho*Y_mean

def g_example_73_E(index,xi_vals,xi_probs):
    X_mean=np.dot(xi_vals,xi_probs)
    return np.arctan(X_mean)

def b_jet_lag_weak(i,j,X,Y,Z,X_initial_probs):
    return omega_0-omega_S-1.0/(R*sigma)*Z[i][j]

def f_jet_lag_weak(i,j,X,Y,Z,X_initial_probs):
    value1=1.0/sigma*(omega_0-omega_S)*Z[i][j]-1.0/(2*R*sigma**2)*(Z[i][j])**2
    num_initial=len(X[0])
    num_per_initial=len(X[i])/num_initial
    X_probs=np.zeros((len(X[i])))
    for k in range(len(X[i])):
        index=int(math.floor(k/num_per_initial))
        X_probs[k]=X_initial_probs[index]/num_per_initial
    c_bar=np.dot(0.5*(np.sin((X[i][j]-X[i])/2.0))**2,X_probs)
    value2=K*c_bar
    c_sun=0.5*(np.sin((X[i][j]-p)/2.0))**2
    value3=F*c_sun
    return value1+value2+value3

def g_jet_lag(index,xi_vals,xi_probs):
    return 0

def b_jet_lag_Pontryagin(i,j,X,Y,Z,X_initial_probs):
    return omega_0-omega_S-1.0/R*Y[i][j]

def f_jet_lag_Pontryagin(i,j,X,Y,Z,X_initial_probs):
    num_initial=len(X[0])
    num_per_initial=len(X[i])/num_initial
    X_probs=np.zeros((len(X[i])))
    for k in range(len(X[i])):
        index=int(math.floor(k/num_per_initial))
        X_probs[k]=X_initial_probs[index]/num_per_initial
    partial_c_bar=np.dot(0.5*np.sin((X[i][j]-X[i])/2.0)*np.cos((X[i][j]-X[i])/2.0),X_probs)
    value1=K*partial_c_bar
    partial_c_sun=0.5*np.sin((X[i][j]-p)/2.0)*np.cos((X[i][j]-p)/2.0)
    value2=F*partial_c_sun
    return value1+value2

def b_trader_Pontryagin(i,j,X,Y,Z,X_initial_probs):
    return -rho*Y[i][j] #rho=1/c_alpha

def f_trader_Pontryagin(i,j,X,Y,Z,X_initial_probs):
    num_initial=len(X[0])
    Y_mean=0
    num_per_initial=len(Y[i])/num_initial
    for k in range(len(Y[i])):
        index=int(math.floor(k/num_per_initial))
        Y_mean+=Y[i][k]*X_initial_probs[index]/num_per_initial
    return c_x*X[i][j]+h_bar*rho*Y_mean

def g_trader_Pontryagin(index,xi_vals,xi_probs):
    x=xi_vals[index]
    return c_g *x

def b_trader_weak(i,j,X,Y,Z,X_initial_probs):
    return -rho*Z[i][j]/sigma #rho=1/c_alpha

def f_trader_weak(i,j,X,Y,Z,X_initial_probs):
    num_initial=len(X[0])
    Z_mean=0
    num_per_initial=len(Z[i])/num_initial
    for k in range(len(Z[i])):
        index=int(math.floor(k/num_per_initial))
        Z_mean+=Z[i][k]*X_initial_probs[index]/num_per_initial
    return 0.5*c_x*X[i][j]**2+X[i][j]*h_bar*rho*Z_mean/sigma-rho*0.5*Z[i][j]**2/sigma**2

def g_trader_weak(index,xi_vals,xi_probs):
    x=xi_vals[index]
    return c_g*0.5*x**2
    
def b_flocking_Pontryagin(i,j,X,Y,Z,X_initial_probs):
    return -Y[i][j]

def f_flocking_Pontryagin(i,j,X,Y,Z,X_initial_probs):
    num_initial=len(X[0])
    X_mean=0
    num_per_initial=len(X[i])/num_initial
    for k in range(len(X[i])):
        index=int(math.floor(k/num_per_initial))
        X_mean+=X[i][k]*X_initial_probs[index]/num_per_initial
    return X[i][j]-X_mean

def g_flocking(index,xi_vals,xi_probs):
    return 0
    
def b_flocking_weak(i,j,X,Y,Z,X_initial_probs):
    return -Z[i][j]/sigma

def f_flocking_weak(i,j,X,Y,Z,X_initial_probs):
    num_initial=len(X[0])
    X_mean=0
    num_per_initial=len(X[i])/num_initial
    for k in range(len(X[i])):
        index=int(math.floor(k/num_per_initial))
        X_mean+=X[i][k]*X_initial_probs[index]/num_per_initial
    return -1.0/(2*sigma**2)*(Z[i][j])**2+0.5*(X[i][j]-X_mean)**2

#This function is essentially solver_bar, except it is used for the
#continuation to initialize with the previous solution. It is written to be
#used with only one level.
def continuation_solver_bar(X_ini,X_initial_probs,Y_ini,Z_ini):
    
    num_initial=len(X_ini[0])
    Y_0_values=np.zeros((num_keep))
    index=0
    X=X_ini
    Y=Y_ini
    Z=Z_ini
    x_vals=np.zeros(num_initial*2**(num_t_fine-1))
    x_probs=[]
    for j in range(num_initial):
        row1=X_initial_probs[j]*np.ones(2**(num_t_fine-1))/(2**(num_t_fine-1))
        x_probs=np.concatenate((x_probs,row1))
    #    print(x_probs)
    for k in range(J):
        for j in range(num_initial*2**(num_t_fine-1)):
            Y[num_t_fine-1][j]=g(j,X[num_t_fine-1],x_probs)
            
            for index2 in range(J_solver_bar):
                for i in reversed(range(num_t_fine-1)):
                    for j in range(num_initial*2**i):
                        #                temp_Y=(Y[i+1][2*j]+Y[i+1][2*j+1]+delta_t_fine*f(i+1,2*j,X,Y,Z,X_initial_probs)+delta_t_fine*f(i+1,2*j+1,X,Y,Z,X_initial_probs))/2.0
                        temp_Y=(Y[i+1][2*j]+Y[i+1][2*j+1])/2.0+delta_t_fine*f(i,j,X,Y,Z,X_initial_probs)
                        #temp_=delta_t_fine*f(i,j,X,Y,Z,X_initial_probs)
                        Y[i][j]=temp_Y
                        Z[i][j]=delta_W/delta_t_fine*(Y[i+1][2*j]-Y[i+1][2*j+1])/2.0
                #                print(k,i,j,temp_)
                for i in range(num_t_fine-1):
                    for j in range(num_initial*2**i):
                        X[i+1][2*j]=X[i][j]+delta_t_fine*b(i,j,X,Y,Z,X_initial_probs)+sigma*delta_W
                        X[i+1][2*j+1]=X[i][j]+delta_t_fine*b(i,j,X,Y,Z,X_initial_probs)-sigma*delta_W

        if k>J-num_keep-1:
            Y_0_values[index]=Y[0][0]
            index+=1
    return [X,Y,Z,Y_0_values]

#solver_bar as defined in Chassagneux, Crisan, Delarue
def solver_bar(X,Y_terminal,X_initial_probs,Y_old):
    num_initial=len(X[0])
    Y=[]
    Z=[]
    for i in range(num_t_fine):
        row2=np.zeros((num_initial*2**i))
        row3=np.zeros((num_initial*2**i))
        Y.append(row2)
        Z.append(row3)
    
    for j in range(len(Y[num_t_fine-1])):
        Y[num_t_fine-1][j]=Y_terminal[j]
    #Y[num_t_fine-1,:]=Y_terminal

    for k in range(J_solver_bar):
        if k>0:
            Y_old=Y
        
        for n in range(num_t_fine-1):
            i=num_t_fine-2-n
            for j in range(num_initial*2**i):
                #Y[i][j]=(Y[i+1][2*j]+Y[i+1][2*j+1]+delta_t_fine*f(i+1,2*j,X,Y,Z,X_initial_probs)+delta_t_fine*f(i+1,2*j+1,X,Y,Z,X_initial_probs))/2.0
                Y[i][j]=(Y[i+1][2*j]+Y[i+1][2*j+1])/2.0+delta_t_fine*f(i,j,X,Y_old,Z,X_initial_probs)
                Z[i][j]=delta_W/delta_t_fine*(Y[i+1][2*j]-Y[i+1][2*j+1])/2.0

    for i in range(num_t_fine-1):
        for j in range(num_initial*2**i):
            X[i+1][2*j]=X[i][j]+delta_t_fine*b(i,j,X,Y,Z,X_initial_probs)+sigma*delta_W
            X[i+1][2*j+1]=X[i][j]+delta_t_fine*b(i,j,X,Y,Z,X_initial_probs)-sigma*delta_W
            if periodic_2_pi:
                X[i+1][2*j]=X[i+1][2*j]%(2*np.pi)
                X[i+1][2*j+1]=X[i+1][2*j+1]%(2*np.pi)
    return [X,Y,Z]

#solver as defined in Chassagneux, Crisan, Delarue
def solver(level,xi_vals,xi_probs):
    #print('Executing solver[level] for level=',level)
    num_initial=len(xi_vals)
    if level==num_t_coarse-1:
        #print('break condition')
        Y_terminal=np.zeros((num_initial))
        for index in range(num_initial):
            Y_terminal[index]=g(index,xi_vals,xi_probs)
        return Y_terminal
    X=[]
    for i in range(num_t_fine):
        X.append([])
        for k in range(num_initial):
            row1=xi_vals[k]*np.ones((2**i))
            X[i]=np.concatenate((X[i],row1))

    X_terminal_probs=[]
    for k in range(num_initial):
        row4=xi_probs[k]*(0.5)**(num_t_fine-1)*np.ones((2**(num_t_fine-1)))
        X_terminal_probs=np.concatenate((X_terminal_probs,row4))

    Y_terminal=np.zeros(num_initial*(2**(num_t_fine-1)))
    num_initial=len(X[0])
    Y=[]
    for i in range(num_t_fine):
        row2=np.zeros((num_initial*2**i))
        Y.append(row2)
    X=(solver_bar(X,Y_terminal,xi_probs,Y))[0]
    
    if level==0:
        Y_0_values=np.zeros((num_keep))
        index=0

    for j in range(J):
        X_terminal=X[num_t_fine-1]
        Y_terminal=solver(level+1,X_terminal,X_terminal_probs)
        [X,Y,Z]=solver_bar(X,Y_terminal,xi_probs,Y)
        if level==0 and j>J-num_keep-1:
            Y_0_values[index]=Y[0]
            index+=1

    Y_initial=Y[0]
    if level==0:
        return [Y_initial,X,Y,Z,Y_0_values]
    return Y_initial

if __name__ == '__main__':
    problem ='flocking_weak' #possible values in order of appearance: jetlag, trader, ex_1, ex_72, ex_73, flocking
    global b
    global f
    global g
    global periodic_2_pi
    global J
    global J_solver_bar
    global num_keep
    global T
    global num_t
    
    global rho
    global sigma
    global a
    global R
    global K
    global F
    global omega_0
    global omega_S
    global p
    global num_intervals_total
    global num_intervals_coarse
    global c_x
    global h_bar
    global c_g
    
    
    if problem =='jetlag_Pontryagin':
        b=b_jet_lag_Pontryagin
        f=f_jet_lag_Pontryagin
        g=g_jet_lag
        periodic_2_pi=True
        J=25
        J_solver_bar=10
        num_keep=5
        T=24.0*10
        num_intervals_total=6
        num_intervals_coarse=1
        x_0=[0.0]
        x_0_probs=[1.0]
        
        # Varible Jet Lag
        R=1
        K=0.01
        F=0.01
        omega_0=2*np.pi/24.5
        omega_S=2*np.pi/24
        p=(3.0/12.0)*np.pi
        sigma=0.1
    elif problem =='jetlag_weak':
        b=b_jet_lag_weak
        f=f_jet_lag_weak
        g=g_jet_lag
        periodic_2_pi=True
        J=25
        J_solver_bar=1
        num_keep=5
        T=24.0*10
        num_intervals_total=6
        num_intervals_coarse=1
        x_0=[0.0]
        x_0_probs=[1.0]
        
        # Varible Jet Lag
        R=1.0
        K=0.01
        F=0.01
        omega_0=2*np.pi/24.5
        omega_S=2*np.pi/24
        p=(3.0/12.0)*np.pi
        sigma=0.1
    elif problem=='ex_1':
        b=b_example_1
        f=f_example_1
        g=g_example_1
        periodic_2_pi=False
        J=25
        J_solver_bar=10
        num_keep=5
        T=1.0
        num_intervals_total=6
        num_intervals_coarse=1
        x_0=[2.0]
        x_0_probs=[1.0]
        sigma=1
        rho=0.1
        a=0.25
    elif problem=='ex_72':
        b=b_example_72
        f=f_example_72
        g=g_example_72
        periodic_2_pi=False
        J=25
        J_solver_bar=10
        num_keep=5
        T=1.0
        num_intervals_total=6
        num_intervals_coarse=1
        x_0=[0.0]
        x_0_probs=[1.0]
        sigma=1.0
        rho=3.5
    elif problem=='ex_73':
        b=b_example_73
        f=f_example_73
        g=g_example_73
        periodic_2_pi=False
        J=25
        J_solver_bar=10
        num_keep=5
        T=1.0
        num_intervals_total=6
        num_intervals_coarse=1
        x_0=[2.0]
        x_0_probs=[1.0]
        sigma=1
        rho=1
    elif problem=='trader_Pontryagin':
        sigma=0.7
        rho=0.05
        c_x=1
        h_bar=10
        c_g=0.3
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
        J_solver_bar=10
        num_keep=5
        T=1.0
        num_intervals_total=6
        num_intervals_coarse=1
        x_0=[1.0]
        x_0_probs=[1.0]
    elif problem=='trader_weak':
        sigma=0.7
        rho=0.03
        c_x=1
        h_bar=10
        c_g=0.3
        b=b_trader_weak
        f=f_trader_weak
        g=g_trader_weak
        periodic_2_pi=False
        J=25
        J_solver_bar=10
        num_keep=5
        T=1.0
        num_intervals_total=6
        num_intervals_coarse=1
        x_0=[1.0]
        x_0_probs=[1.0]
    elif problem=='flocking_Pontryagin':
        sigma=1.0
        b=b_flocking_Pontryagin
        f=f_flocking_Pontryagin
        g=g_flocking
        periodic_2_pi=False
        J=25
        J_solver_bar=10
        num_keep=5
        T=1.0
        num_intervals_total=6
        num_intervals_coarse=1
        x_0=[0.0]
        x_0_probs=[1.0]
    elif problem=='flocking_weak':
        sigma=1.0
        b=b_flocking_weak
        f=f_flocking_weak
        g=g_flocking
        periodic_2_pi=False
        J=25
        J_solver_bar=10
        num_keep=5
        T=1.0
        num_intervals_total=6
        num_intervals_coarse=1
        x_0=[0.0]
        x_0_probs=[1.0]
    
    global num_t_coarse
    num_t_coarse=num_intervals_coarse+1
    global delta_t_coarse
    delta_t_coarse=float(T)/(num_t_coarse-1)
    global num_t_fine
    num_t_fine=int(num_intervals_total/num_intervals_coarse)+1
    global delta_t_fine
    delta_t_fine=delta_t_coarse/(num_t_fine-1)
    global delta_W
    delta_W=math.sqrt(delta_t_fine)
    
    execution='ordinary'
    # possible values in order of appearance:
    # ordinary, changing sigma, changing rho, continuation sigma
    if execution=='ordinary':
        [Y_initial,X,Y,Z,Y_0_values]=solver(0,x_0,x_0_probs)
        print(Y_0_values)
        if problem=='ex_1':
            m_0=0
            for k in range(len(x_0)):
                m_0+=x_0[k]*x_0_probs[k]
            true_Y_0=m_0*math.exp(a*T)/(1+rho/a*(math.exp(a*T)-1.0))
            print('True Answer For Example 1: Y_0=')
            print(true_Y_0)
    elif execution=='changing sigma':
        num_sigma=20
        sigma_values=np.linspace(0.5,10,num_sigma)
        all_Y_0_values=np.zeros((num_sigma,num_keep))
        for index in range(num_sigma):
            index2=0
            sigma=sigma_values[index]
            
            [Y_initial,X,Y,Z,Y_0_values]=solver(0,x_0,x_0_probs)
            all_Y_0_values[index]=Y_0_values
            print(Y_0_values)
            
    elif execution=='changing rho':
        num_rho=10
        rho_values=np.linspace(1,10,num_rho)
        all_Y_0_values=np.zeros((num_rho,num_keep))
        for index in range(num_rho):
            index2=0
            rho=rho_values[index]
            [Y_initial,X,Y,Z,Y_0_values]=solver(0,x_0,x_0_probs)
            all_Y_0_values[index]=Y_0_values
            print(Y_0_values)

    elif execution=='continuation rho':
        delta_rho=0.1
        rho_min=1.0
        rho_max=3.0
        num_rho=int((sigma_max-sigma_min)/delta_sigma)+1
        rho_values=np.linspace(sigma_min,sigma_max,num_sigma)
        all_Y_0_values=np.zeros((num_rho,num_keep))
        
        X=[]
        num_initial=len(x_0)
        for i in range(num_t_fine):
            X.append([])
            for k in range(num_initial):
                row1=x_0[k]*np.ones((2**i))
                X[i]=np.concatenate((X[i],row1))

        Y=[]
        for i in range(num_t_fine):
            if i<num_t_fine-1:
                row2=np.zeros((num_initial*2**i))
                Y.append(row2)
            else:
                Y.append([])
                for k in range(num_initial):
                    row2=g(k,x_0,x_0_probs)*np.ones((2**i))
                    Y[i]=np.concatenate((Y[i],row2))

        Z=[]
        for i in range(num_t_fine):
            row3=np.zeros((num_initial*2**i))
            Z.append(row3)
        
        for index in range(num_rho):
            rho=rho_values[index]
            [X,Y,Z,Y_0_values]=continuation_solver_bar(X,x_0_probs,Y,Z)
            all_Y_0_values[index]=Y_0_values
            print(Y_0_values)

    elif execution=='continuation sigma':
        delta_sigma=1.0
        sigma_min=1.0
        sigma_max=10.0
        num_sigma=int((sigma_max-sigma_min)/delta_sigma)+1
        sigma_values=np.linspace(sigma_min,sigma_max,num_sigma)
        all_Y_0_values=np.zeros((num_sigma,num_keep))
        
        X=[]
        num_initial=len(x_0)
        for i in range(num_t_fine):
            X.append([])
            for k in range(num_initial):
                row1=x_0[k]*np.ones((2**i))
                X[i]=np.concatenate((X[i],row1))

        Y=[]
        for i in range(num_t_fine):
            if i<num_t_fine-1:
                row2=np.zeros((num_initial*2**i))
                Y.append(row2)
            else:
                Y.append([])
                for k in range(num_initial):
                    row2=g(k,x_0,x_0_probs)*np.ones((2**i))
                    Y[i]=np.concatenate((Y[i],row2))

        Z=[]
        for i in range(num_t_fine):
            row3=np.zeros((num_initial*2**i))
            Z.append(row3)
        
        for index in reversed(range(num_sigma)):
            rho=2.0
            sigma=sigma_values[index]
            [X,Y,Z,Y_0_values]=continuation_solver_bar(X,x_0_probs,Y,Z)
            all_Y_0_values[index]=Y_0_values
        print(Y_0_values)
