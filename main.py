0#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 09:24:22 2017
@author: Andrea Angiuli, Christy Graves, Houzhi Li
"""
#boring comment
import numpy as np
import math
import matplotlib.pyplot as plt

def b_example_1(i,j,X,Y,Z,X_initial_probs):
    num_initial=len(X[0])
    Y_mean=0
    num_per_initial=len(Y[i])/num_initial
    for k in range(len(Y[i])):
        index=int(math.floor(k/num_per_initial))
        Y_mean+=Y[i][k]*X_initial_probs[index]/num_per_initial
    return -rho*Y_mean
    
def b_example_72(i,j,X,Y,Z,X_initial_probs):
    return rho*np.cos(Y[i][j])
    
def b_example_73(i,j,X,Y,Z,X_initial_probs):
    return -rho*Y[i][j]

def b_example_73_E(i,j,X,Y,Z,X_initial_probs):
    num_initial=len(X[0])
    Y_mean=0
    num_per_initial=len(Y[i])/num_initial
    for k in range(len(Y[i])):
        index=int(math.floor(k/num_per_initial))
        Y_mean+=Y[i][k]*X_initial_probs[index]/num_per_initial
    return -rho*Y_mean
    
def b_jet_lag_weak(i,j,X,Y,Z,X_initial_probs):
    return omega_0-omega_S-1.0/(R*sigma)*Z[i][j]

def b_jet_lag_Pontryagin(i,j,X,Y,Z,X_initial_probs):
    return omega_0-omega_S-1.0/R*Y[i][j]
    
def f_example_1(i,j,X,Y,Z,X_initial_probs):
    return a*Y[i][j]

def f_example_72(i,j,X,Y,Z,X_initial_probs):
    return 0

def f_example_73(i,j,X,Y,Z,X_initial_probs):
    num_initial=len(X[0])
    X_mean=0
    num_per_initial=len(X[i])/num_initial
    for k in range(len(X[i])):
        index=int(math.floor(k/num_per_initial))
        X_mean+=X[i][k]*X_initial_probs[index]/num_per_initial
    return -math.atan(X_mean)
    
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
    
def f_jet_lag_Pontryagin(i,j,X,Y,Z,X_initial_probs):
    num_initial=len(X[0])
    num_per_initial=len(X[i])/num_initial
    X_probs=np.zeros((len(X[i])))
    for k in range(len(X[i])):
        index=int(math.floor(k/num_per_initial))
        X_probs[k]=X_initial_probs[index]/num_per_initial
    partial_c_bar=np.dot(0.5*np.sin((X[i][j]-X[i])/2.0)*np.cos((X[i][j]-X[i])/2.0),X_probs)
    value1=-K*partial_c_bar
    partial_c_sun=0.5*np.sin((X[i][j]-p)/2.0)*np.cos((X[i][j]-p)/2.0)
    value2=-F*partial_c_sun
    return value1+value2

def g_example_1(index,xi_vals,xi_probs):
    x=xi_vals[index]
    return x
    
def g_example_72(index,xi_vals,xi_probs):
    x=xi_vals[index]
    return np.sin(x)
    
def g_example_73(index,xi_vals,xi_probs):
    x=xi_vals[index]
    return np.arctan(x)

def g_example_73_E(index,xi_vals,xi_probs):
    X_mean=np.dot(xi_vals,xi_probs)
    return np.arctan(X_mean)
    
def g_jet_lag(index,xi_vals,xi_probs):
    return 0

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
    global b
    b=b_example_73
    global f
    f=f_example_73
    global g
    g=g_example_73
    global periodic_2_pi
    periodic_2_pi=False
    global J
    J=10
    global J_solver_bar
    J_solver_bar=10
    global num_keep
    num_keep=5
    global num_intervals_total
    num_intervals_total=6
    global T
    T=1.0
    global num_intervals_coarse
    num_intervals_coarse=1
    global num_t_coarse
    num_t_coarse=num_intervals_coarse+1
    global delta_t_coarse
    delta_t_coarse=T/(num_t_coarse-1)
    global num_t_fine
    num_t_fine=num_intervals_total/num_intervals_coarse+1
    global delta_t_fine
    delta_t_fine=delta_t_coarse/(num_t_fine-1)
    global delta_W
    delta_W=math.sqrt(delta_t_fine) 
    
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
    p=0
    
    x_0=[2.0]
    x_0_probs=[1.0]


    

    num_rho=1
    num_sigma=1
    sigma_values=np.linspace(0.5,10,num_sigma)
    #all_Y_0_values=np.zeros((num_rho,num_keep))
    all_Y_0_values=np.zeros((num_sigma,num_keep))
    #for index in range(num_rho):
    for index in range(num_sigma):
        global rho
        #rho=rho_values[index]
        rho=2.0
        global sigma
        #sigma=sigma_values[index]
        sigma=1.0
    
        [Y_initial,X,Y,Z,Y_0_values]=solver(0,x_0,x_0_probs)
        all_Y_0_values[index]=Y_0_values
        print(Y_0_values)
        #for index2 in range(num_keep):
            #plt.scatter(rho,Y_0_values[index2])
            #plt.scatter(sigma,Y_0_values[index2])
    #plt.savefig('two_level_changing_rho_example_72.eps')
    #plt.savefig('one_level_example_73_change_sigma.eps')
    #np.save('tree_example_73_E_sigma_values',sigma_values)
    #np.save('tree_example_73_E_one_level_changing_sigma',all_Y_0_values)
    
    Y_0=0
    m_0=0
    for k in range(len(x_0)):
        Y_0+=Y_initial[k]*x_0_probs[k]
        m_0+=x_0[k]*x_0_probs[k]

    true_Y_0=m_0*math.exp(a*T)/(1+rho/a*(math.exp(a*T)-1.0))
    print('True Answer For Example 1: Y_0=')
    print(true_Y_0)
    print('Our Answer: Y_0=')
    print(Y_0)
    
    print('Log Num Time Steps')
    print(math.log(num_intervals_total))
    print('Log Difference')
    print(math.log(abs(true_Y_0-Y_0)))
