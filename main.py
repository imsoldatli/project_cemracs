#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 09:24:22 2017

@author: christy
"""

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

def g_example_1(x):
    return x
    
def g_example_72(x):
    return np.sin(x)
    
def g_example_73(x):
    return np.arctan(x)

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
        
    for k in range(J):
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
    return [X,Y,Z]

def solver(level,xi_vals,xi_probs):
    #print('Executing solver[level] for level=',level)
    num_initial=len(xi_vals)
    if level==num_t_coarse-1:
        #print('break condition')
        Y_terminal=g(xi_vals)
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
    global J
    J=10
    global num_keep
    num_keep=5
    global num_intervals_total
    num_intervals_total=6
    global T
    T=1.0
    global a
    a=0.25 
    x_0=[2.0]
    x_0_probs=[1.0]

    num_rho=20
    rho_values=np.linspace(1,6,num_rho)
    num_sigma=1
    sigma_values=np.linspace(0.5,10,num_sigma)
    for index in range(num_rho):
    #for index in range(num_sigma):
        global num_intervals_coarse
        num_intervals_coarse=1
        global rho
        rho=rho_values[index]
        #rho=2.0
        global sigma
        #sigma=sigma_values[index]
        sigma=1
        
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
    
        [Y_initial,X,Y,Z,Y_0_values]=solver(0,x_0,x_0_probs)
        print(Y_0_values)
        for index2 in range(num_keep):
            plt.scatter(rho,Y_0_values[index2])
            #plt.scatter(sigma,Y_0_values[index2])
    plt.savefig('two_level_changing_rho_example_72.eps')
    #plt.savefig('one_level_example_73_change_sigma.eps')
        
    
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