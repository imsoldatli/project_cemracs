#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 09:24:22 2017
@author: Andrea Angiuli, Christy Graves, Houzhi Li
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

def b_example_73_E(i,j,X,Y,Z,X_initial_probs):
    num_initial=len(X[0])
    Y_mean=0
    num_per_initial=len(Y[i])/num_initial
    for k in range(len(Y[i])):
        index=int(math.floor(k/num_per_initial))
        Y_mean+=Y[i][k]*X_initial_probs[index]/num_per_initial
    return -rho*Y_mean
    
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

def sigma_continuation_solver_bar(X_ini,X_initial_probs,Y_ini,Z_ini):
    
    num_initial=len(X_ini[0]) 
    Y_0_values=np.zeros((num_keep))
    index=0
    X=X_ini
    Y=Y_ini
    Z=Z_ini
    x_vals=np.zeros(num_initial*2**(num_t_total-1))
    x_probs=[]
    for j in range(num_initial):
            row1=X_initial_probs[j]*np.ones(2**(num_t_total-1))/(2**(num_t_total-1))
            x_probs=np.concatenate((x_probs,row1))
#    print(x_probs)
<<<<<<< HEAD
    for k in range(J):            
        for j in range(num_initial*2**(num_t_total-1)):
            Y[num_t_total-1][j]=g(j,X[num_t_total-1],x_probs)
            
        for index2 in range(J_solver_bar):
            for i in reversed(range(num_t_total-1)):
                for j in range(num_initial*2**i):
    #                temp_Y=(Y[i+1][2*j]+Y[i+1][2*j+1]+delta_t_fine*f(i+1,2*j,X,Y,Z,X_initial_probs)+delta_t_fine*f(i+1,2*j+1,X,Y,Z,X_initial_probs))/2.0
                    temp_Y=(Y[i+1][2*j]+Y[i+1][2*j+1])/2.0+delta_t_fine*f(i,j,X,Y,Z,X_initial_probs)
                    #temp_=delta_t_fine*f(i,j,X,Y,Z,X_initial_probs)
                    Y[i][j]=temp_Y
                    Z[i][j]=delta_W/delta_t_fine*(Y[i+1][2*j]-Y[i+1][2*j+1])/2.0
    #                print(k,i,j,temp_)
            for i in range(num_t_total-1):
                for j in range(num_initial*2**i):
                    X[i+1][2*j]=X[i][j]+delta_t_fine*b(i,j,X,Y,Z,X_initial_probs)+sigma*delta_W
                    X[i+1][2*j+1]=X[i][j]+delta_t_fine*b(i,j,X,Y,Z,X_initial_probs)-sigma*delta_W
=======
    for k in range(J):
        for i in range(num_t_total-1):
            for j in range(num_initial*2**i):
                X[i+1][2*j]=X[i][j]+delta_t_fine*b(i,j,X,Y,Z,X_initial_probs)+sigma*delta_W
                X[i+1][2*j+1]=X[i][j]+delta_t_fine*b(i,j,X,Y,Z,X_initial_probs)-sigma*delta_W
                
        for j in range(num_initial*2**(num_t_total-1)):
            x_vals[j]=X[num_t_total-1][j]
            
        for j in range(num_initial*2**(num_t_total-1)):
            Y[num_t_total-1][j]=g(j,x_vals,x_probs)
            
        for i in reversed(range(num_t_total-1)):
            for j in range(num_initial*2**i):
#                temp_Y=(Y[i+1][2*j]+Y[i+1][2*j+1]+delta_t_fine*f(i+1,2*j,X,Y,Z,X_initial_probs)+delta_t_fine*f(i+1,2*j+1,X,Y,Z,X_initial_probs))/2.0
                temp_Y=(Y[i+1][2*j]+Y[i+1][2*j+1])/2.0+delta_t_fine*f(i,j,X,Y,Z,X_initial_probs)
                temp_=delta_t_fine*f(i,j,X,Y,Z,X_initial_probs)
                Y[i][j]=temp_Y
                Z[i][j]=delta_W/delta_t_fine*(Y[i+1][2*j]-Y[i+1][2*j+1])/2.0
#                print(k,i,j,temp_)
            
>>>>>>> 5e82e47e17b99c6aa719871d62aa8fa7cd9f5273
            
        if k>J-num_keep-1:
            Y_0_values[index]=Y[0][0]
            index+=1
            
    return [X,Y,Z,Y_0_values]


#def sigma_continuation_solver(level,xi_vals,xi_probs,X_ini,Y_ini):
#    #print('Executing solver[level] for level=',level)
#    num_initial=len(xi_vals)
#    if level==num_t_coarse-1:
#        #print('break condition')
#        Y_terminal=np.zeros((num_initial))
#        for index in range(num_initial):
#            Y_terminal[index]=g(index,xi_vals,xi_probs)
#        return Y_terminal
#    X=X_ini
#    X_terminal_probs=[]
#    for k in range(num_initial):
#        row4=xi_probs[k]*(0.5)**(num_t_fine-1)*np.ones((2**(num_t_fine-1)))
#        X_terminal_probs=np.concatenate((X_terminal_probs,row4))
#    
#    Y_terminal=np.zeros(num_initial*(2**(num_t_fine-1)))
#    
#    num_initial=len(X[0])
#    Y=Y_ini
#
#    if level==0:
#        Y_0_values=np.zeros((num_keep))
#        index=0
#        
#    
#    for j in range(J):
#        X_terminal=X[num_t_fine-1]
#        Y_terminal=solver(level+1,X_terminal,X_terminal_probs)
#        [X,Y,Z]=solver_bar(X,Y_terminal,xi_probs,Y)
#        if level==0 and j>J-num_keep-1:
#            Y_0_values[index]=Y[0]
#            index+=1
#            
#    Y_initial=Y[0]
#    if level==0:
#        return [Y_initial,X,Y,Z,Y_0_values]
#    return Y_initial

if __name__ == '__main__':
    global b
    b=b_example_73
    global f
    f=f_example_73
    global g
    g=g_example_73
    global J
<<<<<<< HEAD
    J=10
    global J_solver_bar
    J_solver_bar=10
=======
    J=25
>>>>>>> 5e82e47e17b99c6aa719871d62aa8fa7cd9f5273
    global num_keep
    num_keep=5
    global num_intervals_total
    num_intervals_total=6
    global num_t_total
    num_t_total=num_intervals_total+1
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
#    delta_t_fine=delta_t_coarse/(num_t_fine-1)
    delta_t_fine=T/num_intervals_total
    global delta_W
    delta_W=math.sqrt(delta_t_fine)
    
    global a
    a=0.25 
    x_0=[2.0]
    x_0_probs=[1.0]

<<<<<<< HEAD
    delta_rho=0.2
    rho_min=1.0
    rho_max=2.6
    num_rho=int((rho_max-rho_min)/delta_rho)+1
    rho_values=np.linspace(rho_min,rho_max,num_rho)
    np.save('tree_example_73_sigma_continuation_rho_values',rho_values)
    delta_sigma=0.1
    sigma_min=1.0
    sigma_max=10.0
    num_sigma=int((sigma_max-sigma_min)/delta_sigma)+1
    sigma_values=np.linspace(sigma_min,sigma_max,num_sigma)
    all_Y_0_values=np.zeros((num_rho,num_keep))
    #all_Y_0_values=np.zeros((num_sigma,num_keep))
=======
    num_rho=1
    rho_values=np.linspace(1,6,num_rho)
    num_sigma=1
    sigma_values=np.linspace(0.5,20,num_sigma)
    #all_Y_0_values=np.zeros((num_rho,num_keep))
    all_Y_0_values=np.zeros((num_sigma,num_keep))
>>>>>>> 5e82e47e17b99c6aa719871d62aa8fa7cd9f5273
    #for index in range(num_rho):
        
#    for index in reversed(range(num_sigma)):
#        global rho
#        #rho=rho_values[index]
#        rho=3.0
#        global sigma
#        sigma=sigma_values[index]
#        #sigma=1
#    
#        [Y_initial,X,Y,Z,Y_0_values]=solver(0,x_0,x_0_probs)
#        all_Y_0_values[index]=Y_0_values
#        print(Y_0_values)
        
        #for index2 in range(num_keep):
            #plt.scatter(rho,Y_0_values[index2])
            #plt.scatter(sigma,Y_0_values[index2])
    #plt.savefig('two_level_changing_rho_example_72.eps')
    #plt.savefig('one_level_example_73_change_sigma.eps')
#    np.save('tree_example_73_E_sigma_values',sigma_values)
#    np.save('tree_example_73_E_one_level_changing_sigma',all_Y_0_values)

# first initialization
<<<<<<< HEAD

    for rho_index in range(num_rho):
        global rho
        rho=rho_values[rho_index]
        #rho=1.7
        X=[]
        num_initial=len(x_0)
        for i in range(num_t_total):
            X.append([])
            for k in range(num_initial):
                row1=x_0[k]*np.ones((2**i))
                X[i]=np.concatenate((X[i],row1))
                
        Y=[]
        for i in range(num_t_total):
            if i<num_t_total-1:
                row2=np.zeros((num_initial*2**i))
                Y.append(row2)
            else:
                Y.append([])
                for k in range(num_initial):
                    row2=g(k,x_0,x_0_probs)*np.ones((2**i))
                    Y[i]=np.concatenate((Y[i],row2))
      
        Z=[]
        for i in range(num_t_total):
            row3=np.zeros((num_initial*2**i))
            Z.append(row3)
    #    print(Y)
    
        for index in reversed(range(num_sigma)):
        #for index in range(num_rho):
            #global rho
            #rho=rho_values[rho_index]
            #rho=3.0
            global sigma
            sigma=sigma_values[index]
            #sigma=1.0
    #        print(Y)
            [X,Y,Z,Y_0_values]=sigma_continuation_solver_bar(X,x_0_probs,Y,Z)
            #print('rho=',rho)
    #        print(Y)
            #all_Y_0_values[index]=Y_0_values
            #print(Y_0_values)
        all_Y_0_values[rho_index]=Y_0_values
        print(Y_0_values)
        
    #np.save('tree_example_73_rho_values_1',rho_values)
    #np.save('tree_example_73_one_level_continuation_rho_1',all_Y_0_values)
    np.save('tree_example_73_sigma_continuation_rho_values',rho_values)
    #np.save('tree_example_73_one_level_continuation_sigma_3',all_Y_0_values)
        
=======
    X=[]
    num_initial=len(x_0)
    for i in range(num_t_total):
        X.append([])
        for k in range(num_initial):
            row1=x_0[k]*np.ones((2**i))
            X[i]=np.concatenate((X[i],row1))
            
    Y=[]
    for i in range(num_t_total):
        if i<num_t_total-1:
            row2=np.zeros((num_initial*2**i))
            Y.append(row2)
        else:
            Y.append([])
            for k in range(num_initial):
                row2=g(k,x_0,x_0_probs)*np.ones((2**i))
                Y[i]=np.concatenate((Y[i],row2))
  
    Z=[]
    for i in range(num_t_total):
        row3=np.zeros((num_initial*2**i))
        Z.append(row3)
#    print(Y)
    for index in reversed(range(num_sigma)):
        global rho
#        rho=rho_values[index]
        rho=4.0
        global sigma
#        sigma=sigma_values[index]
        sigma=1.0
#        print(Y)
        [X,Y,Z,Y_0_values]=sigma_continuation_solver_bar(X,x_0_probs,Y,Z)
        print('sigma=',sigma)
#        print(Y)
        all_Y_0_values[index]=Y_0_values
        print(Y_0_values)
        
>>>>>>> 5e82e47e17b99c6aa719871d62aa8fa7cd9f5273
#    for index in range(num_rho):
#        global rho
#        rho=rho_values[index]
##        rho=5.0
##        global sigma
##        sigma=sigma_values[index]
#        sigma=1.0
##        print(Y)
#        [X,Y,Z,Y_0_values]=sigma_continuation_solver_bar(X,x_0_probs,Y,Z)
#        print('rho=',rho)
##        print(Y)
#        all_Y_0_values[index]=Y_0_values
#        print(Y_0_values)
        
        

#    m_0=0
#    for k in range(len(x_0)):
#        m_0+=x_0[k]*x_0_probs[k]
#    true_Y_0=m_0*math.exp(a*T)/(1+rho/a*(math.exp(a*T)-1.0))
#    print(true_Y_0)
    
#    np.save('continuation_sigma_tree_example_72',sigma_values)
#    np.save('continuation_sigma_tree_example_72_one_level',all_Y_0_values)

#    Y_0=0
#    m_0=0
#    for k in range(len(x_0)):
#        Y_0+=Y_initial[k]*x_0_probs[k]
#        m_0+=x_0[k]*x_0_probs[k]
#
#    true_Y_0=m_0*math.exp(a*T)/(1+rho/a*(math.exp(a*T)-1.0))
#    print('True Answer For Example 1: Y_0=')
#    print(true_Y_0)
#    print('Our Answer: Y_0=')
#    print(Y_0)
#    
#    print('Log Num Time Steps')
#    print(math.log(num_intervals_total))
#    print('Log Difference')
#    print(math.log(abs(true_Y_0-Y_0)))
