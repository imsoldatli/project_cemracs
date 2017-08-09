from __future__ import division
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:37:27 2017
    
@author: Andrea Angiuli, Christy Graves, Houzhi Li
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

def f_example_1(i,j,mu,u,v):
    return a*u[i][j]

def g_example_1(x):
    return x

def b_example_72(i,j,mu,u,v):
    return rho*np.cos(u[i][j])

def f_example_72(i,j,mu,u,v):
    return 0

def g_example_72(x):
    return np.sin(x)

def b_example_73(i,j,mu,u,v):
    return -rho*u[i][j]

def f_example_73(i,j,mu,u,v):
    X_mean=np.dot(x_grid,mu[i])
    return -np.arctan(X_mean)

def g_example_73(x):
    return np.arctan(x)

def b_jet_lag_weak(i,j,mu,u,v):
    return omega_0-omega_S-1.0/(R*sigma)*v[i][j]

def f_jet_lag_weak(i,j,mu,u,v):
    value1=1.0/sigma*(omega_0-omega_S)*v[i][j]-1.0/(2*R*sigma**2)*(v[i][j])**2
    c_bar=np.dot(0.5*(np.sin((x_grid[j]-x_grid)/2.0))**2,mu[i])
    value2=K*c_bar
    c_sun=0.5*(np.sin((p-x_grid[j])/2.0))**2
    value3=F*c_sun
    return value1+value2+value3

def g_jet_lag(x):
    return 0

def b_jet_lag_Pontryagin(i,j,mu,u,v):
    return omega_0-omega_S-1.0/R*u[i][j]

def f_jet_lag_Pontryagin(i,j,mu,u,v):
    partial_c_bar=np.dot(0.5*np.sin((x_grid[j]-x_grid)/2.0)*np.cos((x_grid[j]-x_grid)/2.0),mu[i])
    value1=K*partial_c_bar
    partial_c_sun=0.5*np.sin((x_grid[j]-p)/2.0)*np.cos((x_grid[j]-p)/2.0)
    value2=F*partial_c_sun
    return value1+value2

def b_trader_Pontryagin(i,j,mu,u,v):
    return -rho*u[i][j] #rho=1/c_alpha

def f_trader_Pontryagin(i,j,mu,u,v):
    Y_mean=np.dot(u[i],mu[i])
    return c_x*x_grid[j]+h_bar*rho*Y_mean

def g_trader_Pontryagin(x):
    return c_g *x

def b_trader_weak(i,j,mu,u,v):
    return -rho*v[i][j]/sigma #rho=1/c_alpha

def f_trader_weak(i,j,mu,u,v):
    Z_mean=np.dot(v[i],mu[i])
    return 0.5*c_x*x_grid[j]**2+x_grid[j]*h_bar*rho*Z_mean/sigma-rho*0.5*v[i][j]**2/sigma**2

def g_trader_weak(x):
    return c_g*0.5*x**2


def b_trustworthy_trader(i,j,mu,u,v):
    x_mean=np.dot(x_grid,mu[i])
    return -rho*(eta[0,i]*x_grid[j]+(eta_bar[0,i]-eta[0,i])*x_mean)

def f_trustworthy_trader(i,j,mu,u,v):
    return 0

def g_trustworthy_trader(x):
    return 0
    
def b_flocking_Pontryagin(i,j,mu,u,v):
    return -u[i][j]

def f_flocking_Pontryagin(i,j,mu,u,v):
    X_mean=np.dot(x_grid,mu[i])
    return x_grid[j]-X_mean

def g_flocking(x):
    return 0
    
def b_flocking_weak(i,j,mu,u,v):
    return -v[i][j]/sigma

def f_flocking_weak(i,j,mu,u,v):
    X_mean=np.dot(x_grid,mu[i])
    return -1.0/(2*sigma**2)*(v[i][j])**2+0.5*(x_grid[j]-X_mean)**2

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
    return mu

def backward(mu,u_old,v_old):
    
    u = np.zeros((num_t,num_x))
    v = np.zeros((num_t,num_x))
    
    u[num_t-1,:] = g(x_grid)
    v[num_t-1,:] = v_old[num_t-1,:]
    
    global linear_int
    linear_int=False
    
    for i in reversed(range(num_t-1)):
        for j in range(num_x):
            x_down = x_grid[j] + b(i, j, mu, u_old, v_old) * delta_t - sigma * np.sqrt(delta_t)
            
            x_up = x_grid[j] + b(i, j, mu, u_old, v_old) * delta_t + sigma * np.sqrt(delta_t)
            
            j_down = pi(x_down)
            
            j_up = pi(x_up)
            
            if i==num_t-2:
                
                u[i][j] = (g(x_down) + g(x_up))/2.0 + delta_t*f(i,j,mu,u_old,v_old)
                
                v[i][j] = 1.0/np.sqrt(delta_t) * (g(x_up) - g(x_down))
        
            else:
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
                
                u[i][j] = (u_down + u_up)/2.0 + delta_t*f(i,j,mu,u_old,v_old)
                
                v[i][j] = 1.0/np.sqrt(delta_t) * (u_up - u_down)

    return [u,v]


if __name__ == '__main__':


    problem='trader_Pontryagin'
    execution='changing rho'

    #possible values in order of appearance:
    # jetlag_weak, jetlag_Pontryagin, trader_weak, trader_Pontryagin, ex_1, ex_72, ex_73
    # trustworthy_trader



    global b
    global f
    global g
    global periodic_2_pi
    global J
    global num_keep
    global T
    global num_t
    
    global delta_t
    global t_grid
    global delta_x
    global x_min
    global x_max
    global x_grid
    global rho
    global sigma
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
    elif problem =='jetlag_weak':
        sigma=0.1
        b=b_jet_lag_weak
        f=f_jet_lag_weak
        g=g_jet_lag
        periodic_2_pi=True
        J=25
        num_keep=5
        T=24.0*2
        num_t=int(T)*5+1
        delta_t=T/(num_t-1)
        t_grid=np.linspace(0,T,num_t)
        delta_x=delta_t**2
        num_x=int((2*np.pi)/(delta_x))+1
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
        p=(3.0/12.0)*np.pi
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
        x_min=-10
        x_max=10
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
        num_x=int((x_max-x_min)/delta_x+1)
        x_grid=np.linspace(x_min,x_max,num_x)
        sigma=1
        rho=1
    elif problem=='trader_Pontryagin':
        sigma=0.7
        rho=1.5
        c_x=2
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
        num_t=20
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
        rho=1.5
        c_x=2
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
    # Variable trader

    elif problem=='trustworthy_trader':
        sigma=0.7
        rho=1
        c_x=0.1
        h_bar=2
        c_g=0.3
        b=b_trustworthy_trader
        f=f_trustworthy_trader
        g=g_trustworthy_trader
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
        T=10
        num_t=10*20
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
        


    # possible values in order of appearance:
    # ordinary, changing sigma, changing rho, adaptive,
    # solution_trader, true_start
    if execution=='ordinary':
        mu_0=np.zeros((num_x))
        if periodic_2_pi:
            mu_0=np.load('mu_initial_reference_set_158.npy')
            #mu_0=scipy.io.loadmat('mu_initial_reference_set_158.mat')['mu_initial']
            #mu_0=[mu_0[0][6*i] for i in range(num_x)]
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
        for j in range(J):
            [u,v]=backward(mu,u,v)
            mu=forward(u,v,mu_0)
            if j>J-num_keep-1:
                all_Y_0_values[0][index2]=np.dot(u[0],mu[0])
                index2+=1
        print all_Y_0_values[0]

        #np.save('mu_Pont_t20.npy',mu)



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
        #         j_down = pi(x_grid[j] + b(i, j, mu, u, v) * delta_t - sigma * np.sqrt(delta_t))
        #         j_up = pi(x_grid[j] + b(i, j, mu, u, v) * delta_t + sigma * np.sqrt(delta_t))
        #         mu_u[i][j] = mu[i+1][j_down]+mu[i+1][j_up]
        #         mu_v[i][j] = mu[i+1][j_down]+mu[i+1][j_up]
        # test=np.zeros((num_t,num_x))
        #print(mu_u[num_t//2-1][:])





    ###############
    elif execution=='changing sigma':
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
    
    elif execution=='changing rho':
        num_rho=20
        rho_values=np.linspace(1,21,num_rho)
        all_Y_0_values=np.zeros((num_rho,num_keep))
        value_x=num_keep*[1]
        for index in range(num_rho):
            index2=0
            #rho=rho_values[index]
            c_x=rho_values[index]
            
            
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

            plot_cx=plt.plot(np.multiply(rho,value_x),all_Y_0_values[index])
        plt.title('$\sigma = 0.7$, $\rho = 1.5$, $\bar{h}=2$, $c_g=0.3$, $c_x \in [1,20]$')
        plt.show()

        plt.savefig('grid_trader_pontryagin_changing_cx.png')
        #np.save('grid_trader_pontryagin_rho_larger_x_domain.npy',all_Y_0_values)



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

    elif execution=='solution_trader':
        mu_0=np.zeros((num_x))
        mu_0[int(num_x/2)]=1.0

        mu=np.zeros((num_t,num_x))
        for k in range(num_t):
            mu[k]=mu_0
        u=np.zeros((num_t,num_x))
        v=np.zeros((num_t,num_x))
        mu=forward(u,v,mu_0)

        np.save('solution_trader.npy',mu)
    elif execution=='true_start':
        if periodic_2_pi:
            mu=np.load('mu_reference_set_158.npy')*delta_x
            u=np.load('u_reference_set_158.npy')
        elif problem=='trader_weak' or problem=='trader_Pontryagin':
            mu=true_solution=np.load('solution_trader.npy')
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



            np.save('mu_trader_true_start_t20.npy',mu)

