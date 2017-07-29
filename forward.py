from __future__ import division

__author__ = 'angiuli'

import numpy as np
from math import sqrt
import random


def b0(i,j,X,Y,Z):
    return(2*i-j)

def b1(i,j,X,Y,Z):
    rho=0.1
    Y_mean=0
    for k in range(num_x):
        Y_mean+=Y[i][k]
    Y_mean=Y_mean/(num_x)
    return -rho*Y_mean

def f1(i,j,X,Y,Z):
    a=0.25
    return a*Y[i][j]

def g1(x):
    return x


def pi(x,x_min,x_max,delta_x):

    low=(x-x_min)//delta_x

    if x>=x_max:

        x_index=num_x-1

    elif x<=x_min:

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
    num_x=6
    global delta_x
    delta_x=(x_max-x_min)/(num_x-1)
    global x_grid
    x_grid=np.linspace(x_min,x_max,num_x)
    global sigma
    sigma=1
    global b
    b=b0
    global f
    f=f1
    global g
    g=g1

    mu_0=np.random.rand(1,num_x)
    mu_0=mu_0/mu_0.sum()

    # mu_0=np.zeros((num_x))
    # mu_0[num_x/2]=1.0
    # mu=np.zeros((num_t,num_x))
    # for k in range(num_t):
    #     for j in range(num_x):
    #         mu[k][j]=mu_0[j]
    u=np.zeros((num_t,num_x))
    v=np.zeros((num_t,num_x))

    forward(u,v,mu_0)
