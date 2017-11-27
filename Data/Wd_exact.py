#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:21:17 2017

@author: christy
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import time
import scipy.stats

def firstindex_approx(cdf,target):
    nl=len(cdf)
    for i in range(nl):
        if cdf[i]>target:
#            return i
            if i>0:
                return (i-1)
            else:
                return 0
    return (nl-1)


def Wd_approx_R(x_grid,mu_1,mu_2,p):
    Nint=100000
    CDF_1=np.cumsum(mu_1)
    CDF_2=np.cumsum(mu_2)
    n_1=len(mu_1)
    n_2=len(mu_2)
    u_vec=np.linspace(0,1,Nint)
    du=1.0/Nint
    
    W=0
    for i in range(1,Nint):
        i1=firstindex_exact(CDF_1,u_vec[i])
        i2=firstindex_exact(CDF_2,u_vec[i])
        dW=du*pow(math.fabs(x_grid[i1]-x_grid[i2]),p)
        W+=dW
    W=pow(W,1.0/p)
    return W
    
    
def firstindex_exact(cdf_array,value):
    index=-1;
    n=len(cdf_array)
    if value<=cdf_array[0]:
        index=0

    for i in range(1,n):
        if value>cdf_array[i-1] and value<=cdf_array[i]:
            index=i

    if index==-1:
        index=n-1
    return index
    
def Wd_exact_R(x_grid,mu_1,mu_2,p):
    CDF_1=np.cumsum(mu_1)
    CDF_2=np.cumsum(mu_2)

    u_vec=np.union1d(CDF_1,CDF_2)
    u_vec=np.union1d(u_vec,[0])
    N=len(u_vec)
    fval=0
    for i in range(N):
        if i>0:
            du=u_vec[i]-u_vec[i-1]
        else:
            du=u_vec[i]
        i1=firstindex_exact(CDF_1,u_vec[i])
        i2=firstindex_exact(CDF_2,u_vec[i])
        fval=fval+du*pow(math.fabs(x_grid[i1]-x_grid[i2]),p)
    fval=pow(fval,1.0/p)
    return fval
    
def Wd_exact_circle(x_grid,mu_1,mu_2,p):
    N=len(x_grid)
    Wd=10000
    for i in range(N):
        mu_1=[mu_1[(i+1)%N] for i in range(N)]
        mu_2=[mu_2[(i+1)%N] for i in range(N)]
        Wd=min(Wd,Wd_exact_R(x_grid,mu_1,mu_2,p))
    return Wd