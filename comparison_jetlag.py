#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 18:09:24 2017

@author: christy

For comparing jetlag
"""

def firstindex(list,target):
    nl=len(list)
    for i in range(nl):
        if list[i]>target:
#            return i
            if i>0:
                return (i-1)
            else:
                return 0
    return (nl-1)



def Wd(mu_1,grid_1,mu_2,grid_2,Nint,p=2):
    CDF_1=np.cumsum(mu_1)
    CDF_2=np.cumsum(mu_2)
    n_1=len(mu_1)
    n_2=len(mu_2)
    u_vec=np.linspace(0,1,Nint)
    du=1.0/Nint
    
    W=0
    for i in range(Nint):
        i1=firstindex(CDF_1,u_vec[i])
        i2=firstindex(CDF_2,u_vec[i])
        dW=du*pow(math.fabs(grid_1[i1]-grid_2[i2]),p)
        #print(dW)
        W+=dW
    W=pow(W,1.0/p)
    return W
    
if __name__ == '__main__':
    mu_Pontryagin=np.load('./Data/jetlag/0815_Pontryagin.npy')
    mu_weak=np.load('./Data/jetlag/0815_weak_trunc.npy')
    mu_PDE=np.load('./Data/jetlag/mu_reference_set_158.npy')*delta_x

    mu_Pontryagin=mu_Pontryagin[len(mu_Pontryagin)-1]
    mu_weak=mu_weak[len(mu_weak)-1]
    mu_PDE=mu_PDE[len(mu_PDE)-1]

    num_x=158
    delta_x=2*np.pi/num_x
    x_grid=np.linspace(0,2*np.pi-delta_x,num_x)
    
    d1=Wd(mu_Pontryagin,x_grid,mu_weak,x_grid,100)
    d2=Wd(mu_Pontryagin,x_grid,mu_PDE,x_grid,100)
    d3=Wd(mu_weak,x_grid,mu_PDE,x_grid,100)
    
    print(d1,d2,d3)

