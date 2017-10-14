#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 18:09:24 2017

@author: christy

For comparing jetlag
"""
from Wd_exact import *

    
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
    
    d1=Wd_exact_circle(x_grid,mu_Pontryagin,mu_weak,2)
    d2=Wd_exact_circle(x_grid,mu_Pontryagin,mu_PDE,2)
    d3=Wd_exact_circle(x_grid,mu_weak,mu_PDE,2)
    
    print(d1,d2,d3)

