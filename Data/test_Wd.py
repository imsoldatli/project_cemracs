#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:54:29 2017

@author: christy
"""
import numpy as np
from Wd_exact import *

x_grid=np.linspace(0,1,10)
delta_x=x_grid[1]-x_grid[0]
mu=[0, 0, 0.1, 0.25, 0.15, 0.02, 0.01, 0, 0, 0]
mu_1=mu/np.sum(mu)
mu_2=[mu_1[(i-4)%10] for i in range(10)]
           
cdf1=np.cumsum(mu_1)
cdf2=np.cumsum(mu_2)
           
d1=Wd_exact_R(x_grid,mu_1,mu_2,2)
print(d1)

d1=Wd_approx_R(x_grid,mu_1,mu_2,2)
print(d1)

