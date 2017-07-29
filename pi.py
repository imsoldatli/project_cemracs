from __future__ import division
__author__ = 'angiuli'

import numpy

def pi(x,x_min,x_max,delta_x):


    low=(x-x_min)//delta_x

    if (x-x_min-low*delta_x)<(x_min+(low+1)*delta_x-x):

        x_index=low

    else:

        x_index=low+1

    # print(x_index)
    # print(x-x_min-low*delta_x)
    # print(x_min+(low+1)*delta_x-x)

    return(x_index)

pi(-8.5,-10,4,1.3)