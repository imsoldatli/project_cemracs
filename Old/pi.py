from __future__ import division
__author__ = 'angiuli'

import numpy

def pi(x,x_min,x_max,delta_x):

    low=(x-x_min)//delta_x

    if low>=num_x-1:

        x_index=num_x-1

    elif low<1:

        x_index=0

    elif (x-x_min-low*delta_x)<(x_min+(low+1)*delta_x-x):

        x_index=low
    else:

        x_index=low+1

    return(x_index)

pi(-8.5,-10,4,1.3)
