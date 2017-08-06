__author__ = 'angiuli'
# fuction 1
def pi(x,x_min,x_max,delta_x):

    low=(x-x_min)//delta_x

    if (x-x_min-low*delta_x)<(x_min+(low+1)*delta_x-x):

        x_index=low

    else:

        x_index=low+1


    return(x_index)
#function 2
def pio(x,x_min,x_max,delta_x):




    return(x)
# test has in input a function f and a value x
def test(f,x):

    y=f(x,-10,4,1.3)

    return y

print(test(pi,2)) # test with f=pi
print(pi(2,-10,4,1.3))
print(test(pio,2)) # test with f=pio