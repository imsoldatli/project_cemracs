__author__ = 'angiuli'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math

# load solutions
weak=np.load('./Data/trader/mu_weak_t20.npy')
weak_trunc=np.load('./Data/trader/mu_weak_trunc_t20.npy')
Pont=np.load('./Data/trader/mu_Pont_t20.npy')
true_solution=np.load('./Data/trader/trader_solution.npy')
true_solution_hist=np.load('./Data/trader/trader_solution_hist.npy')
true_start=np.load('./Data/trader/mu_trader_true_start_t20.npy')

# intialize counters of matching points between the solutions

compare_weak_pont=0 #compare weak vs pontryagin
compare_weak_true=0 #compare weak vs true solution
compare_pont_true=0 #compare pont vs true solution
compare_true_start_solution=0 #compare true vs pontryagin started from the true solution
compare_shift=0

# construct x_grid (it will be necessary for the animation and to calculate the mean)
T=1
num_t=20
delta_t=(T-0.06)/(num_t-1)
t_grid=np.linspace(0.06,T,num_t)
delta_x=delta_t**(2)
x_min=-2
x_max=4
num_x=int((x_max-x_min)/delta_x+1)
x_grid=np.linspace(x_min,x_max,num_x)

# dimension of the solutions
row=len(weak)
col=len(weak[0])

# max_diff[t]: max of the differences between two solutions at time t
# max_diff initialization
max_diff_weak_Pont=[] # Weak solution vs Pontryagin solution
max_diff_true_weak=[] # true solution vs Weak solution
max_diff_true_Pont=[] # true solution vs Pontryagin solution
max_diff_true_start_true_solut=[] # Pontryagin solution started in the true solution vs true solution
max_diff_true_start_Pont=[] # Pontryagin solution started in the true solution  vs Pontryagin solution

# clean the solutions by all values behind a threshold and save the index of the remaining values
threshold=10**(-3)
# weak
w=[]
index_w=[]
# Pontryagin
P=[]
index_P=[]
# true solution
T=[]
index_T=[]

row_c=[]

# initializing the mean of each solution
mean_weak=[]
mean_Pont=[]
mean_true_solution=[]

for i in range(row):
    # evaluation of the max_diff
    max_diff_weak_Pont.append(np.max(abs(weak[i]-Pont[i])))
    max_diff_true_weak.append(np.max(abs(weak[i]-true_solution[i])))
    max_diff_true_Pont.append(np.max(abs(true_solution[i]-Pont[i])))
    max_diff_true_start_true_solut.append(np.max(abs(true_solution[i]-true_start[i])))
    max_diff_true_start_Pont.append(np.max(abs(Pont[i]-true_start[i])))

    # cleaning the solutions & counting matching points
    row_iw=[]
    row_w=[]
    row_P=[]
    row_iP=[]
    row_T=[]
    row_iT=[]

    for j in range(col):
        if weak[i][j]>5*10**-2:
            row_w.append(weak[i][j])
            row_iw.append(j)
        if Pont[i][j]>5*10**-2:
            row_P.append(Pont[i][j])
            row_iP.append(j)
        if true_solution[i][j]>10**-2:
            row_T.append(true_solution[i][j])
            row_iT.append(j)

        if abs(weak[i][j]-Pont[i][j])<10**(-2):
            compare_weak_pont=compare_weak_pont+1
        if abs(weak[i][j]-true_solution[i][j])<10**(-2):
            compare_weak_true=compare_weak_true+1
        if abs(Pont[i][j]-true_solution[i][j])<10**(-2):
            compare_pont_true=compare_pont_true+1
    mean_weak.append(np.dot(x_grid,weak[i]))
    mean_Pont.append(np.dot(x_grid,Pont[i]))
    mean_true_solution.append(np.dot(x_grid,true_solution[i]))

    w.append(row_w)
    index_w.append(row_iw)
    P.append(row_P)
    index_P.append(row_iP)
    T.append(row_T)
    index_T.append(row_iT)


# print max diff
plt.axes(xlim=(0, 1), ylim=(0,0.6))
print('max diff: weak vs Pont',max_diff_weak_Pont)
plt.plot(t_grid,max_diff_weak_Pont,'o',label='weak vs Pont')
print('max diff: true vs Pont',max_diff_true_Pont)
#plt.plot(t_grid,max_diff_true_Pont,'o',label='true vs Pont')
print('max diff: true vs weak',max_diff_true_weak)
#plt.plot(t_grid,max_diff_true_weak,'o',label='true vs weak')
print('max diff: true start vs true solution',max_diff_true_start_true_solut)
#plt.plot(t_grid,max_diff_true_start_true_solut,'o',label='Pont start by true vs true solution')
print('max diff: true start vs Pont',max_diff_true_start_Pont)
#plt.plot(t_grid,max_diff_true_start_Pont,'o',label='Pont start by true vs Pont')
plt.xlabel('time grid')
plt.ylabel('max difference')
plt.title('max of the differences between two solutions at time t')

plt.legend()
plt.imshow
plt.savefig('grid_trader_error_max_diff.eps')


# print a specific time of the cleaned vectors
print('w',w[4])
print('P',P[4])
print('T',T[4])
print('index_w',index_w[4])
print('index_P',index_P[4])
print('index_T',index_T[4])


# print mean
print('mean_w',mean_weak)
print('mean_P',mean_Pont)
print('mean_T',mean_true_solution)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# animation: plot all the solutions evolving in time
num_bins=int(num_x/15)
print('num_x',num_x)
num_x_hist=int(num_x/num_bins)
delta_x_hist=np.abs(x_max-x_min)/num_x_hist
x_grid_hist=np.linspace(x_min,x_max,num_x_hist)
mu_weak_hist=np.zeros((num_t,num_x_hist))
mu_weak_trunc_hist=np.zeros((num_t,num_x_hist))
mu_Pont_hist=np.zeros((num_t,num_x_hist))
mu_T_hist=np.zeros((num_t,num_x_hist))

for t in range(num_t):
    for i in range(num_x/num_bins):
        mu_weak_hist[t,i]=np.sum(weak[t,num_bins*i:num_bins*(i+1)])
        mu_weak_trunc_hist[t,i]=np.sum(weak_trunc[t,num_bins*i:num_bins*(i+1)])
        mu_Pont_hist[t,i]=np.sum(Pont[t,num_bins*i:num_bins*(i+1)])
        mu_T_hist[t,i]=np.sum(true_solution[t,num_bins*i:num_bins*(i+1)])

# print('sum mu_w_hist',np.sum(mu_weak_hist[num_t-1]))
# print('sum mu_w_trunc_hist',np.sum(mu_weak_trunc_hist[num_t-1]))
# print('sum mu_P_hist',np.sum(mu_Pont_hist[num_t-1]))
# print('sum mu_T_hist',np.sum(mu_T_hist[num_t-1]))

fig = plt.figure()
ax1 = plt.axes(xlim=(-1, 3), ylim=(0,.8))
line, = ax1.plot([], [],'o')
plotlays, plotcols = [3], ["green","yellow","red","blue"]
labels=['weak','Pont','true','weak trunc']
lines = []
for index in range(4):
    #lobj = ax1.plot([],[],'o',color=plotcols[index],label=labels[index])[0]
    lobj = ax1.plot([],[],color=plotcols[index],label=labels[index])[0]

    plt.legend()
    lines.append(lobj)


def init():
    for line in lines:
        line.set_data([],[])
    return lines

x1,y1 = [],[]
x2,y2 = [],[]
x3,y3= [],[]
x4,y4= [],[]


# fake data
frame_num = 20

def animate(i):

    x1,y1 = [],[]
    x2,y2 = [],[]
    x3,y3= [],[]
    x4,y4= [],[]

    x = x_grid_hist
    y = mu_weak_hist[i]
    x1.append(x)
    y1.append(y)

    x = x_grid_hist
    y = mu_Pont_hist[i]
    x2.append(x)
    y2.append(y)

    x = x_grid_hist
    y = mu_T_hist[i]
    x3.append(x)
    y3.append(y)

    x = x_grid_hist
    y = mu_weak_trunc_hist[i]
    x4.append(x)
    y4.append(y)

    xlist = [x1, x2, x3, x4]
    ylist = [y1, y2, y3, y4]

    #for index in range(0,1):
    for lnum,line in enumerate(lines):
        line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately.

    return lines

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frame_num, interval=450, blit=False)


plt.show()

#anim.save('comparison_mu_trader.mp4')

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

if __name__=='__main__':
    wdist_weak_Pont=Wd(weak[num_t-1],x_grid,Pont[num_t-1],x_grid,10**4)
    print('wdist_weak_Pont',wdist_weak_Pont)
    wdist_true_weak=Wd(weak[num_t-1],x_grid,true_solution[num_t-1],x_grid,10**4)
    print('wdist_true_weak',wdist_true_weak)
    wdist_true_Pont=Wd(true_solution[num_t-1],x_grid,Pont[num_t-1],x_grid,10**4)
    print('wdist_true_Pont',wdist_true_Pont)
    wdist_true_start_true_solut=Wd(true_solution[num_t-1],x_grid,true_start[num_t-1],x_grid,10**4)
    print('wdist_true_start_true_solut',wdist_true_start_true_solut)
    wdist_true_start_Pont=Wd(true_start[num_t-1],x_grid,Pont[num_t-1],x_grid,10**4)
    print('wdist_true_start_Pont',wdist_true_start_Pont)
    wdist_weak_trunc_true=Wd(weak_trunc[num_t-1],x_grid,true_solution[num_t-1],x_grid,10**4)
    print('wdist_weak_trunc_true',wdist_weak_trunc_true)

    wdist_weak_Pont_hist=Wd(mu_weak_hist[num_t-1],x_grid_hist,mu_Pont_hist[num_t-1],x_grid_hist,10**4)
    print('wdist_weak_Pont_hist',wdist_weak_Pont_hist)
    wdist_true_weak_hist=Wd(mu_weak_hist[num_t-1],x_grid_hist,mu_T_hist[num_t-1],x_grid_hist,10**4)
    print('wdist_true_weak_hist',wdist_true_weak_hist)
    wdist_true_Pont_hist=Wd(mu_T_hist[num_t-1],x_grid_hist,mu_Pont_hist[num_t-1],x_grid_hist,10**4)
    print('wdist_true_Pont_hist',wdist_true_Pont_hist)
    # wdist_true_start_true_solut_hist=Wd(true_solution[num_t-1],x_grid,true_start[num_t-1],x_grid,10**4)
    # print('wdist_true_start_true_solut',wdist_true_start_true_solut)
    # wdist_true_start_Pont=Wd(true_start[num_t-1],x_grid,Pont[num_t-1],x_grid,10**4)
    # print('wdist_true_start_Pont',wdist_true_start_Pont)
    wdist_weak_trunc_true_hist=Wd(mu_weak_trunc_hist[num_t-1],x_grid_hist,mu_T_hist[num_t-1],x_grid_hist,10**4)
    print('wdist_weak_trunc_true_hist',wdist_weak_trunc_true_hist)