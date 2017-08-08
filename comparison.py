__author__ = 'angiuli'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
weak=np.load('mu_weak_t20.npy')
Pont=np.load('mu_Pont_t20.npy')
true_solution=np.load('solution_trader.npy')
true_start=np.load('mu_trader_true_start_t20.npy')

compare_weak_pont=0
compare_weak_true=0
compare_pont_true=0
compare_true_start_solution=0

T=1
num_t=20
delta_t=(T-0.06)/(num_t-1)
t_grid=np.linspace(0.06,T,num_t)
delta_x=delta_t**(2)
x_min=-2
x_max=4
num_x=int((x_max-x_min)/delta_x+1)
x_grid=np.linspace(x_min,x_max,num_x)

compare_shift=0
row=len(weak)
col=len(weak[0])
max_diff_weak_Pont=[]
max_diff_true_weak=[]
max_diff_true_Pont=[]
max_diff_true_start_solut=[]
max_diff_true_start_Pont=[]
w=[]
index_w=[]
P=[]
index_P=[]
T=[]
index_T=[]
row_c=[]
mean_w=[]
mean_P=[]
mean_T=[]

for i in range(row):

    max_diff_weak_Pont.append(np.max(abs(weak[i]-Pont[i])))
    max_diff_true_weak.append(np.max(abs(weak[i]-true_solution[i])))
    max_diff_true_Pont.append(np.max(abs(true_solution[i]-Pont[i])))
    max_diff_true_start_solut.append(np.max(abs(true_solution[i]-true_start[i])))
    max_diff_true_start_Pont.append(np.max(abs(Pont[i]-true_start[i])))

    row_iw=[]
    row_w=[]
    row_P=[]
    row_iP=[]
    row_t=[]
    row_it=[]
    #index_w.append([])
#    P.append([])
    #index_P.append([])
    for j in range(col):
        if weak[i][j]>5*10**-2:
            row_w.append(weak[i][j])
            row_iw.append(j)
        if Pont[i][j]>5*10**-2:
            row_P.append(Pont[i][j])
            row_iP.append(j)
        if true_solution[i][j]>10**-2:
            row_t.append(true_solution[i][j])
            row_it.append(j)

        if abs(weak[i][j]-Pont[i][j])<10**(-2):
            compare_weak_pont=compare_weak_pont+1
        if abs(weak[i][j]-true_solution[i][j])<10**(-2):
            compare_weak_true=compare_weak_true+1
        if abs(Pont[i][j]-true_solution[i][j])<10**(-2):
            compare_pont_true=compare_pont_true+1
    mean_w.append(np.dot(x_grid,weak[i]))
    mean_P.append(np.dot(x_grid,Pont[i]))
    mean_T.append(np.dot(x_grid,true_solution[i]))

    w.append(row_w)
    #w[i]=np.concatenate((w[i],row_w))
    index_w.append(row_iw)
    P.append(row_P)
    index_P.append(row_iP)
    T.append(row_t)
    index_T.append(row_it)
max_diff_wP=[]
for i in range(len(w)):
    print(len(w[i]),len(P[i]),len(T[i]))

#    if i<5:
#        max_diff_wP.append(np.max(abs(np.array(w[i])-np.array(P[i]))))
    #print(np.sum(w[i]),np.sum(P[i]))

    #for j in range(len(w[i])):
    #
    #     if abs(w[i][j]-P[i][j])<10**-2:
    #         compare_shift+=1




print('weak vs Pont',max_diff_weak_Pont)
print(max_diff_wP)
print('true vs Pont',max_diff_true_Pont)
print('true vs weak',max_diff_true_weak)
print('true start vs solution',max_diff_true_start_solut)
print('true start vs Pont',max_diff_true_start_Pont)
#print(compare,col*row)
#print(compare_shift)
print('w',w[4])
print('P',P[4])
print('T',T[4])

print('mean_w',mean_w)
print('mean_P',mean_P)
print('mean_T',mean_T)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure()
ax1 = plt.axes(xlim=(-2, 4), ylim=(0,1))
line, = ax1.plot([], [],'o')
plotlays, plotcols = [3], ["green","yellow","red"]
labels=['weak','Pont','true']
lines = []
for index in range(3):
    lobj = ax1.plot([],[],'o',color=plotcols[index],label=labels[index])[0]
    plt.legend()
    lines.append(lobj)


def init():
    for line in lines:
        line.set_data([],[])
    return lines

x1,y1 = [],[]
x2,y2 = [],[]
x3,y3= [],[]

# fake data
frame_num = 20

def animate(i):

    x1,y1 = [],[]
    x2,y2 = [],[]
    x3,y3= [],[]

    x = x_grid
    y = weak[i]
    x1.append(x)
    y1.append(y)

    x = x_grid
    y = Pont[i]
    x2.append(x)
    y2.append(y)

    x = x_grid
    y = true_solution[i]
    x3.append(x)
    y3.append(y)

    xlist = [x1, x2,x3]
    ylist = [y1, y2, y3]

    #for index in range(0,1):
    for lnum,line in enumerate(lines):
        line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately.

    return lines

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frame_num, interval=350, blit=False)


plt.show()

#anim.save('comparison_mu_trader.mp4')
