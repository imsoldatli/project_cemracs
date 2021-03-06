__author__ = 'angiuli'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math
from matplotlib.legend_handler import HandlerLine2D
from Wd_exact import *

sigma=0.7
nfig=0
value_num_t=np.linspace(10,130,7)

index_anim_num_t=1
anim_num_t=int(value_num_t[index_anim_num_t])


mu_weak_delta_t=[]
mu_weak_trunc_delta_t=[]
mu_Pont_delta_t=[]
mu_true_delta_t=[]

y_weak_delta_t=[]
y_weak_trunc_delta_t=[]
y_Pont_delta_t=[]
y_true_delta_t=[]

z_weak_delta_t=[]
z_weak_trunc_delta_t=[]
z_Pont_delta_t=[]
z_true_delta_t=[]

for k in range(len(value_num_t)):
    #load solutions: mu

    mu_weak_delta_t.append(np.load('/home/christy/Dropbox/CEMRACS_MFG/cluster_results/trader_grid/trader_mu_weak_t'+str(int(value_num_t[k]))+'.npy'))
    mu_weak_trunc_delta_t.append(np.load('/home/christy/Dropbox/CEMRACS_MFG/cluster_results/trader_grid/trader_mu_weak_trunc_t'+str(int(value_num_t[k]))+'.npy'))
    mu_Pont_delta_t.append(np.load('/home/christy/Dropbox/CEMRACS_MFG/cluster_results/trader_grid/trader_mu_Pont_t'+str(int(value_num_t[k]))+'.npy'))
    mu_true_delta_t.append(np.load('/home/christy/Documents/MFG_git/large_data_trader/mu_true_t'+str(int(value_num_t[k]))+'.npy'))
    #mu_true_delta_t.append(np.load('/Users/zioepa/Dropbox/CEMRACS_MFG/cluster_results/trader_grid/mu_true_hist_t'+str(int(value_num_t[k]))+'.npy'))
    # load solutions: y
    y_Pont_delta_t.append(np.load('/home/christy/Dropbox/CEMRACS_MFG/cluster_results/trader_grid/trader_Y_Pont_t'+str(int(value_num_t[k]))+'.npy'))
    y_true_delta_t.append(np.load('/home/christy/Documents/MFG_git/large_data_trader/trader_Y_solution'+str(int(value_num_t[k]))+'.npy'))
    # load solutions: z
    z_weak_delta_t.append(np.load('/home/christy/Dropbox/CEMRACS_MFG/cluster_results/trader_grid/trader_Z_weak_t'+str(int(value_num_t[k]))+'.npy'))
    z_weak_trunc_delta_t.append(np.load('/home/christy/Dropbox/CEMRACS_MFG/cluster_results/trader_grid/trader_Z_weak_trunc_t'+str(int(value_num_t[k]))+'.npy'))





# intialize counters of matching points between the solutions
# compare_weak_pont=0 #compare weak vs pontryagin
# compare_weak_true=0 #compare weak vs true solution
# compare_pont_true=0 #compare pont vs true solution
# #compare_true_start_solution=0 #compare true vs pontryagin started from the true solution
# compare_shift=0







#max_diff_true_start_true_solut=[] # Pontryagin solution started in the true solution vs true solution
#max_diff_true_start_Pont=[] # Pontryagin solution started in the true solution  vs Pontryagin solution

# clean the solutions by all values behind a threshold and save the index of the remaining values
# threshold=10**(-3)
# # weak
# w=[]
# index_w=[]
# # Pontryagin
# P=[]
# index_P=[]
# # true solution
# T=[]
# index_T=[]
#
# row_c=[]
#
# # initializing the mean of each solution
# mean_weak=[]
# mean_Pont=[]
# mean_true_solution=[]
max_diff_y_true_Pont_delta_t=[] # true solution vs Pontryagin solution
max_diff_y_true_Pont_delta_t_plot=[]
max_diff_y_true_z_weak_delta_t=[] # true solution vs Weak solution
max_diff_y_true_z_weak_delta_t_plot=[] # true solution vs Weak solution

#max_diff_y_true_z_weak_trunc_delta_t=[] # true solution vs Weak solution
#max_diff_y_true_z_weak_trunc_delta_t_plot=[] # true solution vs Weak solution

#max_diff_z_weak_trunc_weak_delta_t=[] # Weak solution vs Weak solution truncated
#max_diff_z_weak_trunc_weak_delta_t_plot=[] # Weak solution vs Weak solution truncated

max_diff_z_weak_y_Pont_delta_t=[] # Weak solution vs Pontryagin solution
max_diff_z_weak_y_Pont_delta_t_plot=[] # Weak solution vs Pontryagin solution

#max_diff_z_weak_trunc_y_Pont_delta_t=[]
#max_diff_z_weak_trunc_y_Pont_delta_t_plot=[]


max_diff_y_true_Pont=[] # true solution vs Pontryagin solution
max_diff_y_true_z_weak=[] # true solution vs Weak solution
#max_diff_y_true_z_weak_trunc=[] # true solution vs Weak solution
#max_diff_z_weak_trunc_weak=[] # Weak solution vs Weak solution truncated
max_diff_z_weak_y_Pont=[] # Weak solution vs Pontryagin solution
#max_diff_z_weak_trunc_y_Pont=[]


mse_y_true_Pont_delta_t=[] # true solution vs Pontryagin solution
mse_true_z_weak_delta_t=[] # true solution vs Weak solution
mse_y_true_z_weak_trunc_delta_t=[] # true solution vs Weak solution
#mse_z_weak_trunc_weak_delta_t=[] # Weak solution vs Weak solution truncated
mse_z_weak_y_Pont_delta_t=[] # Weak solution vs Pontryagin solution
#mse_z_weak_trunc_y_Pont_delta_t=[]


# max_diff[t]: max of the differences between two solutions at time t
# max_diff initialization
for k in range(len(value_num_t)):
    num_t=int(value_num_t[k])

    mu_weak=mu_weak_delta_t[k]
    mu_weak_trunc=mu_weak_trunc_delta_t[k]
    mu_Pont=mu_Pont_delta_t[k]
    mu_true=mu_true_delta_t[k]

    num_x=mu_weak.shape[1]



    y_Pont=np.asarray(y_Pont_delta_t[k])
    y_true=np.asarray(y_true_delta_t[k])
    z_weak=np.asarray(z_weak_delta_t[k])
    z_weak_trunc=np.asarray(z_weak_trunc_delta_t[k])

    temp_true=np.zeros(num_x)
    temp_Pont=np.zeros(num_x)
    for i in range(num_t):
        # evaluation of the max_diff divided by E[Z]
        temp_true=np.dot(sigma,y_true[i])
        temp_Pont=np.dot(sigma,y_Pont[i])
        mean_Z_weak=np.dot(mu_true[i],z_weak[i])
        # mean_Z_weak_trunc=np.dot(mu_true[i],z_weak_trunc[i])
        mean_Y_true=np.dot(y_true[i],mu_true[i])

        num_true_Pont=np.max(abs(y_true[i]-y_Pont[i]))
        max_diff_y_true_Pont.append(num_true_Pont/mean_Y_true)

        num_weak_true=np.max(abs(z_weak[i]-temp_true))
        max_diff_y_true_z_weak.append(num_weak_true/mean_Z_weak)

        #num_weak_trunc_true=np.max(abs(z_weak_trunc[i]-temp_true))
        #max_diff_y_true_z_weak_trunc.append(num_weak_trunc_true/mean_Z_weak_trunc)

        #num_weak_weak_trunc=np.max(abs(z_weak[i]-z_weak_trunc[i]))
        #max_diff_z_weak_trunc_weak.append(num_weak_weak_trunc/mean_Z_weak_trunc)

        num_weak_Pont=np.max(abs(z_weak[i]-temp_Pont))
        max_diff_z_weak_y_Pont.append(num_weak_Pont/mean_Z_weak)

        #num_weak_trunc_Pont=np.max(abs(z_weak_trunc[i]-temp_Pont))
        #max_diff_z_weak_trunc_y_Pont.append(num_weak_trunc_Pont/mean_Z_weak)

    max_diff_y_true_Pont_delta_t.append(max_diff_y_true_Pont)# true solution vs Pontryagin solution
    max_diff_y_true_Pont_delta_t_plot.append(max_diff_y_true_Pont[num_t-1])
    max_diff_y_true_z_weak_delta_t.append(max_diff_y_true_z_weak) # true solution vs Weak solution
    max_diff_y_true_z_weak_delta_t_plot.append(max_diff_y_true_z_weak[num_t-1]) # true solution vs Weak solution

    #max_diff_y_true_z_weak_trunc_delta_t.append(max_diff_y_true_z_weak_trunc) # true solution vs Weak solution
    #max_diff_y_true_z_weak_trunc_delta_t_plot.append(max_diff_y_true_z_weak_trunc[num_t-1]) # true solution vs Weak solution

    #max_diff_z_weak_trunc_weak_delta_t.append(max_diff_y_true_z_weak) # Weak solution vs Weak solution truncated
    #max_diff_z_weak_trunc_weak_delta_t_plot.append(max_diff_y_true_z_wea[num_t-1]k) # Weak solution vs Weak solution truncated

    max_diff_z_weak_y_Pont_delta_t.append(max_diff_z_weak_y_Pont) # Weak solution vs Pontryagin solution
    max_diff_z_weak_y_Pont_delta_t_plot.append(max_diff_z_weak_y_Pont[num_t-1]) # Weak solution vs Pontryagin solution
    #max_diff_z_weak_trunc_y_Pont_delta_t.append(max_diff_z_weak_trunc_y_Pont)
    #max_diff_z_weak_trunc_y_Pont_delta_t_plot.append(max_diff_z_weak_trunc_y_Pont[num_t-1])


        #max_diff_true_start_true_solut.append(np.max(abs(true_solution[i]-true_start[i])))
        #max_diff_true_start_Pont.append(np.max(abs(Pont[i]-true_start[i])))

        # # cleaning the solutions & counting matching points
        # row_iw=[]
        # row_w=[]
        # row_P=[]
        # row_iP=[]
        # row_T=[]
        # row_iT=[]
        #
        # for j in range(col):
        #     if weak[i][j]>5*10**-2:
        #         row_w.append(weak[i][j])
        #         row_iw.append(j)
        #     if Pont[i][j]>5*10**-2:
        #         row_P.append(Pont[i][j])
        #         row_iP.append(j)
        #     if true_solution[i][j]>10**-2:
        #         row_T.append(true_solution[i][j])
        #         row_iT.append(j)
        #
        #     if abs(weak[i][j]-Pont[i][j])<10**(-2):
        #         compare_weak_pont=compare_weak_pont+1
        #     if abs(weak[i][j]-true_solution[i][j])<10**(-2):
        #         compare_weak_true=compare_weak_true+1
        #     if abs(Pont[i][j]-true_solution[i][j])<10**(-2):
        #         compare_pont_true=compare_pont_true+1
        # mean_weak.append(np.dot(x_grid,weak[i]))
        # mean_Pont.append(np.dot(x_grid,Pont[i]))
        # mean_true_solution.append(np.dot(x_grid,true_solution[i]))
        #
        # w.append(row_w)
        # index_w.append(row_iw)
        # P.append(row_P)
        # index_P.append(row_iP)
        # T.append(row_T)
        # index_T.append(row_iT)




    result_Pw=np.zeros(num_t)
    result_Pwt=np.zeros(num_t)
    result_tP=np.zeros(num_t)
    result_tw=np.zeros(num_t)
    result_twt=np.zeros(num_t)
    result_wwt=np.zeros(num_t)
    for t in range(num_t):
        Pont=np.multiply(sigma,y_Pont[t])
        square_P_w=np.power(Pont-z_weak[t],2)
        result_Pw[t]=np.dot(square_P_w,mu_true[t])

        # square_P_wt=np.power(Pont-z_weak_trunc[t],2)
        #result_Pwt[t]=np.dot(square_P_wt,mu_true[t])

        square_t_P=np.power(y_Pont[t]-y_true[t],2)
        result_tP[t]=np.dot(square_t_P,mu_true[t])

        true=np.multiply(sigma,y_true[t])
        square_t_w=np.power(z_weak[t]-true,2)
        result_tw[t]=np.dot(square_t_w,mu_true[t])

        square_t_wt=np.power(z_weak_trunc[t]-true,2)
        result_twt[t]=np.dot(square_t_wt,mu_true[t])

        #square_w_wt=np.power(z_weak[t]-z_weak_trunc[t],2)
        #result_wwt[t]=np.dot(square_w_wt,mu_true[t])

        #print('2norm_y_sigma_Pont_z_weak',result_w)
        #print('max_diff_y_sigma_Pont_z_weak_trunc',result_wt)
    plot_mse=int(num_t/2)
    mse_y_true_Pont_delta_t.append(result_tP[plot_mse]) # true solution vs Pontryagin solution
    #mse_true_z_weak_delta_t.append(result_tw[plot_mse]) # true solution vs Weak solution
    #mse_y_true_z_weak_trunc_delta_t.append(result_twt[plot_mse]) # true solution vs Weak solution
    #mse_z_weak_trunc_weak_delta_t.append(result_wwt[num_t-1]) # Weak solution vs Weak solution truncated
    #mse_z_weak_y_Pont_delta_t.append(result_Pw[plot_mse]) # Weak solution vs Pontryagin solution
    #mse_z_weak_trunc_y_Pont_delta_t.append(result_Pwt[num_t-1])
#### Plot max_diff
# print max diff
#plt.axes(xlim=(0, 1), ylim=(0,0.6))

print('max diff: y_Pont vs y_true',max_diff_y_true_Pont_delta_t)
plt.figure(nfig)
nfig=nfig+1
plt.plot(value_num_t,max_diff_y_true_Pont_delta_t_plot,'o',label='y_Pont vs y_true')
plt.xlabel('number of time steps')
plt.ylabel('max difference')
plt.legend()
plt.savefig('max_diff_true_Pont.eps')

print('max diff: z_weak vs y_true',max_diff_y_true_z_weak_delta_t)
plt.figure(nfig)
nfig=nfig+1
plt.plot(value_num_t,max_diff_y_true_z_weak_delta_t_plot,'o',label='z_weak vs y_true')
plt.xlabel('number of time steps')
plt.ylabel('max difference')
plt.legend()
plt.savefig('max_diff_weak_true')


#print('max diff: z_weak_trunc vs y_true',max_diff_y_true_z_weak_trunc_delta_t)
#plt.plot(value_num_t,max_diff_y_true_z_weak_trunc_delta_t_plot,'o',label='z_weak_trunc vs y_true')
print('max diff: y_Pont vs z_weak',max_diff_z_weak_y_Pont_delta_t)
plt.figure(nfig)
nfig=nfig+1
plt.plot(value_num_t,max_diff_z_weak_y_Pont_delta_t_plot,'o',label='y_Pont vs z_weak')



#print('max diff: y_Pont vs z_weak_trunc',max_diff_z_weak_trunc_y_Pont_delta_t)
#plt.plot(value_num_t,max_diff_z_weak_trunc_y_Pont_delta_t_plot,'o',label='y_Pont vs z_weak_trunc')
#print('max diff: z_weak vs z_weak_trunc',max_diff_z_weak_trunc_weak_delta_t)
#plt.plot(value_num_t,max_diff_z_weak_trunc_weak_delta_t_plot,'o',label='z_weak vs z_weak_trunc')

plt.xlabel('number of time steps')
plt.ylabel('max difference')
#plt.title('max of the differences between two solutions at time 0 wrt num_t')

plt.legend()
plt.show
plt.savefig('max_diff_Pont_weak')

### Plot MSE
plt.figure(nfig)
nfig=nfig+1
#ax1=plt.axes(xlim=(0,130),ylim=(0,0.17))
thing2, =plt.plot(value_num_t[1:7],mse_y_true_Pont_delta_t[1:7],'o',label='Pontryagin',color='blue')
#plt.plot(value_num_t,mse_true_z_weak_delta_t,'o',label='z_weak vs y_true')
#thing1,=plt.plot(value_num_t,mse_y_true_z_weak_trunc_delta_t,'o',label='Weak truncated')
#plt.plot(value_num_t,mse_z_weak_y_Pont_delta_t,'o',label='mse y_Pont vs z_weak')
#plt.plot(value_num_t,mse_z_weak_trunc_y_Pont_delta_t,'o',label='y_Pont vs z_weak_trunc')
#plt.plot(value_num_t,mse_z_weak_trunc_weak_delta_t,'o',label='z_weak vs z_weak_trunc')

plt.xlabel('number of time steps')
plt.ylabel('Mean Square Error of Controls')
#plt.title('mean square error between two solutions at time 0 wrt num_t')

plt.legend(bbox_to_anchor=(0,1,1,.102),loc=3)
#
plt.show
plt.savefig('MSE_Pont.eps')

# print a specific time of the cleaned vectors
# print('w',w[4])
# print('P',P[4])
# print('T',T[4])
# print('index_w',index_w[4])
# print('index_P',index_P[4])
# print('index_T',index_T[4])
#
#
# # print mean
# print('mean_w',mean_weak)
# print('mean_P',mean_Pont)
# print('mean_T',mean_true_solution)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# animation: plot all the solutions evolving in time




# construct x_grid (it will be necessary for the animation and to calculate the mean)
T=1
mu_weak=mu_weak_delta_t[index_anim_num_t]
mu_weak_trunc=mu_weak_trunc_delta_t[index_anim_num_t]
mu_Pont=mu_Pont_delta_t[index_anim_num_t]
mu_true=mu_true_delta_t[index_anim_num_t]


num_t=int(anim_num_t)
delta_t=(T-0.06)/(num_t-1)
t_grid=np.linspace(0.06,T,num_t)
delta_x=delta_t**(2)
x_min=-2
x_max=4
num_x=int((x_max-x_min)/delta_x+1)
x_grid=np.linspace(x_min,x_max,num_x)

num_bins=int(num_x/15)
print('num_x',num_x)
num_x_hist=int(num_x/num_bins)
delta_x_hist=np.abs(x_max-x_min)/num_x_hist
x_grid_hist=np.linspace(x_min,x_max,num_x_hist)
mu_weak_hist=np.zeros((num_t,num_x_hist))
mu_weak_trunc_hist=np.zeros((num_t,num_x_hist))
mu_Pont_hist=np.zeros((num_t,num_x_hist))
mu_true_hist=np.zeros((num_t,num_x_hist))

for t in range(num_t):
    for i in range(num_x/num_bins):
        mu_weak_hist[t,i]=np.sum(mu_weak[t,num_bins*i:num_bins*(i+1)])
        mu_weak_trunc_hist[t,i]=np.sum(mu_weak_trunc[t,num_bins*i:num_bins*(i+1)])
        mu_Pont_hist[t,i]=np.sum(mu_Pont[t,num_bins*i:num_bins*(i+1)])
        mu_true_hist[t,i]=np.sum(mu_true[t,num_bins*i:num_bins*(i+1)])

# print('sum mu_w_hist',np.sum(mu_weak_hist[num_t-1]))
# print('sum mu_w_trunc_hist',np.sum(mu_weak_trunc_hist[num_t-1]))
# print('sum mu_P_hist',np.sum(mu_Pont_hist[num_t-1]))
# print('sum mu_T_hist',np.sum(mu_T_hist[num_t-1]))

fig=plt.figure(nfig)
nfig=nfig+1
ax1 = plt.axes(xlim=(-1, 3), ylim=(0,1))
line, = ax1.plot([], [],'o')
plotlays, plotcols = [3], ["green","yellow","red","blue"]
labels=['weak','Pontryagin','true solution','weak trunc']
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
    y = mu_true_hist[i]
    x3.append(x)
    y3.append(y)

    x = x_grid_hist
    y = mu_weak_trunc_hist[i]
    #y = mu_Pont_hist[i] #at the moment, we don't have the data for x

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

anim.save('comparison_mu_trader.mp4')



if __name__=='__main__':
    wdist_true_Pont_delta_t=[] # true solution vs Pontryagin solution
    wdist_true_weak_delta_t=[] # true solution vs Weak solution
    #wdist_true_weak_trunc_delta_t=np.zeros(value_num_t) # true solution vs Weak solution
    #wdist_weak_trunc_weak_delta_t=np.zeros(value_num_t) # Weak solution vs Weak solution truncated
    wdist_weak_Pont_delta_t=[] # Weak solution vs Pontryagin solution
    #wdist_weak_trunc_Pont_delta_t=np.zeros(value_num_t)

    # wdist_true_Pont_hist_delta_t=np.zeros(value_num_t) # true solution vs Pontryagin solution
    # wdist_true_weak_hist_delta_t=np.zeros(value_num_t) # true solution vs Weak solution
    # wdist_true_weak_trunc_hist_delta_t=np.zeros(value_num_t) # true solution vs Weak solution
    # wdist_weak_trunc_weak_hist_delta_t=np.zeros(value_num_t) # Weak solution vs Weak solution truncated
    # wdist_weak_Pont_hist_delta_t=np.zeros(value_num_t) # Weak solution vs Pontryagin solution
    # wdist_weak_trunc_Pont_hist_delta_t=np.zeros(value_num_t)
    for k in range(len(value_num_t)):
        num_t=int(value_num_t[k])

        mu_weak=mu_weak_delta_t[k]
        #mu_weak_trunc=mu_weak_trunc_delta_t[k]
        mu_Pont=mu_Pont_delta_t[k]
        mu_true=mu_true_delta_t[k]

        delta_t=(T-0.06)/(num_t-1)
        t_grid=np.linspace(0.06,T,num_t)
        delta_x=delta_t**(2)
        x_min=-2
        x_max=4
        num_x=int((x_max-x_min)/delta_x+1)
        x_grid=np.linspace(x_min,x_max,num_x)

        # num_x=mu_weak.shape[1]
        #
        # y_Pont=y_Pont_delta_t[k]
        # y_true=y_true_delta_t[k]
        #
        # z_weak=z_weak_delta_t[k]
        # z#_weak_trunc=z_weak_trunc_delta_t=[k]

        # wdist_y_true_Pont=np.zeros(num_t) # true solution vs Pontryagin solution
        # wdist_y_true_z_weak=np.zeros(num_t) # true solution vs Weak solution
        # wdist_y_true_z_weak_trunc=np.zeros(num_t) # true solution vs Weak solution
        # wdist_z_weak_trunc_weak=np.zeros(num_t) # Weak solution vs Weak solution truncated
        # wdist_z_weak_y_Pont=np.zeros(num_t) # Weak solution vs Pontryagin solution
        # wdist_z_weak_trunc_y_Pont=np.zeros(num_t)

        wdist_weak_Pont_delta_t.append(Wd(mu_weak[num_t-1],x_grid,mu_Pont[num_t-1],x_grid,10**4))
        print('wdist_weak_Pont'+str(num_t),wdist_weak_Pont_delta_t[k])

        wdist_true_weak_delta_t.append(Wd(mu_weak[num_t-1],x_grid,mu_true[num_t-1],x_grid,10**4))
        print('wdist_true_weak'+str(num_t),wdist_true_weak_delta_t[k])
        wdist_true_Pont_delta_t.append(Wd(mu_true[num_t-1],x_grid,mu_Pont[num_t-1],x_grid,10**4))
        print('wdist_true_Pont'+str(num_t),wdist_true_Pont_delta_t[k])
        #wdist_true_start_true_solut.append(Wd(true_solution[num_t-1],x_grid,true_start[num_t-1],x_grid,10**4))
        #print('wdist_true_start_true_solut',wdist_true_start_true_solut)
        #wdist_true_start_Pont.append(Wd(true_start[num_t-1],x_grid,Pont[num_t-1],x_grid,10**4))
        #print('wdist_true_start_Pont',wdist_true_start_Pont)

        # to uncomment when we have the data
        #wdist_true_weak_trunc_delta_t[k].append(Wd(mu_weak_trunc[num_t-1],x_grid,mu_true[num_t-1],x_grid,10**4))
        #print('wdist_weak_trunc_true',wdist_true_weak_trunc_delta_t[k])
        #wdist_weak_trunc_weak_delta_t[k].append(Wd(mu_weak_trunc[num_t-1],x_grid,mu_weak[num_t-1],x_grid,10**4))
        #print('wdist_true_weak',wdist_weak_trunc_weak_delta_t[k])
        #wdist_weak_trunc_Pont_delta_t[k].append(Wd(mu_weak_trunc[num_t-1],x_grid,mu_Pont[num_t-1],x_grid,10**4))
        #print('wdist_true_weak',wdist_weak_trunc_Pont_delta_t[k])

        # wdist_weak_Pont_hist_delta_t=Wd(mu_weak_hist[num_t-1],x_grid_hist,mu_Pont_hist[num_t-1],x_grid_hist,10**4)
        # print('wdist_weak_Pont_hist',wdist_weak_Pont_hist_delta_t)
        # wdist_true_weak_hist_delta_t=Wd(mu_weak_hist[num_t-1],x_grid_hist,mu_true_hist[num_t-1],x_grid_hist,10**4)
        # print('wdist_true_weak_hist',wdist_true_weak_hist_delta_t)
        # wdist_true_Pont_hist_delta_t=Wd(mu_true_hist[num_t-1],x_grid_hist,mu_Pont_hist[num_t-1],x_grid_hist,10**4)
        # print('wdist_true_Pont_hist',wdist_true_Pont_hist_delta_t)
        # # wdist_true_start_true_solut_hist=Wd(true_solution[num_t-1],x_grid,true_start[num_t-1],x_grid,10**4)
        # # print('wdist_true_start_true_solut',wdist_true_start_true_solut)
        # # wdist_true_start_Pont=Wd(true_start[num_t-1],x_grid,Pont[num_t-1],x_grid,10**4)
        # # print('wdist_true_start_Pont',wdist_true_start_Pont)
        # wdist_weak_trunc_true_hist=Wd(mu_weak_trunc_hist[num_t-1],x_grid_hist,mu_true_hist[num_t-1],x_grid_hist,10**4)
        # print('wdist_weak_trunc_true_hist',wdist_weak_trunc_true_hist)
        # wdist_weak_trunc_Pont_hist=Wd(mu_weak_trunc_hist[num_t-1],x_grid_hist,mu_Pont_hist[num_t-1],x_grid_hist,10**4)
        # print('wdist_weak_trunc_true_hist',wdist_weak_trunc_true_hist)

    plt.figure(nfig)
    nfig=nfig+1
    np.save('wdist_true_Pont_delta_t.npy',wdist_true_Pont_delta_t)
    np.save('wdist_true_weak_delta_t.npy',wdist_true_weak_delta_t)
    np.save('wdist_weak_Pont_delta_t.npy',wdist_true_weak_delta_t)


    plt.plot(value_num_t,wdist_true_Pont_delta_t,'o',label='mu_Pont vs mu_true')
    plt.plot(value_num_t,wdist_true_weak_delta_t,'o',label='mu_weak vs mu_true')
    #plt.plot(value_num_t,wdist_true_weak_trunc_delta_t,'o',label='mu_weak_trunc vs mu_true')
    plt.plot(value_num_t,wdist_weak_Pont_delta_t,'o',label='mu_Pont vs mu_weak')
    #plt.plot(value_num_t,wdist_weak_trunc_Pont_delta_t,'o',label='mu_Pont vs mu_weak_trunc')
    #plt.plot(value_num_t,wdist_weak_trunc_weak_delta_t,'o',label='mu_weak vs mu_weak_trunc')

    plt.xlabel('number of time steps')
    plt.ylabel('Wasserstein distance')
    #plt.title('Wasserstein distance at time T wrt num_t')

    plt.legend()
    plt.show()
    plt.savefig('trader-Wasserstein.eps')









