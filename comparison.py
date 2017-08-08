__author__ = 'angiuli'
import numpy as np
weak=np.load('mu_weak_t20.npy')
Pont=np.load('mu_Pont_t20.npy')
true_solution=np.load('solution_trader.npy')

compare_weak_pont=0
compare_weak_true=0
compare_pont_true=0

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
max_diff=[]
max_diff_true_weak=[]
max_diff_true_Pont=[]
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

    max_diff.append(np.max(abs(weak[i]-Pont[i])))
    max_diff_true_weak.append(np.max(abs(weak[i]-true_solution[i])))
    max_diff_true_Pont.append(np.max(abs(true_solution[i]-Pont[i])))

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




print('weak vs Pont',max_diff)
print(max_diff_wP)
print('true vs Pont',max_diff_true_Pont)
print('true vs weak',max_diff_true_weak)
#print(compare,col*row)
#print(compare_shift)
print('w',w[4])
print('P',P[4])
print('T',T[4])

print('mean_w',mean_w)
print('mean_P',mean_P)
print('mean_T',mean_T)




#print('p',P[row-1])
#print(index_P[row-1])
#print('w',w)
#print(index_w[row-1])

