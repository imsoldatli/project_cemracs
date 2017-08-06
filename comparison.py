__author__ = 'angiuli'
import numpy as np
weak=np.load('mu_weak.npy')
Pont=np.load('mu_Pont.npy')

compare=0
compare_shift=0
row=len(weak)
col=len(weak[0])
max_diff=[]
w=[]
index_w=[]
index_P=[]
P=[]
row_c=[]

for i in range(row):
    max_diff.append(np.max(abs(weak[i]-Pont[i])))
    row_iw=[]
    row_w=[]
    row_P=[]
    row_iP=[]
    #index_w.append([])
#    P.append([])
    #index_P.append([])
    for j in range(col):
        if weak[i][j]>10**-2:
            row_w.append(weak[i][j])
            row_iw.append(j)


        if Pont[i][j]>10**-2:
            row_P.append(Pont[i][j])
            row_iP.append(j)
        if abs(weak[i][j]-Pont[i][j])<10**(-2):
            compare=compare+1
    w.append(row_w)
    #w[i]=np.concatenate((w[i],row_w))
    index_w.append(row_iw)
    P.append(row_P)
    index_P.append(row_iP)
for i in range(len(w)):
    print(len(w[i]),len(P[i]))
    #print(np.sum(w[i]),np.sum(P[i]))

    #for j in range(len(w[i])):
    #
    #     if abs(w[i][j]-P[i][j])<10**-2:
    #         compare_shift+=1


print(w[row-4],P[row-4])


print(max_diff)
print(compare,col*row)
print(compare_shift)
#print('p',P[row-1])
#print(index_P[row-1])
#print('w',w)
#print(index_w[row-1])

