#!/usr/bin/python3.6

#A rendre pour le 23/10 -- laurent.bletzacker@ipsa.fr -- (PDF-.py).zip ou code sur github

import numpy as np
import time
import matplotlib.pyplot as plt

#Exercice 1
def qr_gen(n,A):
    R,Q = np.zeros(shape=(n,n)), np.zeros(shape=(n,n))
    for j in range(0,n):
        for i in range(j):
            R[i,j]=np.vdot(A[:,j],Q[:,i])
        w = A[:,j]
        for k in range(j):
            w=w-R[k,j]*Q[:,k]
        R[j,j] = np.linalg.norm(w)
        Q[:,j] = (1/R[j,j])*w
    return R,Q

#Exercice 2
def resolGS(A, b):
    R,Q = qr_gen(np.shape(A)[0], A)
    Qt=np.transpose(Q)
    Rinv=np.linalg.inv(R)
    return np.dot(np.dot(Rinv,Qt),b)

#Exercice 3
def compare(n):
    taille = [k for k in range(2, n+1)]
    TGS=[]
    for e in taille:
        A=np.random.randint(1,100, size=(e,e))
        b=np.random.randint(1,100, size=(e,1))
        t1=time.time()
        resolGS(A, b)
        t2=time.time()
        TGS.append(t2-t1)
    plt.plot(taille,TGS)
    plt.show()
