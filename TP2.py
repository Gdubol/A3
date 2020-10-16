#!/usr/bin/python3.6

#A rendre pour le 23/10 -- laurent.bletzacker@ipsa.fr -- (PDF-.py).zip ou code sur github

import numpy as np
import time
import matplotlib.pyplot as plt

#Exercice 1
def qr_gen(n,A):
    R,Q = np.zeros(shape=(n,n)), np.zeros(shape=(n,n))
    for j in range(0,n):
        w = A[:,j]
        for i in range(j):
            R[i,j]=np.vdot(A[:,j],Q[:,i])
            w=w-R[i,j]*Q[:,i]
        R[j,j] = np.linalg.norm(w)
        Q[:,j] = (1/R[j,j])*w
    return R,Q

#Exercice 2
#def resolGS(A, b):
#    R,Q = qr_gen(np.shape(A)[0], A)
#    Qt=transpose_np(Q)
#    Rinv=np.linalg.inv(R)
#    return np.dot(Rinv,np.dot(Qt,b))
    
def resolGS(A, b):
    R,Q = qr_gen(np.shape(A)[0], A)
    Qt=transpose_np(Q)
    return resoltrigsup(R,np.dot(Qt,b))

def resolDC(A,b):
    L=DecompositionCholesky(A)
    Lt=transpose_np(L)
    Lt_inv, L_inv=np.linalg.inv(Lt), np.linalg.inv(L)
    return np.dot(Lt_inv,np.dot(L_inv,b))

def transpose_np(A):
    B = np.zeros(shape=np.shape(A))
    for i in range(np.shape(A)[0]):
        B[:,i]=A[i,:]
    return B

def resoltrigsup(A, b):
    x=np.zeros(np.shape(b))
    for i in range(n-1, -1, 0):
        somme=0
        for k in range(i+1, n):
            somme += x[k]
        x[i] = (b[i]-somme)/A[i, i]
    return x
    
#Exercice 3
def compare(n):
    taille = [k for k in range(2, n+1)]
    TGS=[]
    TDC=[]
    for e in taille:
        A=np.random.randint(1,100, size=(e,e))
        b=np.random.randint(1,100, size=(e,1))
        t1=time.time()
        resolGS(A, b)
        t2=time.time()
        resolDC(A, b)
        t3=time.time()
        TGS.append(t2-t1)
        TDC.append(t3-t2)
    plt.plot(taille,TGS,color='b')
    plt.plot(taille,TDC,color='r')
    plt.show()

def DecompositionCholesky(A):
    n = np.shape(A)[0]
    L = np.zeros(shape=(n,n))
    for k in range(0,n):
        for i in range(k,n):
            if k == i:
                somme=0
                for j in range(0,k):
                    somme += (L[k,j])**2
                L[i,k]= (A[k,k]-somme)**(1/2)
            else:
                somme=0
                for j in range(0,k):
                    somme += L[k,j]*L[i,j]
                L[i,k]=(A[i,k]-somme)/L[k,k]
    return L
