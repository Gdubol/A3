#!/usr/bin/python3.6

#A rendre pour le 23/10 -- laurent.bletzacker@ipsa.fr -- (PDF-.py).zip ou code sur github

import numpy as np
import time
import matplotlib.pyplot as plt
import math
import random

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

def resolDC(A,b): #ajouter resol trigsup
    L=DecompositionCholesky(A)
    try:
        Lt=transpose_np(L)
        Lt_inv=np.linalg.inv(Lt)
        L_inv=np.linalg.inv(L)
        X=np.dot(L_inv,b)
        return np.dot(Lt_inv,X)
    except:
        print('echec')
        return np.linalg.solve(A,b)

def transpose_np(A):
    B = np.zeros(shape=np.shape(A))
    for i in range(np.shape(A)[0]):
        B[:,i]=A[i,:]
    return B

def resoltrigsup(A, b):
    n = np.shape(b)[0]
    x=np.zeros(np.shape(b))
    for i in range(n-1, -1, -1):
        somme=0
        for k in range(i+1, n):
            somme += x[k,0]*A[i,k]
        x[i,0] = (b[i,0]-somme)/A[i, i]
    return x

#Exercice 3
def compare(n):
    taille = [k for k in range(2, n+1)]
    TGS=[]
    TDC=[]
    TG=[]
    eTGS=[]
    eTDC=[]
    eTG=[]
    L_cond=[]

    for e in taille:
        while True:
            l=np.random.rand(e,e)
            At=transpose_np(l)
            A=l+At
            if np.linalg.det(A)!=0:
                break
        b=np.random.rand(e,1)
        A_save = np.copy(A)
        b_save = np.copy(b)
        t1=time.time()
        xGS=resolGS(A, b)
        t2=time.time()
        xDC=resolDC(A, b)
        t3=time.time()
        xG, A_inverse=Gauss(A,b)
        t4=time.time()
        TGS.append(t2-t1)
        TDC.append(t3-t2)
        TG.append(t4-t3)

        eTGS.append(math.log10(np.linalg.norm(abs(np.dot(A_save,xGS)-b_save))))
        eTDC.append(math.log10(np.linalg.norm(abs(np.dot(A_save,xDC)-b_save))))
        eTG.append(math.log10(np.linalg.norm(abs(np.dot(A_save,xG)-b_save))))

        L_cond.append(np.dot(np.linalg.norm(A_inverse),np.linalg.norm(A_save)))

    plt.title("Temps d'éxécution en fonction de la taille de la matrice")
    plt.xlabel("Taille de la matrice")
    plt.ylabel("Temps d'execution")
    plt.plot(taille,TGS,color='b',label='Gram-Schmidt') #bleu
    plt.plot(taille,TDC,color='r',label='Décomposition de Cholesky') #rouge
    plt.plot(taille,TG,color='g',label='Gauss') #vert
    plt.legend()
    plt.show()

    L=[eTGS,eTDC,eTG]
    plt.clf()
    plt.title("Erreur en fonction de la taille de la matrice")
    plt.xlabel("Taille de la matrice")
    plt.ylabel("Erreur")
    plt.plot(taille,L[0],color='b',label='Gram-Schmidt')
    #plt.plot(taille,L[1],color='r',label='Décomposition de Cholesky')
    plt.plot(taille,L[2],color='g',label='Gauss')
    plt.legend()
    plt.show()
    
    plt.clf()
    plt.title("Conditionnement en fonction de la taille de la matrice")
    plt.xlabel("Taille de la matrice")
    plt.ylabel("Conditionnement")
    plt.plot(taille, L_cond,label='conditionnement (pour des matrices aléatoires symétriques)')
    plt.legend()
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
                L[i,k]= (abs(A[k,k]-somme))**(1/2) #racine négative
            else:
                somme=0
                for j in range(0,k):
                    somme += L[k,j]*L[i,j]
                L[i,k]=(A[i,k]-somme)/L[k,k]
    return L
"""
def DecompositionCholesky(A):
    n = np.shape(A)[0]
    L = np.zeros(shape=(n,n))
    for i in range(n):
        for k in range(i+1):
            tmp_sum = sum(L[i,j]*L[k,j] for j in range(k))
            if i==k:
                L[i,k] = math.sqrt(abs(A[i,i]-tmp_sum))
            else:
                L[i,k] = (1.0/L[k,k]*(A[i,k]-tmp_sum))
    return L
"""
def Gaussel(A): #A doit contenir des flottants
    Ag=A.copy()
    n = np.shape(A)[0]
    A_inv = np.eye(n,n)
    r=-1
    for j in range(n):
        k = (max([(i[0], abs(i[1])) for i in enumerate(list(Ag[:,j]))][r+1:n+1], key=lambda e: e[1]))[0]
        if Ag[k,j] != 0:
            r+=1
            A_inv[k,:]=A_inv[k,:]/Ag[k,j]
            Ag[k,:]=Ag[k,:]/Ag[k,j]
            if k!=r:
                b1=A_inv[k,:].copy()
                A_inv[k,:]=A_inv[r,:]
                A_inv[r,:]=b1

                b=Ag[k,:].copy()
                Ag[k,:]=Ag[r,:]
                Ag[r,:]=b
            for i in range(n):
                if i!=r:
                    A_inv[i,:]=A_inv[i,:]-(A_inv[r,:]*Ag[i,j])
                    Ag[i,:]=Ag[i,:]-(Ag[r,:]*Ag[i,j])
    return A_inv

def Gauss(A, b):
    A_inverse=np.copy(Gaussel(A))
    return np.dot(A_inverse,b), A_inverse

####################################################################################################################################################


####################################################################################################################################################
