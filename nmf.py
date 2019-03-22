from sys import stdout
from time import time
from numpy import random,where,matmul

def initialize(V,rank):
    """
    (Winit, Hinit) = initialize(V, rank)
    Winit, Hinit: initial solution
    
    Matrix V has the size of n rows and m columns
    n = shape[0]
    m = shape[1]
    """
    shape = V.shape 
    Winit = random.random(size = (shape[0],rank))
    Hinit = random.random(size = (rank,shape[1]))
    
    return Winit, Hinit

def nmf(V, rank, tol, timelimit, maxiter):
    """
    (W,H) = nmf(V,Winit,Hinit,tol,timelimit,maxiter)
    W,H: output solution
    tol: tolerance for a relative stopping condition
    timelimit, maxiter: limit of time and iterations
    """
    (Winit, Hinit) = initialize(V, rank)
    
    W = Winit; 
    H = Hinit; 
    initt = time();
    #shape = V.shape
    eps = 1e-16
    for iter in range (1,maxiter):
        pred_V = matmul(W,H)
        error = V-pred_V
        err = 0.0 
        
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                err += error[i,j] * error[i,j]/2
        if err < tol:
            break

        if time()-initt>timelimit: 
            break
        
        # Update H
        WtV = matmul(W.T,V)
        WtW = matmul(W.T,W)
        WtWH = matmul(WtW,H)
        H = H * WtV/WtWH\
        # Update W
        VHt = matmul(V,H.T)
        HHt = matmul(H,H.T)
        WHHt = matmul(W,HHt)
        W = W * VHt/WHHt
    
        W = where(W>0,W,eps)
        H = where(H>0,H,eps)
        
        if iter % 10 == 0:
            stdout.write('.')
    
    return(W,H)
          
