#!/usr/bin/env python

from numpy import dot
from numpy import random
from nmf import basicnmf
import pandas as pd
import numpy as np
import seaborn as sns
from conses import consensus


def main():

    # Define constants
    Matrix_ORDER = 5
    Matrix_RANK = 5

    # Generate a target Matrix
    origin_W = random.randint(0,9,size=(Matrix_RANK, Matrix_ORDER))
    origin_H = random.randint(0,9,size=(Matrix_ORDER, Matrix_RANK))
    v = dot(origin_W, origin_H) 
    
    # Calculate W&H by multiple iteration
    # def nmf(V,Winit,Hinit,tol,timelimit,maxiter)
    (output_W, output_H) = basicnmf(v, Matrix_RANK, Matrix_ORDER, 0.001, 50, 100)
    cons = consensus(v, Matrix_RANK, Matrix_ORDER,5)
    print("Consensus:\n",cons)
    sns.heatmap(cons)
    
    # Show Results
    print ("\n==Original_Matrix==")
    print ("Matrix_W :\n", origin_W) 
    print ("Matrix_H :\n", origin_H)
    print ("Target_V :\n", v)
        
    print ("\n===Output_Matrix===")
    print ("Matrix_W :\n", output_W)
    print ("Matrix_H :\n", output_H)
    print ("Output_V :\n", dot(output_W, output_H))
    
    
    
    
if __name__ == '__main__':
    main()
