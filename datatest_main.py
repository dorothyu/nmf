#!/usr/bin/env python

from nmf import basicnmf
import pandas as pd
import numpy as np
import seaborn as sns
from conses import consensus,reorderConsensusMatrix


def main():

    
    V = pd.read_csv('input_small_data_DRUG.zero-one.csv',header = 0, index_col = 0)
    V = V.values
    V_col = list(pd.read_csv('input_small_data_DRUG.zero-one.csv',nrows= 0))[1:]
    rank = 3

    # Calculate W&H by multiple iteration
    (output_W, output_H) = basicnmf(V, rank, 0.001, 50, 1000)
    cons = consensus(V, rank, 100)
    M = pd.DataFrame(cons,columns = V_col,index = V_col)
    M1 = reorderConsensusMatrix(M)
    print("Consensus:\n",cons)
    sns.heatmap(M1)

    
    print ("\n===Output_Matrix===")
    print ("Matrix_W :\n", output_W)
    print ("Matrix_H :\n", output_H)
    print ("Output_V :\n", np.matmul(output_W, output_H))
    
  
if __name__ == '__main__':
    main()
