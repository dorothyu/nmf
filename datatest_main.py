#!/usr/bin/env python

from nmf import basicnmf
from scipy.cluster.hierarchy import linkage, leaves_list,dendrogram
from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np
import seaborn as sns
from conses import consensus


def main():

    
    V = pd.read_csv('input_small_data_DRUG.zero-one.csv',header = 0, index_col = 0)
    V = V.values
    V_col = list(pd.read_csv('input_small_data_DRUG.zero-one.csv',nrows= 0))[1:]
    rank = 3

    
    # Calculate W&H by multiple iteration
    (output_W, output_H) = basicnmf(V, rank, 0.001, 50, 1000)
    
    # Draw consensus heatmap
    cons = consensus(V, rank, 10)
    M = pd.DataFrame(cons,columns = V_col,index = V_col)
    M1 = reorderConsensusMatrix(M)
    print("Consensus:\n",cons)
    sns.heatmap(M1)

    
    print ("\n===Output_Matrix===")
    print ("Matrix_W :\n", output_W)
    print ("Matrix_H :\n", output_H)
    print ("Output_V :\n", np.matmul(output_W, output_H))
    
    
def reorderConsensusMatrix(M: np.array):
    M = pd.DataFrame(M)
    Y = 1 - M
    Z = linkage(squareform(Y), method='average')
    ivl = leaves_list(Z)
    ivl = ivl[::-1]
    reorderM = pd.DataFrame(M.values[:, ivl][ivl, :], index=M.columns[ivl], columns=M.columns[ivl])
    return reorderM    



if __name__ == '__main__':
    main()
