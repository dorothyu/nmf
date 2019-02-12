#!/usr/bin/env python

from nmf import projnmf
import pandas as pd
import seaborn as sns
from conses import consensus, reorderConsensusMatrix, cophenetic
import matplotlib.pyplot as plt


def main():

    
    #V = pd.read_csv('MergeExpro_contrib1-GPL570.csv',header = 0, index_col = 0)
    V = pd.read_csv('input_small_data_MUT.zero-one.csv',header = 0, index_col = 0)
    V = V.values
    V_col = list(pd.read_csv('input_small_data_MUT.zero-one.csv',nrows= 0))[1:]
    rank = [2]
    cophe = []

    # Calculate W&H by multiple iteration
    for element in rank:
        (output_W, output_H) =projnmf(V, element, 0.001, 50, 1000)
        cons = consensus(V, element, 100)
        M = pd.DataFrame(cons,columns = V_col,index = V_col)
        M1 = reorderConsensusMatrix(M)
        cophe.append(cophenetic(M1))
        #print("\nConsensus:\n",cons)
        sns.clustermap(M1)
    
# =============================================================================
#     plt.figure(figsize=(10,10))
#     plt.plot(rank,cophe,'bo',rank,cophe,'k')
#     plt.xticks(rank)
#     plt.xlabel('Rank')
#     plt.ylabel('Cophenetic correlation coefficient')
#     plt.show()
# =============================================================================
# =============================================================================
#     print ("\n===Output_Matrix===")
#     print ("Matrix_W :\n", output_W)
#     print ("Matrix_H :\n", output_H)
#     print ("Output_V :\n", np.matmul(output_W, output_H))
# =============================================================================
    
  
if __name__ == '__main__':
    main()
