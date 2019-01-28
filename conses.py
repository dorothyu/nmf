import nmf
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram,cophenet
from scipy.spatial.distance import squareform,pdist

def consensus(V, rank, nloop):
    """ 
    Calculate consensus matrix for columns of V
    Matrix V has the size of n rows and m columns
    n = shape[0]
    m = shape[1]
    """
    shape = V.shape
    m = shape[1]
    consensus = np.zeros((m,m))
    conn = np.zeros((m,m))
    connac = np.zeros((m,m))

    for l in range(nloop):
        (W,H) = nmf.basicnmf(V, rank, 0.001, 50, 1000)
        conn   = connectivity(H)
        connac = connac + conn

    consensus = connac / float(nloop)
    return consensus

def connectivity(H):
    """ 
    Calculate connectivity matrix
    """
    shape = H.shape

    l = []
    for i in range(shape[1]):
        max_i = 0
        max_v = 0
        for index,v in enumerate(H[:,i]):
            if v > max_v :
                max_v = v
                max_i = index
        l.append(max_i)

    mat1 = np.tile(np.matrix(l),(shape[1],1))
    mat2 = np.tile(np.matrix(l).T,(1,shape[1]))

    return np.array(mat1 == mat2, dtype=int)


def reorderConsensusMatrix(M):
    """
    Reorder the consensus matrix to show the clustering result properly
    """
    M = pd.DataFrame(M)
    Y = 1 - M
    Z = linkage(squareform(Y), method='average')
    ivl = leaves_list(Z)
    ivl = ivl[::-1]
    reorderM = pd.DataFrame(M.values[:, ivl][ivl, :], index=M.columns[ivl], columns=M.columns[ivl])
    return reorderM  

def cophenetic(M):
    """
    Calculate the cophenetic correlation coefficient to assess the quality of clutering
    """
    Z = linkage(M, method='average')
    c, cophe_dist = cophenet(Z,pdist(M))
    return c
    
