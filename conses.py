import nmf
import numpy as np


def consensus(V, rank, nloop):
    """ 
    Calculate consensus matrix for columns of V
    Matrix V has the size of n rows and m columns
    """
    shape = V.shape
    consensus = np.zeros((shape[1],shape[1]))
    conn = np.zeros((shape[1],shape[1]))
    connac = np.zeros((shape[1],shape[1]))

    for l in range(nloop):
        (W,H) = nmf.basicnmf(V, rank, 0.001, 50, 100)
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
