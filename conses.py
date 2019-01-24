import nmf
import numpy as np


def consensus(a, Matrix_RANK, Matrix_ORDER,nloop):
    """ 
    Calculate consensus matrix
    """
    (n,m)     = a.shape
    consensus = np.zeros((m,m))
    conn      = np.zeros((m,m))
    connac = np.zeros((m,m))

    for l in range(nloop):
        (w,h) = nmf.basicnmf(a, Matrix_RANK, 0.001, 50, 100)
        conn   = nmfconnectivity(h)
        connac = connac + conn

    consensus = connac / float(nloop)
    return consensus

def nmfconnectivity(h):
    """ 
    Calculate connective matrix
    """
    (k,m) = h.shape

    ar = []
    for i in range(m):
        max_i = 0
        max_v = 0
        for index,v in enumerate(h[:,i]):
            if v > max_v :
                max_v = v
                max_i = index
        ar.append(max_i)

    mat1 = np.tile(np.matrix(ar),(m,1))
    mat2 = np.tile(np.matrix(ar).T,(1,m))

    return np.array(mat1 == mat2, dtype=int)
