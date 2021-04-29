#!/usr/bin/env python
import numpy as np
import pandas as pd

sp = {"end":"\n\n", "sep":"\n\n"}


def luDecomp(matt):
    """
        This compute the LU decomposition of matrix using the Doolittle's algorithms.
        It return a numpy array of the Lower and Upper triangular matrix

    """
    w = len(matt)
    uu = [[0] * w for j in range(w)]
    ll = [[0 for i in range(w)] for j in range(w)]


    for j in range(w):
        ll[j][j] = 1
        for i in range(j+1):
            uusum = sum(uu[k][j] * ll[i][k] for k in range(i))
            uu[i][j] = matt[i][j] - uusum
        for i in range(j, w):
            llsum = sum(uu[k][j] * ll[i][k] for k in range(j))
            ll[i][j] = int((matt[i][j] - llsum) / uu[j][j])

    return np.array(ll), np.array(uu)


mat = [[2, -1, -2],
       [-4, 6, 3],
       [-4, -2, 8]]

a, b = luDecomp(mat)
print(a,b, **sp)
print(a@b, **sp)



def luDecomp2(matt):
    """
        This compute the LU decomposition of matrix using the crout's algorithms.
        It return a numpy array of the Lower and Upper triangular matrix

    """
    w = len(matt)
    uu = [[0] * w for j in range(w)]
    ll = [[0 for i in range(w)] for j in range(w)]

    # for j in range(w):
    #     uu[j][j] = 1

    #     for i in range(j, w):
    #         uusum = sum(ll[i][k] * uu[k][j] for k in range(j))
    #         ll[i][j] = int(matt[i][j] - uusum)

    #     for i in range(j+1, w):
    #         llsum = sum(ll[j][k] * uu[k][i] for k in range(j))
    #         uu[i][j] = int((matt[i][j] - llsum)/ ll[j][j])

    for k in range(w):
        uu[k][k] = 1 

        for j in range(k, w):
            sum0 = sum([ll[j][s] * uu[s][k] for s in range(j)]) #range from index 0
            ll[j][k] = matt[j][k] - sum0 #reversed index

        for j in range(k+1, w):
            sum1 = sum([ll[k][s] * uu[s][j] for s in range(j)]) #range from index 0
            uu[k][j] = (matt[k][j] - sum1) / ll[k][k]

    return np.array(ll), np.array(uu)



a, b = luDecomp2(mat)
print(a,b, **sp)
print(a@b)

