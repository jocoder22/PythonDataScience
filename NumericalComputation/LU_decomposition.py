#!/usr/bin/env python
import numpy as np
import pandas as pd

sp = {"end":"\n\n", "sep":"\n\n"}

def luDecomp(matt):
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
# luDecomp(mat)
