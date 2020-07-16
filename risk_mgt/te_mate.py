#!/usr/bin/env python
import numpy as np
import pandas as pd
from printdescribe import print2


bmask1 = [100, 90, 104, 124, 161, 186, 204, 235, 258, 271, 339, 254, 216, 216, 238, 262, 275]
bmask2 = [100,  93, 110, 136, 182, 216, 245, 291, 330, 359.7, 460, 355, 311, 321, 364, 413, 447]
fund = [100,  91, 104, 127, 167, 190, 206, 234, 260, 271, 346, 256, 221, 223, 243, 262, 273] 

data = pd.DataFrame({"bmask1":bmask1, "bmask2":bmask2, "fund":fund})
data2 = data.pct_change().dropna()
data2["Ret1"] = data2.fund - data2.bmask1
data2["Ret2"] = data2.fund - data2.bmask2
mate = np.sqrt(pow(returns, 2).sum()/data2.shape[0])
print2(data, data2, returns, round(mate*100, 2), sep="\n\n")

