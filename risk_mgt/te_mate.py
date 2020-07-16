#!/usr/bin/env python
import numpy as np
import pandas as pd
from printdescribe import print2


bmask1 = [100, 90, 104, 124, 161, 186, 204, 235, 258, 271, 339, 254, 216, 216, 238, 262, 275]
bmask2 = [100,  93, 110, 136, 182, 216, 245, 291, 330, 359.7, 460, 355, 311, 321, 364, 413, 447]
fund = [100,  91, 104, 127, 167, 190, 206, 234, 260, 271, 346, 256, 221, 223, 243, 262, 273] 

# put data into dataframe
data = pd.DataFrame({"bmask1":bmask1, "bmask2":bmask2, "fund":fund})

# compute simple returns
data2 = data.pct_change().dropna()
data2['fund2'] = data2.fund

# compute active returns
data2["Ret1"] = data2.fund - data2.bmask1
data2["Ret2"] = data2.fund - data2.bmask2

# compute tracking error
te = data2[["Ret1", "Ret2"]].std()

# alternative computation of active returns
data4 = data2[["fund", "fund2"]].sub([data2.bmask1, data2.bmask2], axis='columns')

# compute tracking error
te2 = data2[["fund", "fund2"]].std()

# compute mean adjusted tracking error
mate = np.sqrt(pow(data2, 2).sum()/data2.shape[0])
mate2 = np.sqrt(pow(data4, 2).sum()/data4.shape[0])

# visualize tracing errors
print2(round(te*100, 2), round(te2*100, 2))

# visualize mean adjusted tracking errors
print2(round(mate*100, 2), round(mate2*100, 2))
