#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader as pdr
from datetime import date
from printdescribe import changepath
import time


data = pd.DataFrame({"PPort":[0.993455, 0.677678, 0.533906, 0.111533, 0.2344567], 
                "Names": ["Kidd", "Opeje", "Ueasd", "Ldldd", "Hdjd"], 
                "Shares":[234, 574, 0, 748, 463]})

print(data.head())

kk = "dict list series split records index".split()

for i in kk:
    print(f"This is for {i}")
    print(data.to_dict(orient=i))
    print("\n\n")