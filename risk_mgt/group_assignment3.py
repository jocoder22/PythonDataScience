#!/usr/bin/env python
# Import required modules for this CRT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from printdescribe import print2, describe2, changepath

# import excel sheets
path = r"D:\Wqu_FinEngr\Portfolio Theory and Asset Pricing\GroupWork"

with changepath(path):
    data = pd.read_excel("GWP_PTAP_Data_2010.10.08.xlsx", skiprows=1, nrows = 13, 
                sheet_name='10 SPDRs and S&P 500', index_col=0)

describe2(data)