import numpy as np
import pandas as pd
import pandas_datareader.wb as wb
import matplotlib.pyplot as plt

from printdescribe import print2, changepath


data_r = pd.read_excel("greece_quarterly_30Y_reduced_20201102.xlsx", sheet_name="Reduced")