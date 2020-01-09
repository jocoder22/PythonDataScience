import sys
import pandas as pd 
import numpy as np 

pathtk = r"D:\PPP"
sys.path.insert(0, pathtk)
import wewebs

def print2(*args):
    for arg in args:
        print(arg, end='\n\n')
  
sp = {"sep":"\n\n", "end":"\n\n"} 

marketing = wewebs.market
data = pd.read_csv(marketing)

print(data.head())