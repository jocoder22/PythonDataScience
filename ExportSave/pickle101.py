import pickle
import os
import pandas as pd

with open(r'D:\PythonDataScience\ensemble\lifeExp.csv') as f:
    mydata = pd.read_csv(f)
    
# save pickle file
# method 1
with open(r'D:\PythonDataScience\ExportSave\lifeExp.pkl', 'wb') as ppk:
    pickle.dump(mydata, ppk)
    
# method 2
