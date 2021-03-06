import pickle
import os
from glob import glob
from time import sleep
from datetime import date
import pandas as pd

def print2(*args):
    for arg in args:
        print(arg, end='\n\n')
  

sp = {"sep":"\n\n", "end":"\n\n"} 


mydir = r"D:\PythonDataScience\ExportSave"

with open(r'D:\PythonDataScience\ensemble\lifeExp.csv') as f:
    mydata = pd.read_csv(f)
    
# save pickle file
with open(os.path.join(mydir, "lifeExp.pkl"), 'wb') as ppk:
    pickle.dump(mydata, ppk)
    
# loading the pickle file
with open(os.path.join(mydir, "lifeExp.pkl"), 'rb') as ppk:
    mydatanew = pickle.load(ppk)

print2(mydatanew.head())

# Using pandas methods
# save pickle
pd.to_pickle(mydata, os.path.join(mydir, "lifeExp2.pkl"))
# import pickle file
pandas_pickle = pd.read_pickle(os.path.join(mydir, "lifeExp2.pkl"))
print2(pandas_pickle.head())

filespath = os.path.join(mydir, "*.pkl")
filelist1 = glob(filespath, recursive=True)
print2(filelist1, filespath)


for file in filelist1:
    if os.path.exists(file):
        sleep(4)
        os.remove(file)
    else:
        print2("The file does not exist")
        
        

