import pickle
import os
from glob import glob
from time import sleep
from datetime import date
import pandas as pd

mydir = r"D:\PythonDataScience\ExportSave"
with open(r'D:\PythonDataScience\ensemble\lifeExp.csv') as f:
    mydata = pd.read_csv(f)
    
# save pickle file
# method 1
with open(r'D:\PythonDataScience\ExportSave\lifeExp.pkl', 'wb') as ppk:
    pickle.dump(mydata, ppk)
    
# method 2
filename = f'D:\PythonDataScience\ExportSave\picke_{date.today()}.pkl'
pickle.dump(mydata, open(filename, 'wb'))




filespath = os.path.join(mydir, "*.pkl")
filelist1 = glob(filespath, recursive=True)
print(filelist1, filespath)

for file in filelist1:
    if os.path.exists(file):
        sleep(4)
        os.remove(file)
    else:
        print("The file does not exist")
