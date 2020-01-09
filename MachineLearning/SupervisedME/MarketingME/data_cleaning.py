import sys
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler


pathtk = r"D:\PPP"
sys.path.insert(0, pathtk)
import wewebs

def print2(*args):
    for arg in args:
        print(arg, end='\n\n')
        
scaler = StandardScaler()
  
sp = {"sep":"\n\n", "end":"\n\n"} 

marketing = wewebs.market
data = pd.read_csv(marketing)

print(data.head())

print2(data.dtypes, data.nunique(), data.info(), data.head(), data.shape, data.columns)

cust_id = ["customerID"]
target = ["Churn"]
cat_features = data.nunique()[data.nunique() < 5].keys().tolist()

cat_features.remove(target[0])
num_features = [col for col in data.columns if col not in cust_id + target + cat_features]

print2(cat_features, num_features)

data = pd.get_dummies(data=data[cat_features],  drop_first=True)
print(data.head())