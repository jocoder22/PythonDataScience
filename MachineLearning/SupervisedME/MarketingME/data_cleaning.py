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

print(data.iloc[480:490,:])

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
cust_id = ["customerID"]
target = ["Churn"]
cat_features = data.nunique()[data.nunique() < 5].keys().tolist()

cat_features.remove(target[0])
num_features = [col for col in data.columns if col not in cust_id + target + cat_features]


data_dummy = pd.get_dummies(data=data[cat_features],  drop_first=True)
scaled_num = scaler.fit_transform(data[num_features])
num_df = pd.DataFrame(scaled_num, columns=num_features)

clean_data = data_dummy.merge(right = num_df, how="left", left_index=True, right_index=True)

print2(data_dummy.head(), scaled_num[:10], clean_data.head())
