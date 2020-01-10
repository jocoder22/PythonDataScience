import sys
import os
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer


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


print2(data.dtypes, data.nunique(), data.info(), data.head(), data.shape, data.columns)
print(data.isna().sum())

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
cust_id = ["customerID"]
target = ["Churn"]
cat_features = data.nunique()[data.nunique() < 6].keys().tolist()

cat_features.remove(target[0])
num_features = [col for col in data.columns if col not in cust_id + target + cat_features]


data_dummy = pd.get_dummies(data=data[cat_features],  drop_first=True)
scaled_num = scaler.fit_transform(data[num_features])
num_df = pd.DataFrame(scaled_num, columns=num_features)

features = data_dummy.merge(right = num_df, how="left", left_index=True, right_index=True)

print2(data_dummy.head(), scaled_num[:10], features.head())



# Transform the target variable
lb = LabelBinarizer()
target1 = lb.fit_transform(data[target[0]])
target = pd.DataFrame(target1, columns=["Churn"])


# saving as pickle file
mydir = "D:\PythonDataScience\MachineLearning\SupervisedME\MarketingME"
features.to_pickle(os.path.join(mydir, "features.pkl"))
pd.to_pickle(target, os.path.join(mydir, "target.pkl"))
