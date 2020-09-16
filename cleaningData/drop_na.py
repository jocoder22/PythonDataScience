import numpy as np
import pandas as pd
from printdescribe import print2

# download dataset
url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv"
df = pd.read_csv(url2)

# expore dataset
print2(df.shape, df.dtypes, df.columns, df.info(), df.describe())

# Explore categorical features
for col in [x for x in df.select_dtypes(include=['object'])]:
    print2(df[col].value_counts(ascending=True, dropna=False))
    
    
# Add nan to dataset
num = df.Age.max()
for col in [x for x in df.select_dtypes(include=['object'])]:
    numm = num - 3
    df[col] = np.where(df.Age > numm, np.nan, df[col])
    num = numm

# view the new dataset
print2(df.isnull().sum())
print(df.info())

# explore the categorical datasets
for col in [x for x in df.select_dtypes(include=['object'])]:
    print2(df[col].value_counts(ascending=True, dropna=False))
    

# drop nan row
df.dropna(axis=0)

# At least one or any at all
df.dropna(axis=0, how ="any")

# All or full nan
df.dropna(axis=0, how ="all")

# drop nan columns
df.dropna(axis=1)

# Keep only the col with at least 95% non-NA values.
_95col = int(df.shape[0] * 0.95)
df.dropna(axis=1, thresh=_95col).shape
df.dropna(axis=1, thresh=_95col).columns


# Keep only the row with at least 95% non-NA values.
_95row = int(df.shape[1] * 0.95)
df.dropna(axis=0, thresh=_95row).shape
df.dropna(axis=0, thresh=_95row).index
