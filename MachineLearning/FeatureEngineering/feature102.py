import numpy as np 
import pandas as pd 

sp = '\n\n'

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz'
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
colname = 'Age Sex CP Trestbps Chol Fbs Restecg Thalach Exang Oldpeak Slope CA Thal Label'.split()

df = pd.read_csv(url, names=colname, na_values=['na','Na', '?'])
print(df.shape, df.info(), df.head(), sep=sp, end=sp)

df2 = df.copy()

# Drop based on named column, drop rows
df.dropna(subset=['Chol'], inplace=True)
print(df.shape, sep=sp)
print([name for name in colname if name not in df.columns])


# Drop all column with NA
df.dropna(how='any', axis=1, inplace=True)
print(df.head(), df.shape, sep=sp, end=sp)
print([name for name in colname if name not in df.columns])


# fill in values
df2['Chol'] = df2['Chol'].fillna(round(df2['Chol'].mean()))
print(df2.info(), sep=sp, end=sp)



