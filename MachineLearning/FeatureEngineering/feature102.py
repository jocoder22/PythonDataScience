import numpy as np 
import pandas as pd 

sp = '\n\n'

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz'
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
colname = 'Age Sex CP Trestbps Chol Fbs Restecg Thalach Exang Oldpeak Slope CA Thal Label'.split()

df = pd.read_csv(url, names=colname, na_values=['na','Na', '?'])
print(df.shape, df.info(), df.head(), sep=sp, end=sp)


