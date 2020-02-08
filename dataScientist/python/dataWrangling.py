import numpy as np
import pandas as pd

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")

path = r"C:\Users\okigb\Downloads\survey-results-public.csv"
path2 = r"C:\Users\okigb\Downloads\survey-results-schema.csv"

df = pd.read_csv(path)

df2 = df.CousinEducation.value_counts().reset_index()
df2.rename(columns={"index":"Method", "CousinEducation":"Count"}, inplace=True)

print2(df.CousinEducation, df.columns, df.shape, df2.head())

schema = pd.read_csv(path2)
print2(schema, list(schema[schema.Column == "CousinEducation"]["Question"]))

answerlist = []
for i in range(df2.shape[0]):
    for k in  df2['Method'][i].split("; "):
        if k not in answerlist:
            answerlist.append(k)
            
print2(answerlist)

