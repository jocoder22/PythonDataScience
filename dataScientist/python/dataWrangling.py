import numpy as np
import pandas as pd

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")

path = r"C:\Users\okigb\Downloads\survey-results-public.csv"

df = pd.read_csv(path)

df2 = df.CousinEducation.value_counts().reset_index()
df2.rename(columns={"index":"Method", "CousinEducation":"Count"}, inplace=True)

print2(df.CousinEducation, df.columns, df.shape, df2.head())

