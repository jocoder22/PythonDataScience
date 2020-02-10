import numpy as np
import pandas as pd

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")
        
path = r"C:\Users\okigb\Downloads\survey-results-public.csv"
path2 = r"C:\Users\okigb\Downloads\survey-results-schema.csv"

df = pd.read_csv(path)
schema = pd.read_csv(path2)
print2(df.head(), schema.head())

