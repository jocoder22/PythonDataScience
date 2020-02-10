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


question = list(schema[schema["Column"] == "JobSatisfaction"]["Question"])[0]
prop = sum(df.JobSatisfaction.isnull()) / df.shape[0]
prop2 = df.JobSatisfaction.isnull().mean()
prop3 = (df.JobSatisfaction.isnull() == False).mean()
print2(question, prop, prop2, prop3)

