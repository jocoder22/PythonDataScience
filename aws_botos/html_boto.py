import pandas as pd 
import boto3
import os

path = "D:\PythonDataScience\pandas"

os.chdir(path)

df = pd.read_csv("people.csv")

print(df.head())

