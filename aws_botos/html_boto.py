import pandas as pd 
import boto3

import os

print(os.getcwd())

path = r"D:\PythonDataScience\aws_botos\html"

os.chdir(path)

with open("D:\TimerSeriesAnalysis\AMZN.csv") as file:
    df = pd.read_csv(file)

print(df.head())

# Generate an HTML table with no border and selected columns
df.to_html('./people_No_border.html', render_links=True,
           # Keep specific columns only
           columns= "Date,Open,Close,Adj Close,Volume".split(","), 
           # Set border
           border=0)

# Generate an html table with border and all columns.
df.to_html('./people_border.html',render_links=True,
           border=1)