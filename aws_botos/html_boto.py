import pandas as pd 
import boto3


with open("D:\PythonDataScience\pandas\people.csv") as file:
    df = pd.read_csv(file)

print(df.head())

# Generate an HTML table with no border and selected columns
df.to_html('./people_No_border.html', render_links=True,
           # Keep specific columns only
           columns= "name  age gender".split(), 
           # Set border
           border=0)

# Generate an html table with border and all columns.
df.to_html('./people_border.html',render_links=True,
           border=1)