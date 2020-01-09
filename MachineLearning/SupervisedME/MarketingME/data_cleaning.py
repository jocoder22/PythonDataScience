import pandas as pd 
import numpy as np 

url = "https://assets.datacamp.com/production/repositories/4976/datasets/252c7d50740da7988d71174d15184247463d975c/telco.csv"

data = pd.read_csv(url)

print(data.head())