from zipfile import ZipFile
from io import BytesIO
import requests
import matplotlib.pyplot as plt
import pandas as pd
Import pandas

# Create file path: file_path
# file_path = 'Summer Olympic medallists 1896 to 2008 - EDITIONS.tsv'

url = 'https://assets.datacamp.com/production/repositories/516/datasets/2d14df8d3c6a1773358fa000f203282c2e1107d6/Summer%20Olympic%20medals.zip'

response = requests.get(url)

# unzip the content

zipp = ZipFile(BytesIO(response.content))
print(zipp.namelist())

mylist = [filename for filename in zipp.namelist()]
mymedal2 = pd.read_csv(zipp.open(mylist[8]), sep='\t')
# mymedal2 = pd.read_csv(zipp.open(file_path), sep='\t')

# Load DataFrame from file_path: editions
editions = pd.read_csv(zipp.open(mylist[8]), sep='\t')
# editions = pd.read_csv(zipp.open(file_path), sep='\t')

ioc_codes = pd.read_csv(zipp.open(mylist[9]))
allmedals = pd.read_csv(zipp.open(mylist[7]), sep='\t', skiprows=4)
