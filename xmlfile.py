import os
import pandas as pd 
import matplotlib.pyplot as plt
from lxml import objectify as xl

print(os.getcwd())
os.chdir('C:/Users/.../Code/Section 1')

with open('PopPyramids.xml') as f:
    root = xl.parse(f).getroot()

xmlfile = list()

for entry in root.entry:
    entry_fields = dict()
    for var in entry.var:
        entry_fields[var.attrib['name']] = var.pyval
    xmlfile.append(entry_fields)

xmlfile

xmlfile[0]
len(xmlfile) # 25080
xmlfile[0:2]


# Parsing to list;
xmlfile2 = list()
for entry in root.entry:
    entry_fields = list()
    for var in entry.var:
        entry_fields.append(var.pyval)
    xmlfile2.append(entry_fields)

xmlfile2[1:4]

f.close()

# Form pandas DataFrame;
xmlData = pd.DataFrame(xmlfile)
xmlData.head()

cols = [col for col in xmlData if col not in ['Age', 'Year', 'Country', 'Region']]
index_list = xmlData[['Country', 'Year', 'Age']].values.T.tolist()
xmlData = pd.DataFrame(xmlData[cols].values, columns=cols, index=index_list)



# Reading JSON files;
data = pd.read_json('PopPyramids.json',orient='index').T
data = pd.read_json('PopPyramids.json')