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

