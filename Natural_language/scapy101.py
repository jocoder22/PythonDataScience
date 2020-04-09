import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy


sp = '\n\n'
largge = 'en_core_web_lg'
path = r"D:\PythonDataScience\tweeter"
os.chdir(path)

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")

data = pd.read_csv('nyt2.csv')

text_clean = []

mytext = str()

for text in data['News_content']:
    mytext += text + " "

print2(data.head(), mytext)