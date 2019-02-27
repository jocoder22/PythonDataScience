#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy


sp = '\n\n'
path = "C:\\Users\\Jose\\Desktop\\PythonDataScience\\tweeter\\"
# path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\tweeter\\'
os.chdir(path)

data = pd.read_csv('nyt2.csv')

text_clean = []

mytext = str()

for text in data['News_content']:
    mytext += text + " "
    
# Instantiate the English model: nlp
nlp = spacy.load('en')


for text in data['News_content']:
    spacyd = nlp(text)
    for ent in spacyd.ents:
        print(ent.label_, ent.text)
        print(" ", sep=sp)
