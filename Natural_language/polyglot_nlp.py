#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import polyglot
from polyglot.text import Text, Word


# parts_by_year = sets[['year', 'num_parts']].groupby(
#     'year', as_index=False).mean()
# # parts_by_year
# # Plot trends in average number of parts by year
# parts_by_year.plot(x='year', y='num_parts')


# colors_summary = colors.groupby('is_trans', as_index=False).count()

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
nlp = spacy.load('en', tagger=False, parser=False, matcher=False)


for text in data['News_content']:
    spacyd = nlp(text)
    for ent in spacyd.ents:
        print(ent.label_, ent.text)
        print(" ", sep=sp)
