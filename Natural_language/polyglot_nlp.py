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

# from polyglot.downloader import downloader
# downloader.download("embeddings2.en")
# downloader.download("pos2.en")
# downloader.download("ner2.en")

sp = '\n\n'
path = "C:\\Users\\Jose\\Desktop\\PythonDataScience\\tweeter\\"
# path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\tweeter\\'
os.chdir(path)

data = pd.read_csv('nyt2.csv')


mytext = str()

for text in data['News_content']:
    mytext += text + " "
    
text = Text(mytext)


for sent in text.sentences:
    print(sent, sep=sp)
    for entity in sent.entities:
        print(entity.tag, entity)
        print(" ", sep=sp)
