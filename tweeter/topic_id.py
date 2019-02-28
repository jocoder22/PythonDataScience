#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import pickle

from text_preprocessing import preprocessText as pptext
import pickle
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel


sp = '\n\n'
path = "C:\\Users\\Jose\\Desktop\\PythonDataScience\\tweeter\\"
# path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\tweeter\\'
os.chdir(path)

data = pd.read_csv('nyt.csv')

text_clean = []

for text in data['News_content']:
    text_clean.append(pptext(text))

    

print(text_clean[:3])


dictionary = Dictionary(text_clean)
corpus = [dictionary.doc2bow(text) for text in text_clean]
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')