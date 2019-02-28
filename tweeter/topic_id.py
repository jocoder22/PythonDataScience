#!/usr/bin/env python
import spacy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
import nltk
nltk.download('punkt')

import text_preprocessing
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel

sp = '\n\n'
path = "C:\\Users\\Jose\\Desktop\\PythonDataScience\\tweeter\\"
# path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\tweeter\\'
os.chdir(path)

data = pd.read_csv('nyt.csv')

text_clean = []

for text in data['News_content']:
    text_clean.append(text_preprocessing.preprocessText(text))

    

print(text_clean[:3])