#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
from scapy.all import *

import text_preprocessing
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from nltk import word_tokenize, sent_tokenize

sp = '\n\n'
# path = "C:\\Users\\Jose\\Desktop\\PythonDataScience\\tweeter\\"
path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\tweeter\\'
os.chdir(path)

data = pd.read_csv('nyt2.csv')

text_clean = []

mytext = str()

for text in data['News_content']:
    mytext += text + " "


mysentences = sent_tokenize(mytext)
sent_token = [word_tokenize(x) for x in mysentences]


print(mysentences, sent_token, sep=sp)
