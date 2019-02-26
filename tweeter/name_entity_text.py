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
import nltk


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

speech_tag = [nltk.pos_tag(y) for y in sent_token]

sent_chunk = nltk.ne_chunk_sents(speech_tag, binary=True)

for sentence in sent_chunk:
    for word in sentence:
        if hasattr(work, "label") and word.label() == "NE":
            print(word)

# print(mysentences, sent_token, sep=sp)
