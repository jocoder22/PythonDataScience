#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import pickle
import pyLDAvis.gensim

from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.ldamodel import LdaModel
from text_preprocessing import cleantext as pptext


sp = '\n\n'
path = "C:\\Users\\Jose\\Desktop\\PythonDataScience\\tweeter\\"
# path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\tweeter\\'
os.chdir(path)

data = pd.read_csv('nyt.csv')

text_clean = []

for text in data['News_content']:
    text_clean.append(pptext(text).split())


print(text_clean[:3])


dictionary = Dictionary(text_clean)
corpus = [dictionary.doc2bow(text) for text in text_clean]
pickle.dump(corpus, open('topicModels//corpus2.pkl', 'wb'))
dictionary.save('topicModels//dictionary2.gensim')

ldamodel = LdaModel(corpus, num_topics = 15, id2word=dictionary, passes=15)
ldamodel.save('topicModels//model15.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

# Visaulization works only on Jupyter Notebook
# type jupyter notebook or jupyter console
dictionary = Dictionary.load('topicModels//dictionary2.gensim')
corpus = pickle.load(open('topicModels//corpus2.pkl', 'rb'))
ldamd = LdaModel.load('topicModels//model15.gensim')

lda_display = pyLDAvis.gensim.prepare(ldamd, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)