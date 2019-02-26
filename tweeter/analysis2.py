#!/usr/bin/env python
import spacy
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

sp = '\n\n'
# path = "C:\\Users\\Jose\\Desktop\\PythonDataScience\\tweeter\\"
path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\tweeter\\'
os.chdir(path)

data = pd.read_csv('nyt.csv')

text_clean = []

for text in data['News_content']:
    text_clean.append(text_preprocessing.preprocessText(text))

    

print(text_clean[:3])

# Create a Dictionary from the articles: dictionary
dictionary = Dictionary(text_clean)

# Select the id for "abuse": abuse_id
abuse_id = dictionary.token2id.get("abuse")

print(abuse_id, sep=sp)

# Use abuse_id with the dictionary to print the word
print(dictionary.get(abuse_id))

# Create a MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in text_clean]

# Print the first 10 word ids with their frequency counts from the fifth document
print(corpus[4][:10])


# Save the fifth document: doc
doc = corpus[4]

# Sort the doc for frequency: bow_doc
bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)

# Print the top 5 words of the document alongside the count
for word_id, word_count in bow_doc[:5]:
    print(dictionary.get(word_id), word_count)

# Create the defaultdict: total_word_count
total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count

# Create a sorted list from the defaultdict: sorted_word_count
sorted_word_count = sorted(total_word_count.items(),
                           key=lambda w: w[1], reverse=True)

# Print the top 5 words across all documents alongside the count
for word_id, word_count in sorted_word_count[:5]:
    print(dictionary.get(word_id), word_count)


tdf = TfidfModel(corpus)
print(tdf[corpus[1]])


# Calculate the tfidf weights of doc: tfidf_weights
# tfidf_weights = tdf[doc]
tfidf_weights = tdf[corpus[10]]

# Print the first five weights
print(tfidf_weights[:5], sep=sp)

# Sort the weights from highest to lowest: sorted_tfidf_weights
sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

# Print the top 5 weighted words
for term_id, weight in sorted_tfidf_weights[:5]:
    print(dictionary.get(term_id), weight)



# Instantiate the English model: nlp
nlp = spacy.load('en', tagger=False, parser=False, matcher=False)


for text in data['News_content']:
    spacyd = nlp(text)
    for ent in spacyd.ents:
        print(ent.label_, ent.text)
        print(" " , sep=sp)
