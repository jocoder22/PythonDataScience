import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from collections import defaultdict


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

print2(list(data.index))


# Instantiate the English model: nlp
nlp = spacy.load(largge)


# # calculate similarity
# for ele in data['News_content'][3]:
#     nydict = defaultdict(list)
#     ele_token = nlp(ele)
#     # ele_token2 = nlp(' '.join([str(t) for t in ele_token if not t.is_stop]))
#     print(ele_token)
#     for text in data['News_content'][3]:
#         text_token = nlp(text)
#         text_token2 = nlp(' '.join([str(t) for t in text_token if not t.is_stop]))        
#         print2(ele_token2.similarity(text_token2))

# search_doc = nlp("This was very strange argument between american and british person")
# main_doc = nlp("He was from Japan, but a true English gentleman in my eyes, and another one of the reasons as to why I liked going to school.")

# search_doc_no_stop_words = nlp(' '.join([str(t) for t in search_doc if not t.is_stop]))
# main_doc_no_stop_words = nlp(' '.join([str(t) for t in main_doc if not t.is_stop]))

# print(search_doc_no_stop_words.similarity(main_doc_no_stop_words))


about_text = ("""Gus Proto is a Python developer currently
            working for a London-based Fintech
            company. He is interested in learning
            Natural Language Processing.""")


about_doc = nlp(about_text)

for token in about_doc:
    if not token.is_stop:
        print (token)

words_all = [token.text for token in about_doc if (not token.is_punct) and (not token.is_stop)]
words_all = ' '.join(str(t) for t in words_all)
# about_no_stopword_doc = nlp(' '.join([str(t) for t in words_all if not t.is_stop]))
# about_no_stopword_doc = ' '.join(str(t) for t in words_all if not t.is_stop)
# print (about_no_stopword_doc)
print2(words_all)