#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer, PorterStemmer
import string 


def xprint(d):
    for arg in (d.head(), d.info(), d.shape, d. columns):
        print(arg, end='\n\n')


def preprocessText(text):
    excludePunt = set(string.punctuation)
    excludePunt.update(('"', "'"))
    stopword = set(stopwords.words("english"))
    stopword.update(("said", "to", "th", "e", "cc", "subject", "http", "from", "new", "time", "times", "york",
                    "sent", "ect", "u", "fwd", "w", "n", "s", "www", "com", "de", "one", "may", "home", "u", "la",
                    "advertisement", "information", "service", "—", "year", "would"))
    wordlemm = WordNetLemmatizer()
    wordporter = SnowballStemmer("english")

    # wordporter = PorterStemmer(ignore_stopwords=False)
    # form word tokens
    text2 = word_tokenize(text)
    
    # Retain alphabetic words: alpha_only
    alpha_only = [t.lower() for t in text2 if t.isalpha()]

    # Remove all stop words: no_stops
    no_stops = [t for t in alpha_only if t not in stopword]

    # Lemmatize all tokens into a new list: lemmatized
    lemmat = [wordlemm.lemmatize(t) for t in no_stops]
    result = " ".join(word for word in lemmat)
    return (result, lemmat)


sp = '\n\n'
path = r'D:\PythonDataScience\MachineLearning\FeatureEngineering'
os.chdir(path)
data = pd.read_csv('textdata.csv', compression='gzip')

# preprocess the text
text_cleanList = []
text_cleanstring = []

for text in data['News_content']:
    text_cleanstring.append(preprocessText(text)[0])
    text_cleanList.extend(preprocessText(text)[1])  

data['cleanText'] = text_cleanstring

print(text_cleanstring, sep='\n\n')
print(text_cleanList, sep='\n\n')


# Most common used word
kdict = defaultdict(int)
for val in text_cleanList:
    kdict[val] += 1
    
ksort = sorted(kdict.items(), reverse=True, key=lambda x: x[1])
ks = pd.DataFrame(ksort,  columns=['word', 'freq'])

xprint(ks)


# Instantiate the sklearn countvectorizer
# and limit the number of features generated
cvect = CountVectorizer(min_df=0.1 ,max_df=0.7)

# fit the sklearn countvectorizer
cvect.fit(data['cleanText'])

# print feature names
print(len(cvect.get_feature_names()), cvect.get_feature_names(), sep=sp, end=sp)

# Transform the text content
News_content_vectorized = cvect.transform(data['cleanText'])

# convert to array
News_content_Varray = News_content_vectorized.toarray()
print(News_content_Varray)

# convert to dataframe
News_content_df = pd.DataFrame(News_content_Varray, 
                     columns=cvect.get_feature_names()).add_prefix('C_')

# concat the data tables
data2 = pd.concat([data, News_content_df], axis=1)



xprint(data2) 
print(data[['cleanText']].head())
