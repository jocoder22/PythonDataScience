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


sp = '\n\n'
path = r'C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\FeatureEngineering'
os.chdir(path)
data = pd.read_csv('textdata.csv', compression='gzip')


# Instantiate the sklearn countvectorizer
# and limit the number of features generated
cvect = CountVectorizer(max_features=20, ngram_range=(2,2), stop_words='english')

# fit the sklearn countvectorizer
cvect.fit(data['News_content'])

# print feature names
print(len(cvect.get_feature_names()), cvect.get_feature_names(), sep=sp, end=sp)


# Transform the text content
News_content_vectorized = cvect.transform(data['News_content'])

# convert to array
News_content_Varray = News_content_vectorized.toarray()
print(News_content_Varray, sep=sp, end=sp)


# convert to dataframe
News_content_df = pd.DataFrame(News_content_Varray, 
                     columns=cvect.get_feature_names()).add_prefix('Ngram_')

# inspect dataframe
print(News_content_df.iloc[0].sort_values(ascending=False), end=sp)

# print 5 top words 
print(News_content_df.sum().sort_values(ascending=False).head(), end=sp)

# concat the data tables
data2 = pd.concat([data, News_content_df], axis=1)


xprint(data2)