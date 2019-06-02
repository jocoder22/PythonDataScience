#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer 

sp = '\n\n'
path = r'C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\FeatureEngineering'
os.chdir(path)
data = pd.read_csv('textdata.csv', compression='gzip')

def xprint(d):
    for arg in (d.head(), d.info(), d.shape, d. columns):
        print(arg, end='\n\n')



# Instantiate the sklearn countvectorizer
# and limit the number of features generated
cvect = CountVectorizer()

# fit the sklearn countvectorizer
cvect.fit(data['News_content'])

# print feature names
print(len(cvect.get_feature_names()), cvect.get_feature_names(), sep=sp, end=sp)

# Transform the text content
News_content_vectorized = cvect.transform(data['News_content'])

# convert to array
News_content_Varray = News_content_vectorized.toarray()
print(News_content_Varray)

xprint(data)


