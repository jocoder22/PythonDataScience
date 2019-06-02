#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer


def xprint(d):
    for arg in (d.head(), d.info(), d.shape, d. columns):
        print(arg, end='\n\n')


sp = '\n\n'
path = r'C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\FeatureEngineering'
os.chdir(path)
data = pd.read_csv('textdata.csv', compression='gzip')

# Instantiate the sklearn countvectorizer
# and limit the number of features generated
Tfidvect = TfidfVectorizer(max_features=10, stop_words='english')

# fit the sklearn countvectorizer
Tfidvect .fit(data['News_content'])

# print feature names
print(len(Tfidvect .get_feature_names()), Tfidvect .get_feature_names(), sep=sp, end=sp)

# Transform the text content
News_content_vectorized = Tfidvect .transform(data['News_content'])

# convert to array
News_content_Varray = News_content_vectorized.toarray()
print(News_content_Varray)

# convert to dataframe
News_content_df = pd.DataFrame(News_content_Varray, 
                     columns=Tfidvect .get_feature_names()).add_prefix('Tf_')

# inspect dataframe
print(News_content_df.loc[0].sort_values(ascending=False), end=sp)

# concat the data tables
data2 = pd.concat([data, News_content_df], axis=1)


xprint(data2)

