#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer 


path = r'C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\FeatureEngineering'
os.chdir(path)
data = pd.read_csv('textdata.csv', compression='gzip')

def xprint(d):
    for arg in (d.head(), d.info(), d.shape, d. columns):
        print(arg, end='\n\n')



# Instantiate the sklearn countvectorizer
cvect = CountVectorizer()

# fit the sklearn countvectorizer
cvect.fit(data['News_content'])

# print feature names
print(cvect.get_feature_names())

xprint(data)


