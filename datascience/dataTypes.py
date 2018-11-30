import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

categoricalFeature = pd.Series(['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                'Friday', 'Saturday', 'Sunday'])

catmapping = pd.get_dummies(categoricalFeature)

catmapping['Friday']
catmapping['Tuesday']


# Using sklearn
label1 = LabelEncoder()
hot1 = OneHotEncoder()
weekdays = ['Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday', 'Sunday']
fit_data = label1.fit_transform(weekdays)
hot1.fit([[fit_data[0]], [fit_data[1]], [fit_data[2]],
          [fit_data[3]], [fit_data[4]]])

print(hot1.transform([label1.transform(['Monday'])]).toarray())
print(hot1.transform([label1.transform(['Tuesday'])]).toarray())


# Working with text
categories = ['sci.med', 'sci.space']
sciencenews = fetch_20newsgroups(categories=categories)
print(sciencenews.data[0])  # content
sciencenews.filenames   # location of file
print(sciencenews.target[0])  # topic


vect = CountVectorizer()
wordcount = vect.fit_transform(sciencenews.data)
wordcount.shape


wordlist = vect.get_feature_names()
for n in wordcount[0].indices:
    print('The word "{}" appears {} times'.format(wordlist[n],
          wordcount[0, n]))


# count the number of words
counter = 0
countlist5 = []
countlist6 = []
countlist7 = []
for i in range(wordcount.shape[0]):
    innercount = 0
    for n in wordcount[i].indices:
        print('The word "{}" appears {} times'.format(wordlist[n],
              wordcount[i, n]))
        innercount += 1
        counter += 1
        
    countlist5.append(counter)
    countlist6.append({i: counter})
    countlist7.append({i: innercount})


print(counter)
countlist5[:10]
countlist6[-10:]
countlist7[-10:]
