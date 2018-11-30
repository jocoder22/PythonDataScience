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


# Calculate freq
vect2 = TfidfVectorizer(use_idf=False, norm='l1')
wordfreq = vect2.fit_transform(sciencenews.data)
wordlist = vect2.get_feature_names()
for n in wordfreq[0].indices:
    print('Word "{}" has frequency {:.3f}'.format(wordlist[n], wordfreq[0, n]))


vect3 = TfidfVectorizer()
wordfit2 = vect3.fit_transform(sciencenews.data)
wordlist2 = vect3.get_feature_names()
for n in wordfit2[0].indices:
    print('Word "{}" has frequency {:.3f}'.format(wordlist2[n], wordfit2[0, n]))


# words combination
# ## Uni-gram
countvect = CountVectorizer(ngram_range=(1, 1), stop_words=[], min_df=1)
wordcountc = countvect.fit_transform(sciencenews.data)
wordlistc = countvect.get_feature_names()
print("Last 10 words on the list = ", wordlistc[-10:])
wordswithNumbers = [wordlistc[n] + "(" + str(wordcountc[0, n]) + ")"
                    for n in wordcountc[0].indices]
print("Last ten words with occurances: ", wordswithNumbers[-10:])


# ## Bi-gram
countvectb = CountVectorizer(ngram_range=(2, 2))
wordcountcb = countvectb.fit_transform(sciencenews.data)
wordlistcb = countvectb.get_feature_names()
print("Last 10 words on the list = ", wordlistcb[-10:])
wordswithNumbersb = [wordlistcb[n] + "(" + str(wordcountcb[0, n]) + ")"
                     for n in wordcountcb[0].indices]
print("Last ten words with occurances: ", wordswithNumbersb[-10:])


# ## Uni-gram and Bi-gram
countvectbu = CountVectorizer(ngram_range=(2, 2))
wordcountcbu = countvectbu.fit_transform(sciencenews.data)
wordlistcbu = countvectbu.get_feature_names()
print("Last 10 words on the list = ", wordlistcbu[-10:])
wordswithNumbersbu = [wordlistcbu[n] + "(" + str(wordcountcbu[0, n]) + ")"
                      for n in wordcountcbu[0].indices]
print("Last ten words with occurances: ", wordswithNumbersbu[-10:])