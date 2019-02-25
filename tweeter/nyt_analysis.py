#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import string

excludePunt = set(string.punctuation)
excludePunt.update(('"', "'"))
stopword = set(stopwords.words("english"))
stopword.update(("said", "to", "th", "e", "cc", "subject", "http", "from", "new", "time", "times", "york",
             "sent", "ect", "u", "fwd", "w", "n", "s", "www", "com", "de", "one", "may", "home",
             "advertisement", "information", "service", "—", "year", "would"))
wordlemm = WordNetLemmatizer()
wordporter = SnowballStemmer("english")


sp = '\n\n'
path = "C:\\Users\\Jose\\Desktop\\PythonDataScience\\tweeter\\"
os.chdir(path)

data = pd.read_csv('nyt2.csv')
print(data.shape)

# instantiate a sentiment analyser
sid = SentimentIntensityAnalyzer()

# score the data
data['scores'] = data.News_content.apply(sid.polarity_scores).apply(lambda x: x['compound'])


# Define word cleaning function
def cleantext(text, stop):
    text = str(text).rstrip()
    stopfree = " ".join([word for word in text.lower().split() if (
        (word not in stopword) and (not word.isdigit()))])

    puncfree = ''.join(word for word in stopfree if word not in excludePunt)
    lemmy = " ".join(wordlemm.lemmatize(word)
                          for word in puncfree.split())
    # result = " ".join(wordporter.stem(word) for word in lemmy.split())
    return lemmy


text_clean = []

for text in data['News_content']:
    text_clean.append(cleantext(text, stopword).split()) 

data['wordList'] = text_clean

kdict = dict()

# Most common used word
for idx, val in data['wordList'].iteritems():
    for i in val:
        if i in kdict:
            kdict[i] += 1
        else:
            kdict[i] = 1
    
ksort = sorted(kdict.items(), reverse=True, key=lambda x: x[1])
ks = pd.DataFrame(ksort,  columns=['word', 'freq'])

print(ksort[:50], ks.head(), sep=sp)
print(data.head(), data['wordList'], sep=sp)

print(data.loc[:,['scores']].tail(), data.scores.mean(), data.describe(), sep=sp)


dfsent = data['scores']
dfsent.plot()
plt.ylabel('Sentiment Score')
plt.xlabel('NYT News Section')
plt.show()


# plt.hbar(ks.word.iloc[:20],ks.freq.iloc[:20])
ks.iloc[:20].plot('word', 'freq', kind='barh')
plt.ylabel(" ")
plt.xlabel('word frequency')
plt.show()