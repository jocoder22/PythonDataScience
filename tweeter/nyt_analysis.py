#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import string

plt.style.use('ggplot')
excludePunt = set(string.punctuation)
excludePunt.update(('"', "'"))
stopword = set(stopwords.words("english"))
stopword.update(("said", "to", "th", "e", "cc", "subject", "http", "from", "new", "time", "times", "york",
             "sent", "ect", "u", "fwd", "w", "n", "s", "www", "com", "de", "one", "may", "home", "u", "la",
             "advertisement", "information", "service", "—", "year", "would"))
wordlemm = WordNetLemmatizer()
wordporter = SnowballStemmer("english")


sp = '\n\n'
# path = "C:\\Users\\Jose\\Desktop\\PythonDataScience\\tweeter\\"
path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\tweeter\\'
os.chdir(path)

data = pd.read_csv('nyt.csv')
print(data.shape)

# instantiate a sentiment analyser
sid = SentimentIntensityAnalyzer()

# score the data
data['scores'] = data.News_content.apply(sid.polarity_scores).apply(lambda x: x['compound'])


# Define word cleaning function
def cleantext(text, stop):
    text = str(text).rstrip()
    stopfree = " ".join([word for word in text.lower().split() if (
        (word not in stop) and (not word.isdigit()))])
        # if word.isalpha()
    puncfree = ''.join(word for word in stopfree if word not in excludePunt)
    lemmy = " ".join(wordlemm.lemmatize(word)
                          for word in puncfree.split())
    # result = " ".join(wordporter.stem(word) for word in lemmy.split())
    return lemmy


def preprocessText(text, stop):
    # form word tokens
    text2 = word_tokenize(text)
    # Retain alphabetic words: alpha_only
    alpha_only = [t.lower() for t in text2 if t.isalpha()]

    # Remove all stop words: no_stops
    no_stops = [t for t in alpha_only if t not in stop]

    # Lemmatize all tokens into a new list: lemmatized
    lemmat = [wordlemm.lemmatize(t) for t in no_stops]
    
    return lemmat

text_clean = []

# for text in data['News_content']:
#     text_clean.append(cleantext(text, stopword).split()) 


for text in data['News_content']:
    text_clean.append(preprocessText(text, stopword))

data['wordList'] = text_clean

data.to_csv('nyt_clean.csv', index=False)

kdict = defaultdict(int)

# Most common used word
for idx, val in data['wordList'].iteritems():
    for i in val:
        kdict[i] += 1
    
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


data['BagsWords'] = data.News_content.apply(word_tokenize)
datawordbags = word_tokenize(data.News_content[2].lower())
allword = " "

for idx, val in data['News_content'].iteritems():
    allword = allword + "  " + val + "  "

# Get Counter object
countdict = Counter(word_tokenize(allword))
print(countdict.most_common(10), sep=sp)
print(Counter(datawordbags).most_common(10))
sortcount = sorted(countdict,  reverse=True)
# print(sortcount, countdict, sep=sp)
print(type(sortcount), type(countdict), sep=sp)


