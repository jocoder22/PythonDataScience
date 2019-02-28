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

excludePunt = set(string.punctuation)
excludePunt.update(('"', "'"))
stopword = set(stopwords.words("english"))
stopword.update(("said", "to", "th", "e", "cc", "subject", "http", "from", "new", "time", "times", "york",
                 "sent", "ect", "u", "fwd", "w", "n", "s", "www", "com", "de", "one", "may", "home", "u", "la",
                 "advertisement", "information", "service", "—", "year", "would"))
wordlemm = WordNetLemmatizer()
wordporter = SnowballStemmer("english")



# Define word cleaning function
def cleantext(text):
    text = str(text).rstrip()
    stopfree = " ".join([word for word in text.lower().split() if (
        (word not in stopword) and (not word.isdigit()))])
    # if word.isalpha()
    puncfree = ''.join(word for word in stopfree if word not in excludePunt)
    lemmy = " ".join(wordlemm.lemmatize(word)
                     for word in puncfree.split())
    # result = " ".join(wordporter.stem(word) for word in lemmy.split())
    return lemmy


def preprocessText(text):
    from nltk import word_tokenize, wordpunct_tokenize
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.stem.snowball import SnowballStemmer

    excludePunt = set(string.punctuation)
    excludePunt.update(('"', "'"))
    stopword = set(stopwords.words("english"))
    stopword.update(("said", "to", "th", "e", "cc", "subject", "http", "from", "new", "time", "times", "york",
                     "sent", "ect", "u", "fwd", "w", "n", "s", "www", "com", "de", "one", "may", "home", "u", "la",
                     "advertisement", "information", "service", "—", "year", "would"))
    wordlemm = WordNetLemmatizer()
    wordporter = SnowballStemmer("english")
    # form word tokens
    text2 = word_tokenize(text)
    # Retain alphabetic words: alpha_only
    alpha_only = [t.lower() for t in text2 if t.isalpha()]

    # Remove all stop words: no_stops
    no_stops = [t for t in alpha_only if t not in stopword]

    # Lemmatize all tokens into a new list: lemmatized
    lemmat = [wordlemm.lemmatize(t) for t in no_stops]

    return lemmat
