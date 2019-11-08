from nltk import word_tokenize
from nltk.corpus import stopwords
import string 


def print2(*args):
    for memb in args:
        print(memb.head(), memb.info(), memb.describe(), sep = '\n\n', end = '\n\n')


def tokenize(text):

    excludePunt = set(string.punctuation)
    stopword = set(stopwords.words("english"))
    # form word tokens
    text2 = word_tokenize(text)
    
    # Retain alphabetic words: alpha_only
    alpha_only = [t.lower() for t in text2 if t.isalpha()]

    # Remove all punctuation words: 
    wordtokens = [t for t in alpha_only if t not in [excludePunt, stopword]]

    return wordtokens

