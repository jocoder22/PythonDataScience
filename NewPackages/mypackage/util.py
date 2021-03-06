from collections import Counter
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string 



def print2(*args):
    """
    Function that print descriptive summary

    Input: DataFrame

    """
    sp = {"sep":"\n\n", "end":"\n\n"}
    
    for memb in args:
        print(memb.head(), memb.info(), memb.describe(), **sp)



def tokenize(text):
    """
    This is a function to form tokens of text passages:

    Input: Text

    Output: List of words

    """

    excludePunt = set(string.punctuation)
    excludePunt.update(('"', "'"))
    stopword = set(stopwords.words("english"))
    stopword.update(("said", "to", "th", "e", "cc", "subject", "http", "from", "new", "time", "times", "york",
                    "sent", "ect", "u", "fwd", "w", "n", "s", "www", "com", "de", "one", "may", "home", "u", "la",
                    "advertisement", "information", "service", "—", "year", "would"))

    wordlemm = WordNetLemmatizer()
   
    # form word tokens
    text2 = word_tokenize(text)

    # Retain alphabetic words: alpha_only
    alpha_only = [t.lower() for t in text2 if t.isalpha()]

    # Remove all punctuation words: 
    wordtokens = [t for t in alpha_only if t not in stopword and t not in excludePunt]

    # Lemmatize all tokens into a new list: lemmatized
    lemmat = [wordlemm.lemmatize(t) for t in wordtokens]

    # return list of words
    return lemmat




def countwordtokens(counters):
    """
    Function to count number of words in a list

    Input: List of words

    Output: tuple of words and number of occurane

    """

    # sum the counts
    return sum(counters, Counter())


def plotcount(countObject, n_common=6):
    """
    Function to compute most common words

    Input:
        1. countObject: tuple of words and number of occurance
        2. n_common: int

    Output: Histogram 

    """
    
    # Get the most common n words
    topx = countObject.most_common(n_common)

    # Plot top n words
    plot_most_common(topx)

    return topx


def plot_most_common(tops):
    """"
    Function to plot histogram of the most common words

    Input: words

    Output: Histogram

    """

    # form dict from list of tuple
    top_items_dict = dict(tops)

    # create x range values
    xx = range(len(top_items_dict))

    # create y values and y labels
    yvalues =  list(top_items_dict.values())
    ylab = list(top_items_dict.keys())

    # plot bar chart
    plt.figure()
    plt.bar(xx, yvalues, align='center')
    plt.xticks(xx, ylab, rotation='vertical')
    plt.tight_layout()
    plt.show()

