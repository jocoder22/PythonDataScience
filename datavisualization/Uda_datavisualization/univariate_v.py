import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print2(*args):
    for arg in args:
        print(arg, sep="\n\n", end="\n\n")


url = "http://data.insideairbnb.com/united-states/ny/new-york-city/2020-03-13/visualisations/listings.csv"

airbnb = pd.read_csv(url)

print2(airbnb.head(), airbnb.columns)


"""
# Bar chart, for qualitative, categorical data
g = sns.countplot(data=airbnb, x="neighbourhood_group", color=sns.color_palette()[0])
plt.xlabel("NYC Boroughs")
plt.ylabel("Counts")
plt.title("Counts of NYC Airbnb Listing")
plt.show()


# sorted bar plots on counts
bcolor = sns.color_palette()[0]
cat_order = airbnb["neighbourhood_group"].value_counts().index
sns.countplot(data=airbnb, x="neighbourhood_group", color = bcolor, order = cat_order)
plt.xlabel("NYC Boroughs")
plt.ylabel("Counts")
plt.title("Counts of NYC Airbnb Listing")
plt.show()

# sorted bar chart on ordinal categories
# this method requires pandas v0.21 or later
ordering_ = ['Entire home/apt', 'Hotel room', 'Private room', 'Shared room']
cat_ordered = pd.api.types.CategoricalDtype(ordered = True, categories = ordering_)
airbnb['room_type'] = airbnb['room_type'].astype(cat_ordered)

sns.countplot(data=airbnb, x="room_type", color = bcolor)
plt.xlabel("Room Types")
plt.ylabel("Counts")
plt.xticks(rotation = 30)
plt.title("Counts of NYC Airbnb Listing Room Types")
plt.show()

"""
# horizontal bars, pass y values instead, remember to label appropriately
# sorted bar plots on counts
bcolor = sns.color_palette()[0]
plt.figure(facecolor='white')
plt.axes(frameon=False)
cat_order = airbnb["neighbourhood_group"].value_counts().index
g = sns.countplot(data=airbnb, y="neighbourhood_group", color = bcolor, order = cat_order)
plt.ylabel(" ")
plt.xlabel(" ")
# plt.axis('off')
plt.xticks([])
plt.title("Counts of NYC Airbnb Listing among the Boroughs")

for i, v in enumerate(airbnb["neighbourhood_group"].value_counts()):
    plt.text(v, i, " "+str(v), color=bcolor, va='center', fontweight='bold')
    # g.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')

plt.show()



path2 = r"C:\Users\HP\Desktop"
os.chdir(path2)




fig1 = plt.figure(facecolor='white')
ax1 = plt.axes(frameon=False)
# ax1.set_frame_on(False)
ax1.get_xaxis().tick_bottom()
ax1.axes.get_yaxis().set_visible(False)
# ax1.axes.get_xaxis().set_visible(False)
sns.countplot(data=airbnb, x="neighbourhood_group", color = bcolor, order = cat_order)
plt.title("Counts of NYC Airbnb Listing among the Boroughs")
plt.xlabel(" ")

for idx, key in enumerate(airbnb["neighbourhood_group"].value_counts()):
    #    ypoint = key + 390
       plt.text(idx, key , str(key)+"\n", color='black', va='bottom', ha='center', fontweight='bold')

plt.savefig("plot1.png")
plt.show()



"""
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import string



def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    wordporter = SnowballStemmer("english")

    stopword = set(stopwords.words("english"))
    
    # Retain alphabetic words: alpha_only
    alpha_only = [t.lower() for t in tokens if t.isalpha()]

    # Remove all stop words: no_stops
    _tokens = [t for t in alpha_only if t not in stopword]

    no_stop_tokens = [wordporter.stem(word) for word in _tokens]

    

    clean_tokens = []

    # for tok in tokens:
    for tok in no_stop_tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


m = "this is 1 in 200 men: went to bed last night. I saw 4 birds, generously going blue see"
print(tokenize(m))




excludePunt = set(string.punctuation)
stopword = set(stopwords.words('english'))
stopword.update(("to", "th", "e", "cc", "subject", "http", "from",
             "sent", "ect", "u", "fwd", "w", "n", "s", "www", "com"))
wordlemm = WordNetLemmatizer()
wordporter = SnowballStemmer("english")


# Define word cleaning function
def cleantext(text):
    text = str(text).rstrip()

    excludePunt = set(string.punctuation)
    stopword = set(stopwords.words('english'))

    wordlemm = WordNetLemmatizer()
    wordporter = SnowballStemmer("english")

    stopfree = " ".join([word for word in text.lower().split() if (
        (word not in stopword) and (not word.isdigit()))])

    # puncfree = ''.join(word for word in stopfree if word not in excludePunt)
    lemmy = " ".join(wordlemm.lemmatize(word)
                          for word in stopfree.split())
    result = " ".join(wordporter.stem(word) for word in lemmy.split())

    return result

print(cleantext(m))



def preprocessText(text):
    excludePunt = set(string.punctuation)
    excludePunt.update(('"', "'"))
    stopword = set(stopwords.words("english"))
    stopword.update(("said", "to", "th", "e", "cc", "subject", "http", "from", "new", "time", "times", "york",
                    "sent", "ect", "u", "fwd", "w", "n", "s", "www", "com", "de", "one", "may", "home", "u", "la",
                    "advertisement", "information", "service", "—", "year", "would"))
    wordlemm = WordNetLemmatizer()
    wordporter = SnowballStemmer("english")

    # wordporter = PorterStemmer(ignore_stopwords=False)
    # form word tokens
    text2 = word_tokenize(text)
    
    # Retain alphabetic words: alpha_only
    alpha_only = [t.lower() for t in text2 if t.isalpha()]

    # Remove all stop words: no_stops
    no_stops = [t for t in alpha_only if t not in stopword]

    # Lemmatize all tokens into a new list: lemmatized
    lemmat = [wordlemm.lemmatize(t) for t in no_stops]
    result = " ".join(word for word in lemmat)
    return (result, lemmat)

print(preprocessText(m))


"""