#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from zipfile import ZipFile
from io import BytesIO

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import string
import re
plt.style.use('ggplot')





sp = '\n\n'

url2 = 'https://assets.datacamp.com/production/repositories/2162/datasets/94f2356652dc9ea8f0654b5e9c29645115b6e77f/chapter_4.zip'

# download all the zip files
response = requests.get(url2)

# unzip the content
zipp = ZipFile(BytesIO(response.content))

# Dsiplay files names in the zip file
mylist = [filename for filename in zipp.namelist()]


data = pd.read_csv(zipp.open(mylist[5]))

data2 = data.copy()

print(data2.content.head())

data2['newcontent'] = data2['content'].apply(word_tokenize)
# data2['newcontent'] = data2.apply(
    # lambda row: word_tokenize(row['content']), axis=1)
# data2['newcontent'] = data2['newcontent'].rstrip()



excludePunt = set(string.punctuation)
stopword = set(stopwords.words('english'))
stopword.update(("to", "cc", "subject", "http", "from",
             "sent", "ect", "u", "fwd", "www", "com"))
wordlemm = WordNetLemmatizer()
wordporter = SnowballStemmer("english")


# Define word cleaning function
def cleantext(text, stop):
    text = str(text).rstrip()
    stopfree = " ".join([word for word in text.lower().split() if (
        (word not in stopword) and (not word.isdigit()))])

    puncfree = ''.join(word for word in stopfree if word not in excludePunt)
    lemmy = " ".join(wordlemm.lemmatize(word)
                          for word in puncfree.split())
    result = " ".join(wordporter.stem(word) for word in lemmy.split())
    return result


text_clean = []
for text in data['clean_content']:
    text_clean.append(cleantext(text, stopword)) 
print(text_clean[:2])

data2['cleanedcontent'] = text_clean
print(data2[['content', 'cleanedcontent']].head())


