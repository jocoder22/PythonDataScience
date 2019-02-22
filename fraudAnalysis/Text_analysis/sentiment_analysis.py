#!/usr/bin/env python
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from zipfile import ZipFile
from io import BytesIO

import gensim
from gensim import corpora

import pyLDAvis.gensim as gensimvis
import pyLDAvis

from nltk.sentiment.vader import SentimentIntensityAnalyzer

plt.style.use('ggplot')

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sp = '\n\n'

url2 = 'https://assets.datacamp.com/production/repositories/2162/datasets/94f2356652dc9ea8f0654b5e9c29645115b6e77f/chapter_4.zip'


try:
    response = requests.get(url2, timeout=3)
    response.raise_for_status()
except requests.exceptions.HTTPError as errh:
    print("Http Error:", errh)
except requests.exceptions.ConnectionError as errc:
    print("Error Connecting:", errc)
except requests.exceptions.Timeout as errt:
    print("Timeout Error:", errt)
except requests.exceptions.RequestException as err:
    print("OOps: Something Else ...", err)


# unzip the content
zipp = ZipFile(BytesIO(response.content))

# Dsiplay files names in the zip file
mylist = [filename for filename in zipp.namelist()]


# load the file into pandas dataframe
data = pd.read_csv(zipp.open(mylist[5]))

data['Date'] = pd.to_datetime(data.Date)
data.set_index('Date', inplace=True)

# print(data.head(), data.columns, sep=sp)

# instantiate a sentiment analyser
sid = SentimentIntensityAnalyzer()

# scores = data['content'].apply(sid.polarity_scores)
# sentiment = scores.apply(lambda s: s['compound'])

data['scores'] = data.content.apply(sid.polarity_scores).apply(lambda x: x['compound'])


print(data.loc[:,['scores']].head())

# subset the dataframe for words: sell or buy
dfm = data.scores[data.content.str.contains('sell')].resample('1 d').mean()
dfy = data.scores[data.content.str.contains('buy')].resample('1 d').mean()

# fill NaN with previous scores
dfy.fillna(method='ffill', inplace=True)
dfm.fillna(method='ffill', inplace=True)

# Plot the sentiment over time
dfy.plot()
dfm.plot()
plt.ylabel('Sentiment Score')
plt.xlabel('Time')
plt.legend(('buy', 'sell'))
plt.show()



