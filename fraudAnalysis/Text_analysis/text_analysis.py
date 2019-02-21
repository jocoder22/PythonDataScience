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

data['clean_content'] = data['clean_content'].astype('str')

# create a list of the input data
textlist = list(data['clean_content'].str.split())


# Create a dictionary of words and number of occurances
# {"book": 23, "paul": 5}
dictionary = corpora.Dictionary(textlist)

dictionary.filter_extremes(no_below=5, keep_n=50000)

# create a corpus: list of list of tuples of number representing a word and number of occurances
# [[(0,1), (1,1), (2,1)]]  this is a sparse matrix
corpus = [dictionary.doc2bow(text) for text in textlist]



model = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, 
                                        id2word=dictionary, passes=15)


topics = model.print_topics(num_words=4)
for topic in topics:
    print(topic, sep=sp)
