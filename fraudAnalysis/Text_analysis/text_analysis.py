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

# model for the topics using LdaModel
model = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, 
                                        id2word=dictionary, passes=15)

topics = model.print_topics(num_words=4)
for topic in topics:
    print(topic, sep=sp)


# ds = pyLDAvis.gensim.prepare(model, corpus, dictionary, sort_topics=False)
# pyLDAvis.display(ds)


def get_topic_details(ldamodel, corpus):
    topic_details_df = pd.DataFrame()
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                topic_details_df = topic_details_df.append(pd.Series(
                    [int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
    topic_details_df.columns = ['Dominant_Topic', '% Score', 'Topic_Keywords']
    return topic_details_df


print(get_topic_details(model, corpus).head())

# Add original text to topic details in a dataframe
contents = pd.DataFrame({'Original text': textlist})
topic_details = pd.concat(
    [get_topic_details(model, corpus), contents], axis=1)

# Create flag for text highest associated with topic 3
topic_details['flag'] = np.where(
    (topic_details['Dominant_Topic'] == 3.0), 1, 0)
print(topic_details.head())
