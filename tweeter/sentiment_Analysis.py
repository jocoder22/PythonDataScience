#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import re
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import TweetTokenizer


path = 'C:\\Users\\Jose\\Desktop\\'
os.chdir(path)


sp = '\n\n'

def flatten_tweets(tweets_json):
    tweets = []
    
    # Iterate through each tweet
    for tweet in tweets_json:

        tweet_obj = json.loads(tweet)

        # Store the user screen name in 'user-screen_name'
        tweet_obj['user-screen_name'] = tweet_obj['user']['screen_name']
    
        # Check if this is a 140+ character tweet
        if 'extended_tweet' in tweet_obj:
            # Store the extended tweet text in 'extended_tweet-full_text'
            tweet_obj['extended_tweet-full_text'] = tweet_obj['extended_tweet']['full_text']
    
        if 'retweeted_status' in tweet_obj:
            # Store the retweet user screen name in 'retweeted_status-user-screen_name'
            tweet_obj['retweeted_status-user-screen_name'] = tweet_obj['retweeted_status']['user']['screen_name']

            # Store the retweet text in 'retweeted_status-text'
            tweet_obj['retweeted_status-text'] = tweet_obj['retweeted_status']['text']
    
            if 'extended_tweet' in tweet_obj['retweeted_status']:
                # Store the extended retweet text in 'retweeted_status-extended_tweet-full_text'
                tweet_obj['retweeted_status-extended_tweet-full_text'] = tweet_obj['retweeted_status']['extended_tweet']['full_text']
            
        tweets.append(tweet_obj)
    return tweets

# # Flatten the tweets and store in `tweets`
# tweets = flatten_tweets('tweetmm.json')

# # Create a DataFrame from `tweets`
# ds_tweets = pd.DataFrame('tweets')

# # Print out the first 5 tweets from this dataset
# print(ds_tweets['text'].values[0:5])



url = 'http://www.jocoder22.com/medication/all/JSON'

response = requests.get(url)

js = response.json()

data = pd.DataFrame(js['AllMEDS'])

data['Side_Effects'] = data['Adverse Effect'].apply(lambda x: x.split(','))

kdict = dict()


# Most common side effects
for idx, val in data['Side_Effects'].iteritems():
    for i in val:
        if i in kdict:
            kdict[i] += 1
        else:
            kdict[i] = 1
    
ksort = sorted(kdict.items(), reverse=True, key=lambda x: x[1])

print(ksort[:5])
print(data.head(), data['Side_Effects'], sep=sp)


# Get a single value
max_ad = max(kdict, key=kdict.get)
print(max_ad)


german_text = 'Wann gehen wir Pizza essen? 🍕 Und fährst du mit Über? 🚕 :) (-:'
emoji = "['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"
print(regexp_tokenize(german_text, emoji))