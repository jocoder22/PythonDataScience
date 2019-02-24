#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import re


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

print(data.head(), data['Side_Effects'], sep=sp)
