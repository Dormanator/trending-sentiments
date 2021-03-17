import os

import tweepy
import streamlit as st
import pandas as pd


class TwitterService:
    """ Handles access to Twitter API and transformation of response to dataframe"""

    def __init__(self):
        self.api = None

    def connect(self):
        if self.api is None:
            auth = tweepy.AppAuthHandler(os.getenv('TWITTER_KEY'), os.getenv('TWITTER_SECRET_KEY'))
            self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    @st.cache(show_spinner=True)
    def search(self, query):
        results = self.api.search(q=query, count=100, tweet_mode='extended', result_type='recent')
        json_data = [r._json for r in results]
        return self._json_to_dataframe(json_data)

    def _json_to_dataframe(self, json_data):
        cols_to_include = ['id', 'created_at', 'full_text', 'tweet', 'retweet_count', 'favorite_count',
                           'entities.hashtags', 'user.id', 'user.screen_name']
        dataframe = pd.json_normalize(json_data)
        # If tweets exist
        if 'full_text' in dataframe:
            # If re-tweets, get full original tweet and add RT tag
            if 'retweeted_status.full_text' in dataframe:
                dataframe['tweet'] = dataframe['retweeted_status.full_text'].fillna(dataframe['full_text'])
                retweet_mask = ~dataframe['retweeted_status.full_text'].isnull()
                retweet_tags = dataframe.loc[retweet_mask, 'full_text'].apply(lambda s: s.split(':')[0])
                dataframe.loc[retweet_mask, 'full_text'] = retweet_tags + ': ' + dataframe.loc[retweet_mask, 'tweet']
            else:
                dataframe['tweet'] = dataframe['full_text']
            dataframe['created_at'] = pd.to_datetime(dataframe['created_at'])
            return dataframe[cols_to_include]
