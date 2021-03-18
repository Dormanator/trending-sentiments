import re
import string

import pandas as pd
import numpy as np


class TransformService:
    """Handles the transformation of data to formats expected by app"""

    # Convert Twitter JSON response to dataframe with proper columns and types
    def convert_json_to_dataframe(self, json_data):
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

    # Remove whitespace between text, urls, punctuation, trailing whitespace
    def clean_tweet(self, tweet):
        result = re.sub(r'\s+', ' ', tweet)
        result = re.sub(r"https?://[A-Za-z0-9./]+", ' ', result)
        return result \
            .translate(str.maketrans('', '', string.punctuation)) \
            .strip()

    # Map sentiment scores to text labels
    def map_sentiment_label(self, score):
        category = 'Neutral'
        if score == 0:
            category = 'Negative'
        elif score == 2:
            category = 'Positive'
        return category

    # Map time period sample took place over to interaction labels
    def map_interaction_label(self, time_delta):
        interaction_level = 'Very Low'
        if pd.Timedelta("12 hours") <= time_delta < pd.Timedelta("1 days"):
            interaction_level = 'Low'
        elif pd.Timedelta("4 hours") <= time_delta < pd.Timedelta("12 hours"):
            interaction_level = 'Medium'
        elif pd.Timedelta("2 hours") <= time_delta < pd.Timedelta("4 hours"):
            interaction_level = 'High'
        elif time_delta < pd.Timedelta("2 hours"):
            interaction_level = 'Very High'
        return interaction_level

    # Generate a dataframe with sentiment time series formatted for use in a altair chart
    def gen_sentiment_by_time_dataframe(self, dataframe):
        dataframe = dataframe.copy()
        # Get counts per sentiment level for every timestamp to the minute
        # Df with shape: created_at           Negative  Neutral   Positive
        #                2000-01-01 12:34:00  1         0         2
        tweets_by_sentiment = dataframe.groupby(dataframe['created_at'].map(lambda x: x.replace(second=0)))['sentiment_text']\
            .value_counts()\
            .unstack(fill_value=0)\
            .reset_index()
        # Build tweet frequency by sentiment time series dataframe
        # Df with shape: Created              Tweets    Sentiment
        #                2000-01-01 12:34:00  2         Positive
        #                2000-01-01 12:34:00  1         Negative
        time_and_sentiment = np.empty(shape=[0, 3])
        for sentiment in ['Negative', 'Neutral', 'Positive']:
            temp_df = tweets_by_sentiment[['created_at', sentiment]].copy()
            temp_df['Sentiment'] = sentiment
            temp_df['Sentiment'] = temp_df['Sentiment'].astype('category')
            time_and_sentiment = np.vstack((time_and_sentiment, temp_df.to_numpy()))
        return pd.DataFrame(time_and_sentiment, columns=['Created', 'Tweets', 'Sentiment'])
