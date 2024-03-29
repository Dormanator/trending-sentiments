import re

import numpy as np
import pandas as pd


class TransformerPipeline:
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

    # Methods for cleaning text
    def _remove_links(self, s):
        return re.sub(r'https?://[A-Za-z0-9./]+', '', s)

    def _remove_mentions(self, s):
        return re.sub(r'(@[A-Za-z0-9_]+)', '', s)

    def _remove_hashtags(self, s):
        return re.sub(r'(#[A-Za-z0-9_]+)', '', s)

    def _trim_spaces(self, s):
        result = re.sub(r'\s+', ' ', s)
        return result.strip()

    def clean_tweet(self, tweet):
        result = self._remove_links(tweet)
        result = self._remove_hashtags(result)
        result = self._remove_mentions(result)
        result = self._trim_spaces(result)
        return result

    # Map sentiment scores to text labels
    def map_sentiment_label(self, score):
        category = 'Neutral'
        if score <= -0.05:
            category = 'Negative'
        elif score >= 0.05:
            category = 'Positive'
        return category

    # Map time period sample took place over to interaction labels
    def map_interaction_label(self, time_delta, sample_size):
        # Consider time it takes for each 100 tweets to occur
        normalized_size = sample_size / 100
        normalized_delta = time_delta / normalized_size;
        interaction_level = 'Very Low'
        if pd.Timedelta("12 hours") <= normalized_delta < pd.Timedelta("1 days"):
            interaction_level = 'Low'
        elif pd.Timedelta("4 hours") <= normalized_delta < pd.Timedelta("12 hours"):
            interaction_level = 'Medium'
        elif pd.Timedelta("2 hours") <= normalized_delta < pd.Timedelta("4 hours"):
            interaction_level = 'High'
        elif normalized_delta < pd.Timedelta("2 hours"):
            interaction_level = 'Very High'
        return interaction_level

    # Generate a dataframe with tweet frequency time series formatted for use in a altair chart
    def gen_tweets_by_time_dataframe(self, dataframe):
        dataframe = dataframe[['created_at']].copy()
        # Get counts per sentiment level for every timestamp to the minute
        # Df with shape: Created                Tweets
        #                2000-01-01 12:34:00    2
        #                2000-01-01 12:35:00    3
        tweets_by_sentiment = dataframe.groupby(dataframe['created_at'].map(lambda x: x.replace(second=0))).count()
        return tweets_by_sentiment \
            .rename(columns={"created_at": "Tweets"}) \
            .reset_index() \
            .rename(columns={"created_at": "Created"})

    # Generate a dataframe with sentiment score time series data
    def gen_sentiment_score_by_time_dataframe(self, dataframe):
        dataframe = dataframe[['created_at', 'sentiment_score', 'sentiment_text']].copy()
        # Get every timestamp to the minute and associated sentiment scores
        # Df with shape: Created              Score         Sentiment
        #                2000-01-01 12:34:00  0.23423       Positive
        dataframe['created_at'] = dataframe['created_at']
        return dataframe.rename(columns={
            "created_at": "Created",
            "sentiment_score": "Sentiment Score",
            "sentiment_text": "Sentiment"
        })

    def _flatten_hashtag_arr(self, arr):
        flat_entities = []
        for entity in arr:
            hashtag_entry = [d['text'] for d in entity]
            flat_entities.append(hashtag_entry)
        return [hashtag for sublist in flat_entities for hashtag in sublist]

    def gen_hashtag_counts_dataframe(self, dataframe):
        hashtags = self._flatten_hashtag_arr(dataframe['entities.hashtags'].to_list())
        df_hashtags = pd.DataFrame({'Hashtag': pd.Series(hashtags, dtype='str'),
                                    'Count': pd.Series(np.ones(len(hashtags)), dtype='int')},
                                   columns=['Hashtag', 'Count'])
        # Build dataframe with hashtag counts
        # Df with shape: Hashtag    Count
        #                avatar     54
        #                nfl        32
        return df_hashtags.groupby(df_hashtags['Hashtag'].map(lambda s: s.lower())) \
            .sum() \
            .sort_values(by=['Count'], ascending=False) \
            .reset_index()
