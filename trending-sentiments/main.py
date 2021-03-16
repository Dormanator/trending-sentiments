import os
import re
import string

import streamlit as st
import altair as alt
import tweepy
import pandas as pd
import numpy as np
import stanza

from dotenv import load_dotenv

load_dotenv()

# Todo: refactor into services - TwitterService, TransformService, SentimentService ??

@st.cache(show_spinner=True)
def load_model():
    stanza.download('en', model_dir='./model')


def connect():
    auth = tweepy.AppAuthHandler(os.getenv('TWITTER_KEY'), os.getenv('TWITTER_SECRET_KEY'))
    return tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


@st.cache(show_spinner=True)
def search(query):
    results = api.search(q=query, count=100, tweet_mode='extended', result_type='recent')
    json_data = [r._json for r in results]
    return json_data


def json_to_dataframe(json_data):
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


def clean_tweet(tweet):
    # Remove whitespace between text, urls, punctuation, trailing whitespace
    result = re.sub(r'\s+', ' ', tweet)
    result = re.sub(r"https?://[A-Za-z0-9./]+", ' ', result)
    return result \
        .translate(str.maketrans('', '', string.punctuation)) \
        .strip()


@st.cache(show_spinner=True)
def predict_sentiment(tweet):
    if tweet:
        doc = nlp(tweet)
        return doc.sentences[0].sentiment
    else:
        return 0


def map_sentiment(score):
    category = 'Neutral'
    if score == 0:
        category = 'Negative'
    elif score == 2:
        category = 'Positive'
    return category


if __name__ == '__main__':
    # Setup Page Title
    st.set_page_config(page_title="Trending Sentiments", page_icon="📈", initial_sidebar_state="expanded", )
    st.markdown(
        """<style>
            table {text-align: left !important}
        </style>
        """, unsafe_allow_html=True)

    # Setup Stanza NLP Model & Twitter API
    load_model()
    nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
    api = connect()

    # Setup Page Header
    st.write("""
    # 📈 Trending Sentiments
    Discover trending sentiments on Twitter with a hashtag or keyword search.
    """)

    # Setup Sidebar
    # Handle user input
    userInput = st.sidebar.text_input('Search for a hashtag or keyword to begin', '#Avatar')
    st.sidebar.write("""
    Created by Ryan Dorman
    """)

    if not userInput:
        st.warning('Please input a search value.')
        st.stop()

    json_tweets = search(userInput)
    df = json_to_dataframe(json_tweets)

    # Predict tweet sentiments using Stanza CNN classifier
    with st.spinner('Analyzing Sentiments...'):
        df['sentiment'] = df['tweet'].map(clean_tweet).map(predict_sentiment)
        df['sentiment_text'] = df['sentiment'].map(map_sentiment)
        df['sentiment_text'].astype('category')
    st.balloons()

    # Start of Page Body
    st.write("""
    ## 100 Most Recent Tweets for:
    """, userInput)

    # Graph of Tweet Sentiment over time
    tweets_by_sentiment = df.groupby(df['created_at'].map(lambda x: x.replace(second=0)))['sentiment_text'] \
        .value_counts() \
        .unstack(fill_value=0) \
        .reset_index()
    # Build tweet frequency by sentiment time series dataframe
    time_and_sentiment = np.empty(shape=[0, 3])
    for sentiment in ['Negative', 'Neutral', 'Positive']:
        temp_df = tweets_by_sentiment[['created_at', sentiment]].copy()
        temp_df['Sentiment'] = sentiment
        temp_df['Sentiment'] = temp_df['Sentiment'].astype('category')
        time_and_sentiment = np.vstack((time_and_sentiment, temp_df.to_numpy()))
    df_time_and_sentiment = pd.DataFrame(time_and_sentiment, columns=['Created', 'Tweets', 'Sentiment'])
    # Graph sentiment time series
    chart_time_and_sentiment = alt.Chart(df_time_and_sentiment).mark_bar().encode(
        x='Created',
        y='sum(Tweets)',
        color=alt.Color('Sentiment',
                        sort=alt.EncodingSortField('Sentiment', order='ascending'),
                        scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative']),
                        legend=alt.Legend(title="Sentiments")
                        ),
        order=alt.Order(
            # Sort the segments of the bars by this field
            'Sentiment',
            sort='ascending'
        )
    )
    st.write("""
    ### Tweets by Sentiment Over Time
    """)
    st.altair_chart(chart_time_and_sentiment, use_container_width=True)

    # Todo: Interaction row

    # length of time period 100 most recent occurred in kpi (e.g., Occurred in 6 hours)

    # Current interaction rating: very low (> 24hrs), low (24hrs-12), med (12-4), high (4-2), very high (1)

    # Todo: Tweet descriptive statistics row

    # top 5 hashtags bar chart

    # favorite tweet & sentiment

    # most re-tweeted tweet & sentiment

    # User descriptive statistics row
    col1, col2 = st.beta_columns(2)
    user_counts = df['user.screen_name'].value_counts()

    # number of unique users kpi
    num_users = user_counts.size
    with col1:
        st.write("""
        ### Unique Users
        """, num_users)

    # user with most tweets
    user_max_tweets = user_counts.head(3).index.values
    count_max_tweets = user_counts.head(3).values
    df_top_tweets = pd.DataFrame({'User': user_max_tweets, 'Tweets': count_max_tweets})

    with col2:
        st.write("""
        ### Users with Most Tweets
        """)
        st.table(df_top_tweets.assign(hack='').set_index('hack'))

    # Table With Tweets and Sentiment
    with st.beta_expander("All Tweets Analyzed"):
        st.table(df[['created_at', 'user.screen_name', 'full_text', 'sentiment_text']].rename(columns={
            'created_at': 'Created',
            'user.screen_name': 'User',
            'full_text': 'Tweet',
            'sentiment_text': 'Sentiment'
        }))
