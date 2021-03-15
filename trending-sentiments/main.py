import os
import re
import string

import streamlit as st
import tweepy
import pandas as pd
import numpy as np
import stanza

from dotenv import load_dotenv

load_dotenv()


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
    # TODO: add 'RT @user:' to full tweet text
    retweet_map = ~dataframe['retweeted_status.full_text'].isnull()
    end_of_user_tags = dataframe.loc[retweet_map, 'full_text'].str.index(':')
    end_of_user_tags
    # 
    dataframe['tweet'] = dataframe['retweeted_status.full_text'].fillna(dataframe['full_text'])
    dataframe['created_at'] = pd.to_datetime(dataframe['created_at']).dt.tz_convert(None)
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
    doc = nlp(tweet)
    return doc.sentences[0].sentiment


if __name__ == '__main__':
    # Setup Page Title
    st.set_page_config(page_title="Trending Sentiments", page_icon="📈", initial_sidebar_state="expanded", )

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
    st.balloons()
    st.write(df['tweet'][0], df['sentiment'][0])

    # Start of Page Body
    st.write("""
    ## 100 Most Recent Tweets for:
    """, userInput)

    # Todo: Frequency counts per minute/hour/day of pos & neg tweets
    # Or, Stacked barchart of sentiment & intensity over time for tweets??
    tweets_per_min = df['created_at'].map(lambda x: x.replace(second=0)).value_counts()
    st.write("""
    ### Frequency of Tweets
    """)
    st.line_chart(tweets_per_min)

    # Todo: Time period descriptive statistics row

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
    user_max_tweets = user_counts[[0]].index.values[0]
    count_max_tweets = user_counts[[0][0]]
    with col2:
        st.write("""
        ### User with Most Tweets
        """, '@', user_max_tweets, '&nbsp;&nbsp;&nbsp;&nbsp;', count_max_tweets)
