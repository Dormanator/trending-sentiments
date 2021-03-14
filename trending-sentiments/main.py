import os
import streamlit as st
import tweepy
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

COLS_TO_SAMPLE = ['id', 'created_at', 'full_text', 'retweet_count', 'favorite_count',
         'entities.hashtags', 'user.id', 'user.screen_name']

# Connect to Twitter
auth = tweepy.AppAuthHandler(os.getenv('TWITTER_KEY'), os.getenv('TWITTER_SECRET_KEY'))
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# Setup Page Title and Header
st.set_page_config(page_title="Trending Sentiments", page_icon="📈", initial_sidebar_state="expanded",)
st.write("""
# 📈 Trending Sentiments
Discover trending sentiments on Twitter with a hashtag or keyword search.
""")

# Setup Sidebar
# Handle user input
search = st.sidebar.text_input('Search Twitter', '#Avatar')
st.sidebar.write("""
Created by Ryan Dorman
""")

if not search:
    st.warning('Please input a search value.')
    st.stop()

# Search Twitter
results = api.search(q=search, count=100, tweet_mode='extended', result_type='recent')
json_data = [r._json for r in results]

# Convert JSON to Dataframe
df = pd.json_normalize(json_data)
df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert(None)
df = df[COLS_TO_SAMPLE]

# Start of Twitter result body
st.write("""
## 100 Most Recent Tweets for:
""", search)

# User descriptive statistics
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


# Todo: Frequency counts per minute/hour/day of pos & neg tweets
tweets_per_min = df['created_at'].map(lambda x: x.replace(second=0)).value_counts()
st.write("""
### Frequency of Tweets
""")
st.line_chart(tweets_per_min)

# length of time period 100 most recent occurred in kpi (e.g., Occurred in 6 hours)
# Current interaction rating: very low (> 24hrs), low (24hrs-12), med (12-4), high (4-2), very high (1)

# Stacked barchart of sentiment & intensity over time for tweets

# top 5 hashtags bar chart

# favorite tweet & sentiment

# most re-tweeted tweet & sentiment
