import os
import streamlit as st
import tweepy
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

auth = tweepy.AppAuthHandler(os.getenv('TWITTER_KEY'), os.getenv('TWITTER_SECRET_KEY'))
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

results = api.search(q='#Avatar', count=100, tweet_mode='extended', result_type='recent')
json_data = [r._json for r in results]

df = pd.json_normalize(json_data)
df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert(None)

df = df[['id', 'created_at', 'full_text', 'retweet_count', 'favorite_count',
         'entities.hashtags', 'user.id', 'user.screen_name']]

st.write("""
#  Trending Sentiments
Discover trending sentiments on Twitter with a hashtag or keyword search.
""")

tweets_per_min = df['created_at'].map(lambda x: x.replace(second=0)).value_counts()
st.write("""
## Frequency of 100 Most Recent Tweets
""")
st.line_chart(tweets_per_min)

# length of time period 100 most recent occurred in kpi (e.g., Occurred in 6 hours)

# number of unique users kpi

# Current interaction rating: very low (> 24hrs), low (24hrs-12), med (12-4), high (4-2), very high (1)


# Scatter of sentiment & intensity over time for tweets

# Frequency counts per minute/hour/day of pos & neg tweets

# top 5 hashtags bar chart


# user with most tweets

# favorite tweet & sentiment

# most re-tweeted tweet & sentiment
