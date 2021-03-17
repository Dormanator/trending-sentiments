import os

import streamlit as st
import stanza
import tweepy
import altair as alt
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from transform_service import TransformService

load_dotenv()


# Todo: Move transformations into transformService

@st.cache()
def load_model():
    stanza.download('en', model_dir='./model')


def predict_sentiment(tweet):
    if tweet:
        doc = nlp(tweet)
        return doc.sentences[0].sentiment
    else:
        return 0


def twitter_connect():
    auth = tweepy.AppAuthHandler(os.getenv('TWITTER_KEY'), os.getenv('TWITTER_SECRET_KEY'))
    return tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


if __name__ == '__main__':
    transform = TransformService()

    # Setup Page Title
    st.set_page_config(page_title="Trending Sentiments", page_icon="📈", initial_sidebar_state="expanded", )
    st.markdown(
        """<style>
            table {text-align: left !important}
        </style>
        """, unsafe_allow_html=True)

    # Setup Stanza NLP Model & Twitter API
    with st.spinner('🔨 Getting everything ready...'):
        api = twitter_connect()
        load_model()
        nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')

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

    with st.spinner('🔎 Searching for tweets...'):
        results = api.search(q=userInput, count=100, tweet_mode='extended', result_type='recent')
        json_data = [r._json for r in results]
        df = transform.convert_json_to_dataframe(json_data)

    # Predict tweet sentiments using Stanza CNN classifier
    with st.spinner('⏳ Analyzing sentiments. This may take a moment...'):
        df['sentiment_text'] = df['tweet'] \
            .map(transform.clean_tweet) \
            .map(predict_sentiment) \
            .map(transform.map_sentiment_label)
        df['sentiment_text'].astype('category')
    st.balloons()

    # Start of Page Body
    st.write("""
    ## 100 Most Recent Tweets for:
    """, userInput)

    # Row: Interaction descriptive stats
    col1, col2, col3 = st.beta_columns(3)

    # Col: length of time period 100 most recent occurred
    time_range = df['created_at'].max() - df['created_at'].min()
    with col1:
        st.write("""
            ### Occurred Over
            """, time_range)

    # Col: Current interaction rating: very low (> 24hrs), low (24hrs-12), med (12-4), high (4-2), very high (<2)
    interaction_description = transform.map_interaction_label(time_range)
    with col2:
        st.write("""
            ### Interaction Level
            """, interaction_description)

    # Col: Sentiment most seen across the sample
    most_common_sentiment = df['sentiment_text'].mode()[0]
    with col3:
        st.write("""
            ### Overall Sentiment
            """, most_common_sentiment)

    # Get counts per sentiment level for every timestamp to the minute
    # Df with shape: created_at           Negative  Neutral   Positive
    #                2000-01-01 12:34:00  1         0         2
    tweets_by_sentiment = df.groupby(df['created_at'].map(lambda x: x.replace(second=0)))['sentiment_text'] \
        .value_counts() \
        .unstack(fill_value=0) \
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
    df_time_and_sentiment = pd.DataFrame(time_and_sentiment, columns=['Created', 'Tweets', 'Sentiment'])
    # Row: Graph of predictive sentiment time series
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
        ),
        tooltip=['Tweets', 'Created']
    )
    st.write("""
    ### Sentiment Frequency Over Time
    """)
    st.altair_chart(chart_time_and_sentiment, use_container_width=True)

    # Todo: Tweet descriptive statistics row

    # top 5 hashtags bar chart

    # favorite tweet & sentiment

    # most re-tweeted tweet & sentiment

    # Row: User descriptive stats
    col1, col2 = st.beta_columns(2)

    # Col: Number of unique users
    user_counts = df['user.screen_name'].value_counts()
    num_users = user_counts.size
    with col1:
        st.write("""
        ### Unique Users
        """, num_users)

    # Col: User with most tweets
    user_max_tweets = user_counts.head(3).index.values
    count_max_tweets = user_counts.head(3).values
    df_top_tweets = pd.DataFrame({'User': user_max_tweets, 'Tweets': count_max_tweets})
    with col2:
        st.write("""
        ### Users with Most Tweets
        """)
        st.table(df_top_tweets.assign(hack='').set_index('hack'))

    # Row: Table with all sample data records
    with st.beta_expander("All Tweets Analyzed"):
        st.table(df[['created_at', 'user.screen_name', 'full_text', 'sentiment_text']].rename(columns={
            'created_at': 'Created',
            'user.screen_name': 'User',
            'full_text': 'Tweet',
            'sentiment_text': 'Sentiment'
        }))
