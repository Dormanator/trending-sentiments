import os

import streamlit as st
import tweepy
import altair as alt
import pandas as pd
import numpy as np

# Import correct TF on Win & Linux
try:
    import tf_nightly as tf
    import tensorflow_text_nightly as text
except ImportError:
    import tensorflow as tf
    import tensorflow_text as text

from dotenv import load_dotenv
from transform_service import TransformService

load_dotenv()

def download_model():
    # check for model dir
    # if exists do nothing
    # if not exists download zip and extract to ./bert_model
    pass

@st.cache()
def load_model():
    model = tf.saved_model.load('./bert_model')
    return model


def twitter_connect():
    auth = tweepy.AppAuthHandler(os.getenv('TWITTER_KEY'), os.getenv('TWITTER_SECRET_KEY'))
    return tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


if __name__ == '__main__':
    transform = TransformService()

    # Setup Page Title and Styles
    st.set_page_config(page_title='Trending Sentiments', page_icon='📈', initial_sidebar_state='expanded', )
    st.markdown(
        """<style>
            table {text-align: left !important}
        </style>
        """, unsafe_allow_html=True)

    # Setup Sentiment Prediction Model & Twitter API
    with st.spinner('🔨 Getting everything ready...'):
        api = twitter_connect()
        sentiment_model = load_model()
        # nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')

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
        # df['sentiment_text'] = df['tweet'] \
        #     .map(transform.clean_tweet) \
        #     .map(predict_sentiment) \
        #     .map(transform.map_sentiment_label)
        clean_tweets = df['tweet'].map(transform.clean_tweet).to_list()
        sentiment_scores = tf.sigmoid(sentiment_model(tf.constant(clean_tweets)))
        df['sentiment_score'] = np.array(sentiment_scores).flatten()
        df['sentiment_text'] = df['sentiment_score'].map(transform.map_sentiment_label)
        df['sentiment_text'].astype('category')
        st.balloons()

    # Start of Page Body
    st.write("""
    ## 100 Most Recent Tweets for: <u>{}</u>
    """.format(userInput), unsafe_allow_html=True)

    # Row: Interaction descriptive stats
    col1, col2, col3 = st.beta_columns(3)

    # Col: length of time period 100 most recent occurred
    time_range = df['created_at'].max() - df['created_at'].min()
    with col1:
        st.write("""
            ### Occurred Over
            """, str(time_range))

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

    # Row: Graph of predictive sentiment time series
    df_sentiment_by_time = transform.gen_sentiment_by_time_dataframe(df)
    # Create stacked bar chart
    chart_sentiment_by_time = alt.Chart(df_sentiment_by_time).mark_bar().encode(
        x='Created',
        y=alt.Y('sum(Tweets)', axis=alt.Axis(title="Count")),
        color=alt.Color('Sentiment',
                        # Setup color by sentiment category
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
    st.altair_chart(chart_sentiment_by_time, use_container_width=True)

    # Row: Top Tweets descriptive stats row
    col1, col2 = st.beta_columns(2)

    # Top favorite & sentiment
    top_favorite = df.loc[df['favorite_count'] == df['favorite_count'].max()]
    with col1:
        st.write("""
        ### Top Favorite Tweet   
        """)
        st.write("""
        **Text:** {}  
        **User:** {}  
        **Sentiment:** {}    
        """.format(
            top_favorite['full_text'].values[0],
            top_favorite['user.screen_name'].values[0],
            top_favorite['sentiment_text'].values[0]
        ))

    # Col: Top re-tweet & sentiment
    top_retweet = df.loc[df['retweet_count'] == df['retweet_count'].max()]
    with col2:
        st.write("""
        ### Top Re-Tweet
        """)
        st.write("""
        **Text:** {}  
        **User:** {}  
        **Sentiment:** {}    
        """.format(
            top_retweet['full_text'].values[0],
            top_retweet['user.screen_name'].values[0],
            top_retweet['sentiment_text'].values[0]
        ))

    # Row: Top hashtags bar chart
    df_top_hashtags = transform.gen_hashtag_counts_dataframe(df)
    chart_top_hashtags = alt.Chart(df_top_hashtags.head(5)).mark_bar().encode(
        x=alt.X('Count', axis=alt.Axis(tickMinStep=1)),
        y=alt.Y('Hashtag', axis=alt.Axis(title=""), sort='-x')) \
        .configure_axis(labelFontSize=12)
    st.write("""
    ### Top 5 Hashtags
    """)
    st.altair_chart(chart_top_hashtags, use_container_width=True)

    # Row: User descriptive stats
    col1, col2 = st.beta_columns(2)

    # Col: Number of unique users
    user_counts = df['user.screen_name'].value_counts()
    num_users = user_counts.size
    with col1:
        st.write("""
        ### Unique Users
        """, str(num_users))

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
        st.write(
            df[['created_at', 'user.screen_name', 'full_text', 'sentiment_text', 'sentiment_score']].rename(columns={
                'created_at': 'Created',
                'user.screen_name': 'User',
                'full_text': 'Tweet',
                'sentiment_text': 'Sentiment',
                'sentiment_score': 'Sentiment Score'
            }))
