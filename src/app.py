import os
import datetime

import streamlit as st
import tweepy
import altair as alt
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformer_service import TransformerService

# Only use dotenv in dev
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("dotenv not found. Using sys env vars...")


def twitter_connect():
    auth = tweepy.AppAuthHandler(os.getenv('TWITTER_KEY'), os.getenv('TWITTER_SECRET_KEY'))
    return tweepy.API(auth)


def main():
    transformer = TransformerService()
    analyzer = SentimentIntensityAnalyzer()

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

    # Setup Page Header
    st.write("""
      # 📈 Trending Sentiments
      Discover trending sentiments on Twitter with a hashtag or keyword search.
      """)

    # Setup Sidebar
    # Handle user input
    user_input = st.sidebar.text_input('Search for a hashtag or keyword to begin', '#Avatar')
    st.sidebar.write("""
      Created by [Ryan Dorman](https://github.com/dormanator)
      """)

    # App description and info
    with st.sidebar.beta_expander("About"):
        st.write("""
        **Trending Sentiments** is a data exploration application for analyzing hashtags and keywords in tweets. 
        The application provides descriptive statistics on hashtag/term interaction, top tweets, and user 
        participation. It provides predictive statistics on tweets' sentiments. Sentiment prediction is made 
        using [VADER (Valence Aware Dictionary and sEntiment Reasoner)](https://github.com/cjhutto/vaderSentiment).
        """, unsafe_allow_html=True)

    # Check Twitter API rate limits and handle search state
    results = api.rate_limit_status()
    rate_limit_info = results['resources']['search']['/search/tweets']

    if not user_input:
        st.warning('⛔ Please input a search value.')
        st.stop()

    if rate_limit_info['remaining'] == 0:
        local_tz = datetime.datetime.utcnow().astimezone().tzinfo
        reset_time = pd.to_datetime(rate_limit_info['reset'], unit='s') \
            .tz_localize('utc') \
            .tz_convert(local_tz) \
            .strftime('%I:%M:%S %p')
        st.warning('⏲ We have to wait before getting more data. '
                   'Try again at {}.'.format(reset_time))
        st.stop()

    with st.spinner('🔎 Searching for tweets...'):
        results = api.search(q=user_input, count=100, tweet_mode='extended', result_type='recent')
        json_data = [r._json for r in results]
        df = transformer.convert_json_to_dataframe(json_data)

    # Predict tweet sentiments using VADER
    with st.spinner('⏳ Analyzing sentiments. This may take a moment...'):
        df['sentiment_score'] = df['tweet'] \
            .apply(transformer.clean_tweet) \
            .apply(analyzer.polarity_scores) \
            .apply(lambda d: d.get('compound'))
        df['sentiment_text'] = df['sentiment_score'].map(transformer.map_sentiment_label)
        df['sentiment_text'].astype('category')

    # Start of Page Body
    st.write("""
      ## 100 Most Recent Tweets for: <u>{}</u>
      """.format(user_input), unsafe_allow_html=True)

    # Row: Interaction descriptive stats
    col1, col2, col3 = st.beta_columns(3)

    # Col: length of time period 100 most recent occurred
    time_range = df['created_at'].max() - df['created_at'].min()
    with col1:
        st.write("""
              ### Occurred Over
              """, str(time_range))

    # Col: Current interaction rating: very low (> 24hrs), low (24hrs-12), med (12-4), high (4-2), very high (<2)
    interaction_description = transformer.map_interaction_label(time_range)
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
    df_sentiment_by_time = transformer.gen_sentiment_text_by_time_dataframe(df)
    # Create stacked bar chart
    chart_sentiment_by_time = alt.Chart(df_sentiment_by_time).mark_bar().encode(
        x='Created',
        y=alt.Y('sum(Tweets)', axis=alt.Axis(title="Count", tickMinStep=1)),
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

    # Todo: Row with sentiment analysis

    # Todo: Sentiment score distribution, https://altair-viz.github.io/gallery/histogram_with_a_global_mean_overlay.html
    # https://altair-viz.github.io/gallery/scatter_with_histogram.html

    # Todo: Future Sentiment Prediction
    # https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b
    st.write("""
      ### Sentiment Score Over Time
      """)
    df_sentiment_score_by_time = transformer.gen_sentiment_score_by_time_dataframe(df)
    chart_sentiment_score_by_time = alt.Chart(df_sentiment_score_by_time).mark_circle(size=60).encode(
        x='Created',
        y='Sentiment Score',
        color=alt.Color('Sentiment',
                        # Setup color by sentiment category
                        sort=alt.EncodingSortField('Sentiment', order='ascending'),
                        scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative']),
                        legend=alt.Legend(title='Sentiments')
                        ),
        tooltip=['Sentiment Score', 'Created']
    )
    st.altair_chart(chart_sentiment_score_by_time, use_container_width=True)

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
    df_top_hashtags = transformer.gen_hashtag_counts_dataframe(df)
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


if __name__ == '__main__':
    main()
