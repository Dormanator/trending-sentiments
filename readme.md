# 📈 Trending Sentiments

⚠️ **As of late 2023, Trending Sentiments no longer works with free tier access to the Twitter (X) API. A basic level paid account ($100/mo) is required to query tweets for analysis. Due to this change the app will no longer be avalible online at https://trendingsentiments.com for public use.** ⚠️

## 🎓 About

Trending Sentiments is a data exploration application for analyzing hashtags and keywords in tweets created with [Streamlit](https://streamlit.io/). The application provides descriptive statistics on hashtag/term interaction, top tweets, and user participation. It provides predictive statistics on tweets' sentiments. Sentiment predictions are made using [VADER](https://github.com/cjhutto/vaderSentiment).

## 🚀 Quick Start

### Access Online

- Open your web browser and navigate to https://trendingsentiments.com
- Input a search term and hit Enter on your keyboard
- Discover!

### Setup

- Download and Install Python 3 from https://python.org
- Clone the [repository](https://github.com/Dormanator/trending-sentiments)
- Sign up as a developer at https://developer.twitter.com to obtain access to the API
- Once signed up as a Twitter developer save a reference to your API key and secret

### Download and Install

- Extract the archive onto your local machine to a `trending-sentiments` directory
- Navigate to the `trending-sentiments` directory on your local machine with the command prompt or terminal
- From within the `trending-sentiments` directory run: `pip install -r requirements.txt`
- In the `trending-sentiments` directory create a `.env` file that includes your Twitter API key and secret

```
 TWITTER_KEY=<YOUR KEY>
 TWITTER_SECRET_KEY=<YOUR SECRET>
```

### Run Tests

- From within the trending-sentiments directory run: `python -m unittest  `

### Run Application

- From within the trending-sentiments directory run: `streamlit run app/app.py`
- Open your web browser and navigate to http://localhost:8501

### Build and Run Docker Container

- Download and install Docker from https://www.docker.com
- Start up Docker on your local machine
- From within the trending-sentiments directory run: `docker build -t trending-sentiments .`
- Once the container has built run it with:
  `docker run -p 8501:8501 -e TWITTER_KEY=<YOUR KEY> -e TWITTER_SECRET_KEY=<YOUR SECRET> trending-sentiments`

## 📖 References

- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. _Eighth International Conference on Weblogs and Social Media_ (ICWSM-14). Ann Arbor, MI, June 2014.
