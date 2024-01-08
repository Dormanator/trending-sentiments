# 📈 Trending Sentiments

⚠️ **Disclaimer: In an unexpected plot twist, Twitter (renamed "X") decided their API is too cool for school and slapped a price tag on it. Sadly, this means our humble website, trendinsentiments.com, and its modest GitHub companion are now just quaint relics of a freer internet era. Peek at the code if you like, but running it? That's gonna cost you now. RIP free access.** ⚠️

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

<<<<<<< HEAD

=======

> > > > > > > 97795e0ce17aa040e87860b0ddf73bde0930bed7

- Download and install Docker from https://www.docker.com
- Start up Docker on your local machine
- From within the trending-sentiments directory run: `docker build -t trending-sentiments .`
- Once the container has built run it with:
  <<<<<<< HEAD
  `docker run -p 8501:8501 -e TWITTER_KEY=<YOUR KEY> -e TWITTER_SECRET_KEY=<YOUR SECRET> trending-sentiments`

## 📖 References

- # Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. _Eighth International Conference on Weblogs and Social Media_ (ICWSM-14). Ann Arbor, MI, June 2014.
  `docker run -p 8501:8501 -e TWITTER_KEY=<YOUR KEY> -e TWITTER_SECRET_KEY=<YOUR SECRET> trending-sentiments`

## 📖 References

- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. _Eighth International Conference on Weblogs and Social Media_ (ICWSM-14). Ann Arbor, MI, June 2014.
  > > > > > > > 97795e0ce17aa040e87860b0ddf73bde0930bed7
