import re
import string

import pandas as pd
import numpy as np

class TransformService:
    """Handles the transformation of data to formats expected by UI"""

    # Remove whitespace between text, urls, punctuation, trailing whitespace
    def clean_tweet(self,  tweet):
        result = re.sub(r'\s+', ' ', tweet)
        result = re.sub(r"https?://[A-Za-z0-9./]+", ' ', result)
        return result \
            .translate(str.maketrans('', '', string.punctuation)) \
            .strip()

    # Map sentiment scores to text labels
    def map_sentiment(self, score):
        category = 'Neutral'
        if score == 0:
            category = 'Negative'
        elif score == 2:
            category = 'Positive'
        return category

    # Map time period 100 tweet took place in to interaction labels
    def map_interaction(self, time_delta):
        interaction_level = 'Very Low'
        if pd.Timedelta("12 hours") <= time_delta < pd.Timedelta("1 days"):
            interaction_level = 'Low'
        elif pd.Timedelta("4 hours") <= time_delta < pd.Timedelta("12 hours"):
            interaction_level = 'Medium'
        elif pd.Timedelta("2 hours") <= time_delta < pd.Timedelta("4 hours"):
            interaction_level = 'High'
        elif time_delta < pd.Timedelta("2 hours"):
            interaction_level = 'Very High'
        return interaction_level