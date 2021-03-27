import unittest
import json
import numpy as np

from app.transformer_pipeline import TransformerPipeline


def _get_mock_sentiment_predictions(df):
    df_copy = df.copy()
    mock_scores = np.arange(-1.0, 1, 0.2)
    mock_text = ['Negative'] * 5 + ['Neutral'] + ['Positive'] * 4
    df_copy['sentiment_score'] = mock_scores
    df_copy['sentiment_text'] = mock_text
    return df_copy


def _load_mock_json():
    with open('./test/resources/test_twitter_response.json', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        return json_data['_json']


class TestTransformerPipeline(unittest.TestCase):
    """Tests the transformation of data to formats expected by app"""

    transformer = TransformerPipeline()

    def test_convert_json_to_dataframe(self):
        expected_rows = 10
        expected_cols = ['id', 'created_at', 'full_text', 'tweet', 'retweet_count', 'favorite_count',
                         'entities.hashtags', 'user.id', 'user.screen_name']

        mock_json = _load_mock_json()
        test_df = self.transformer.convert_json_to_dataframe(mock_json)

        # Test shape
        self.assertEqual(expected_rows, len(test_df))
        for col in expected_cols:
            self.assertTrue(col in test_df)

    def test_clean_tweet(self):
        raw_tweet = '  In the #Avatar       https://www.test.com      sequels, you \t' \
                    '\n won’t just return   to Pandora — you’ll explore new parts of the world. @officialavatar '
        expected_tweet = 'In the sequels, you won’t just return to Pandora — you’ll explore ' \
                         'new parts of the world.'

        clean_tweet = self.transformer.clean_tweet(raw_tweet)
        self.assertEqual(clean_tweet, expected_tweet)

    def test_map_sentiment_label(self):
        mock_df = self.transformer.convert_json_to_dataframe(_load_mock_json())
        mock_df = _get_mock_sentiment_predictions(mock_df)
        expected_text = mock_df['sentiment_text'].to_list()

        sentiment_text = mock_df['sentiment_score'].map(self.transformer.map_sentiment_label).to_list()

        self.assertListEqual(sentiment_text, expected_text)

    def test_map_interaction_label(self):
        mock_time_deltas = []
        expected_text = []

    def test_gen_tweets_by_time_dataframe(self):
        pass

    def test_gen_sentiment_score_by_time_dataframe(self):
        pass

    def test_gen_hashtag_counts_dataframe(self):
        pass


if __name__ == '__main__':
    unittest.main()
