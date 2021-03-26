import unittest
import json

from app.transformer_pipeline import TransformerPipeline


class TestTransformerPipeline(unittest.TestCase):
    """Tests the transformation of data to formats expected by app"""

    transformer = TransformerPipeline()
    test_json = None
    test_df = None

    def _load_test_json(self):
        with open('./test/resources/test_twitter_response.json', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            self.test_json = json_data['_json']

    def test_convert_json_to_dataframe(self):
        expected_rows = 10
        expected_cols = ['id', 'created_at', 'full_text', 'tweet', 'retweet_count', 'favorite_count',
                           'entities.hashtags', 'user.id', 'user.screen_name']
        self._load_test_json()

        self.test_df = self.transformer.convert_json_to_dataframe(self.test_json)

        # Test shape
        self.assertEqual(expected_rows, len(self.test_df))
        for col in expected_cols:
            self.assertTrue(col in self.test_df)

    def test_clean_tweet(self):
        pass

    def test_predict_sentiment(self):
        pass

    def test_map_sentiment_label(self):
        pass

    def test_map_interaction_label(self):
        pass

    def test_gen_tweets_by_time_dataframe(self):
        pass

    def test_gen_sentiment_score_by_time_dataframe(self):
        pass

    def test_gen_hashtag_counts_dataframe(self):
        pass


if __name__ == '__main__':
    unittest.main()
