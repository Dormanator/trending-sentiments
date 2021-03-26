import unittest

from app.transformer_pipeline import TransformerPipeline


class TestTransformerPipeline(unittest.TestCase):
    """Tests the transformation of data to formats expected by app"""

    def test_convert_json_to_dataframe(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_clean_tweet(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_map_sentiment_label(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

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
