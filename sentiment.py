import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def get_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(text)
    compound = score['compound']

    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'