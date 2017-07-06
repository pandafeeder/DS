import tweepy
from textblob import TextBlob

CONSUMER_KEY = 'nje7lLkQhU598VvB3rjcjVcgm'
CONSUMER_SECRET = '6ue6dr7liLYTiy86BMD1XMcPQmh8mjeTqJsJTGktqbK96eJuoQ'
ACCESS_TOKEN = '296302632-bUYHlbcWDHPE4V7Y4ozapjQcrOtRYvw5EHhg82Ey'
ACCESS_SECRET = '3m7LMgSnIM4P70FogiPwWe74tuOXeXb20QSxbB1mfRiPg'


auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

api = tweepy.API(auth)

public_tweets = api.search('iphone8')

for t in public_tweets:
    print(t.text)
    analysis = TextBlob(t.text)
    print(analysis.sentiment)
