import codecs, json, csv
import pandas as pd

#read a json file downloaded with twitterscraper
with codecs.open('tweets18SnP500_3.json','r','utf-8') as f:
	All_Tweets = json.load(f,encoding='utf-b')

#tweets is now a list of tweets
#save into a csv
file = "tweets18SnP500_3.csv"  #file name

#open csv file for append
target_file = open(file, 'w', encoding='utf-8', newline='')
csv_file = csv.writer(target_file, delimiter=',', quotechar='"')
count=0 #a counter
i = 0 #a counter

eng_tweet = pd.DataFrame()
eng_tweet['timestamp'] = None
eng_tweet['likes'] = None
eng_tweet['content'] = None
eng_tweet['language'] = None

for tweet in All_Tweets:
    eng_tweet.loc[i,'timestamp'] = tweet['timestamp']
    eng_tweet.loc[i,'likes'] = tweet['likes']
    eng_tweet.loc[i,'content'] = tweet['text']
    eng_tweet.loc[i,'count'] = i
    if (tweet['html'].find('lang="en"')!=-1):
        eng_tweet.loc[i,'language'] = "ENGLISH!!!"
    else:
        eng_tweet.loc[i,'language'] = "NON ENGLISH!!!"
    i = i + 1
    count=count+1

#write to the csv (not required)
eng_tweet.to_csv('C:/Users/Smit/tweets18SnP500_3.csv')    
target_file.close()

##################################################
#make a for loop to iterate through the list of tweets
#use the function tone analysis to compute the sentiment
#save the sentiment output in a column and then categorize
#into +ve and -ve sentiment

from textblob import TextBlob

MixedTweet = pd.read_csv('tweets18SnP500_3.csv', index_col=0)

sentyEng = []

for i,text in enumerate(MixedTweet.content):
  blob = TextBlob(text)
  print(blob.sentiment)
  if blob.sentiment[0]>0:
     print('Positive')
     sentyEng.append('Positive')
  elif blob.sentiment[0]<0:
     print('Negative')
     sentyEng.append('Negative')
  else:
     print('Neutral')
     sentyEng.append('Neutral')

MixedTweet['Sentiments'] = sentyEng
MixedTweet.to_csv('C:/Users/Smit/tweets18SnP500_3.csv')