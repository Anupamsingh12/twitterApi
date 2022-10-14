from django.shortcuts import render
from rest_framework.response import Response
from django.http import JsonResponse
from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from rest_framework import permissions
from api.serializers import UserSerializer, GroupSerializer,serializers
from rest_framework.decorators import api_view
import numpy as np
import pickle
import os
from .models import ml_model
from django.core import serializers
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
import tweepy
import pandas as pd
import re
import numpy 
from sklearn.feature_extraction import text
from .models import ModelPredictions,newsModelPredictions
stop = text.ENGLISH_STOP_WORDS
consumer_key = 'HgEwalkiOGT4GHwRr9dqCa7UU' #API Key
consumer_secret = 'zASPPh2IGxN8hqTMxSnAkGQtMSpHUoL7qR8GFcQCPJ8HEXUFAJ' #API key secret
access_token = '1466028468005703680-fs47RLAsOeWrkqN4TQapR8GZwnKhiX'
access_token_secret = '3cogqCkPSffIgtn6vfjcVaVZufcE8XvsyxO28NWqeKpsh'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import PorterStemmer
import json
import requests



def clean_tweet(tweet):
    return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', str(tweet).lower()).split())

# from textblob.classifiers import NaiveBayesClassifier
# df=pd.read_csv('api/train.csv')
# df['tweet'] = df['tweet'].apply(lambda x : clean_tweet(x))
# df['tweet']= df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
# df['tweet']=df['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
# mast_df=df[['tweet','label']].copy()
# mast_df=list(zip(df['tweet'], df['label']))
# first2k = mast_df[0:2000]
# cl = NaiveBayesClassifier(first2k)
# #print("====================== model trainned cl =============================")
# ps = PorterStemmer()

def index(request):
    return HttpResponse("yey site is working fine")


def get_tweets1(df5,Topic1, Count1, coordinates, result_type, until_date):
    i=0
    #for tweet in tweepy.Cursor(api.search_tweets, q=Topic,count=100, lang='en',exclude='retweets').items():
    #for tweet in tweepy.Cursor(api.search_tweets, geocode = coordinates, lang = 'en', result_type = result_type, until = until_date, count = 100).items(max_tweets)
    for tweet in tweepy.Cursor(api.search_tweets, q=Topic1, count=Count1, geocode = coordinates, lang = 'en', result_type = result_type, until = until_date).items():
    #for tweet in tweepy.Cursor(api.search_tweets, q=Topic,count=100, lang='en').items():
        # #print(i, end='\r')
        df5.loc[i, 'Date']= tweet.created_at
        df5.loc[i, 'User']= tweet.user.name
        df5.loc[i, 'IsVerified']= tweet.user.verified
        df5.loc[i,'Tweet']=tweet.text
        #df.loc[i,'Likes']=tweet.favourite_count
        df5.loc[i,'User_location']= tweet.user.location
        
        #df.to_excel('{}.xlsx'.format('TweetDataset'),index=False)
        i=i+1
        if i>Count1:
            break
        else:
            pass

def getTweetFromData(a,b,c,d,e,df5):
    # #print(df5.head())
    # coordinates = '34.083656,74.797371,150mi'
    # Topic1 = 'militant'
    # Count1 = 150
    # result_type = 'recent'
    # until_date = '2022-05-30'
    coordinates = a
    Topic1 = b
    Count1 = c
    result_type = d
    until_date = e
    #df5 = pd.DataFrame(columns=["Date","User","IsVerified","Tweet","Likes","RT",'User_location'])
    get_tweets1(df5,Topic1, Count1, coordinates, result_type, until_date)

def remove_pattern(text,pattern):
    
    # re.findall() finds the pattern i.e @user and puts it in a list for further task
    r = re.findall(pattern,text)
    
    # re.sub() removes @user from the sentences in the dataset
    for i in r:
        text = re.sub(i,"",text)
    
    return text



def get_articles(file): 
    article_results = [] 
    for i in range(len(file)):
        article_dict = {}
        article_dict['title'] = file[i]['title']
        article_dict['author'] = file[i]['author']
        article_dict['source'] = file[i]['source']
        article_dict['description'] = file[i]['description']
        article_dict['content'] = file[i]['content']
        article_dict['pub_date'] = file[i]['publishedAt']
        article_dict['url'] = file[i]["url"]
        article_dict['photo_url'] = file[i]['urlToImage']
        article_results.append(article_dict)
    return article_results

def source_getter(df):
    source = []
    for source_dict in df['source']:
        source.append(source_dict['name'])
    df['source'] = source #append the source to the df
def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)




@api_view(['GET'])
def allStats(request):
    q=request.GET
    qq=q.get('search_string', 'null')
    fd=q.get('from_date', 'null')
    td=q.get('to_date', 'null')
    location=q.get('location', 'null')

    
    if q['type']== 'twitter':
        if (not qq=='null') and (not fd=='null') and (not td=='null') and (not location=='null'):
            # #print("")
            pred = ModelPredictions.objects.filter(query_time__gte = fd,query_time__lte= td,query_String=qq,location_cordinate=location)
            return JsonResponse({'data':list(pred.values())})
        elif (not qq=='null') and (not fd=='null') and (not td=='null'):
            # #print("")
            pred = ModelPredictions.objects.filter(query_time__gte = fd,query_time__lte= td,query_String=qq)
            return JsonResponse({'data':list(pred.values())})
        elif (not fd=='null') and (not td=='null'):
            # #print("ttttttttttttttttttt")
            pred = ModelPredictions.objects.filter(time__gte = fd,time__lte= td)
            return JsonResponse({'data':list(pred.values())})
        elif not qq=='null' :
            # #print("yaha")
            pred = ModelPredictions.objects.filter(query_String=qq)
            return JsonResponse({'data':list(pred.values())})
        elif not location=='null' :
            # #print("yaha")
            pred = ModelPredictions.objects.filter(location_cordinate=location)
            return JsonResponse({'data':list(pred.values())})
        
        
        
        else:
            #print("yahahaaa")
            pred=ModelPredictions.objects.all()
            return JsonResponse({'data':list(pred.values())})
    elif  q['type']== 'news':
        
        if not qq =='null'  :
            pred=newsModelPredictions.objects.filter(query_String=qq)
            return JsonResponse({'data':list(pred.values())})
        else:    
            pred=newsModelPredictions.objects.all()
            return JsonResponse({'data':list(pred.values())})
    return JsonResponse({"data":[]})

   
    



# from textblob import TextBlob
# def analyze_sentiment(tweet):
#     analysis = TextBlob(tweet)
#     if analysis.sentiment.polarity > 0:
#         return 1
#     elif analysis.sentiment.polarity == 0:
#         return 0
#     else:
#         return -1
# # def analyze_sentiment2(tweet):
# #     analysis = cl.classify(tweet)
# #     return analysis


# @api_view(['POST'])
# def twitterSentiment(request):
    
    
#     if request.method == 'POST':
     
#         # #print("asdadasdasd")
#         body_unicode = request.body.decode('utf-8')
#         body = json.loads(body_unicode)
#         a=body["coordinates"]  
#         b=body["topic"]  
#         c=int(body["count"])  
#         d=body["result_type"]  
#         e=body["until_date"] 
        
#         df5 = pd.DataFrame(columns=["Date","User","IsVerified","Tweet","Likes","RT",'User_location'])

        
      
#         getTweetFromData(a,b,c,d,e,df5)
#         if len(df5)<100:
#             return JsonResponse({"message":"not enough data returned from twitter try to increse the range"})
#         #print(df5)
#         if df5.empty:
#             return JsonResponse({"message":"not enough data returned from twitter try to increse the range"})
#         live_dataset = df5.copy()
#         live_dataset['clean_tweet'] = live_dataset['Tweet'].apply(lambda x : clean_tweet(x))
#         # #print(live_dataset['clean_tweet'])
#         live_dataset["Sentiment"] = live_dataset['clean_tweet'].apply(lambda x : analyze_sentiment(x))
#         df5['sentiment']=live_dataset["Sentiment"]
#         # #print(df5.head())

        

#         # #print(live_dataset["Sentiment"])
#         ppp= live_dataset["Sentiment"].value_counts()
#         # #print(ppp)
#         #print(ppp[-1])
#         #print(ppp[0])
#         #print(ppp[1])
        
#         mod = ModelPredictions(pred_type="twitter",positive_count=ppp[1],negetive_count=ppp[-1],neutral_count=ppp[0],total_result=len(live_dataset["Sentiment"])
#         ,query_String=b,location_cordinate=a,query_time=e)
#         mod.save()
#         tokenized_tweet=live_dataset['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
        
#         tokenized_tweet = live_dataset['clean_tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#         for i in range(len(tokenized_tweet)):
#             tokenized_tweet[i] = ''.join(tokenized_tweet[i])
#         tokenized_tweet= tokenized_tweet.apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
#         tokenized_tweet=  tokenized_tweet.str.replace('\d+', '')
#         yyy=tokenized_tweet
        
#         xxx= yyy.str.split(expand=True).stack().value_counts()
#         # for data in live_dataset.columns:
#             # #print(data)
#         live_dataset =live_dataset.drop(['Date', 'User','IsVerified','Likes','RT',"User_location","Tweet"], axis = 1)
#         live_dataset.to_csv('api/output.csv', mode='a', index=False, header=False)

#         return JsonResponse({"data":{"date":df5['Date'].values.tolist(),'User':df5['User'].values.tolist(),
#         "IsVerified":df5['IsVerified'].values.tolist(),"Tweet":df5['Tweet'].values.tolist(),"User_location":df5['User_location'].values.tolist()
#         ,"label":live_dataset["Sentiment"].to_list()
#         },"wordCounts":xxx.to_dict()})

# # from textblob.classifiers import NaiveBayesClassifier
# # df=pd.read_csv('api/train.csv')
# # df['tweet'] = df['tweet'].apply(lambda x : clean_tweet(x))
# # df['tweet']= df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
# # df['tweet']=df['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
# # mast_df=df[['tweet','label']].copy()
# # mast_df=list(zip(df['tweet'], df['label']))
# # first2k = mast_df[0:2000]
# # cl = NaiveBayesClassifier(first2k)

from django.http import HttpResponse
from datetime import datetime

import csv
@api_view(['GET'])
def save_file(request):
    date=datetime.today()
    filename="output-{}.csv".format(date)
    data = open(os.path.join('api/output.csv'),'r',encoding="utf8").read()
 
    resp = HttpResponse(data, content_type ='text/csv', headers={'Content-Disposition': 'attachment; filename={}'.format(filename)})
   
    return resp


import bs4

from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
@api_view(['GET'])
def getTrendingNews(request):
    news_url="https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
    Client=urlopen(news_url)
    xml_page=Client.read()
    Client.close()
    #    print(xml_page)
    soup_page=soup(xml_page,"xml")
    news_list=soup_page.findAll("item")
    # Print news title, url and publish date
    lst=[]

    for news in news_list:
        dic={'publishDate':news.pubDate.text,'newsText':news.title.text}
        lst.append(dic)
    return JsonResponse({'data':lst})

@api_view(['GET'])
def getTrendingTweets(request):
    consumer_key = "HgEwalkiOGT4GHwRr9dqCa7UU"
    consumer_secret = "zASPPh2IGxN8hqTMxSnAkGQtMSpHUoL7qR8GFcQCPJ8HEXUFAJ"
    access_token = "1466028468005703680-fs47RLAsOeWrkqN4TQapR8GZwnKhiX"
    access_token_secret = "3cogqCkPSffIgtn6vfjcVaVZufcE8XvsyxO28NWqeKpsh"

    # authorization of consumer key and consumer secret
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

    # set access to user's access key and access secret
    auth.set_access_token(access_token, access_token_secret)

    # calling the api
    api = tweepy.API(auth)

    # WOEID of London
    woeid = 23424848

    # fetching the trends
    trends = api.get_place_trends(id = woeid)

    # printing the information
    print("The top trends for the location are :")
    i=1
    lst=[]
    for value in trends:
        for trend in value['trends']:
            pp={'id':i,'name':trend['name']}
            print(trend['name'])
            lst.append(pp)
            i=i+1
    return JsonResponse({'data':lst})








# # for new model....=========================================
  

from keras.models import load_model

# Load model
model = load_model('api/rnn_model/best_model.h5')

max_words = 5000
max_len=50

def tweet_to_words(tweet):
    ''' Convert tweet text into a sequence of words '''
    
    # convert to lowercase
    text = tweet.lower()
    # remove non letters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # tokenize
    words = text.split()
    # remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # apply stemming
    words = [PorterStemmer().stem(w) for w in words]
    # return list
    return words


from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


def predict_class(text):
    '''Function to predict sentiment class of the passed text'''
    
    # sentiment_classes = ['Negative', 'Neutral', 'Positive']
    sentiment_classes = [-1, 0, 1]
    max_len=50
    # print(text)
    # Transforms text to a sequence of integers using a tokenizer object
    # tokenizer=''
    with open('api/rnn_model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)  
    xt = tokenizer.texts_to_sequences(text)
    # print(xt)
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # print(xt)
    # Do the prediction using the loaded model
    yt = model.predict(xt).argmax(axis=1)
    # Print the predicted sentiment
    # print('The predicted sentiment is', sentiment_classes[yt[0]])
    return sentiment_classes[yt[0]]




@api_view(['POST'])
def twitterSentimentNew(request):
    
    
    if request.method == 'POST':
     
        # #print("asdadasdasd")
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        a=body["coordinates"]  
        b=body["topic"]  
        c=int(body["count"])  
        d=body["result_type"]  
        e=body["until_date"] 
        
        df5 = pd.DataFrame(columns=["Date","User","IsVerified","Tweet","Likes","RT",'User_location'])

        
      
        getTweetFromData(a,b,c,d,e,df5)
        if len(df5)<100:
            return JsonResponse({"message":"not enough data returned from twitter try to increse the range"})
        #print(df5)
        if df5.empty:
            return JsonResponse({"message":"not enough data returned from twitter try to increse the range"})
        live_dataset = df5.copy()
        live_dataset['clean_tweet'] = live_dataset['Tweet'].apply(lambda x : clean_tweet(x))
        # #print(live_dataset['clean_tweet'])
        live_dataset["Sentiment"] = live_dataset['clean_tweet'].apply(lambda x : predict_class([x]))
        df5['sentiment']=live_dataset["Sentiment"]
        # #print(df5.head())

        

        # #print(live_dataset["Sentiment"])
        ppp= live_dataset["Sentiment"].value_counts()
        # #print(ppp)
        #print(ppp[-1])
        #print(ppp[0])
        #print(ppp[1])
        
        mod = ModelPredictions(pred_type="twitter",positive_count=ppp[1],negetive_count=ppp[-1],neutral_count=ppp[0],total_result=len(live_dataset["Sentiment"])
        ,query_String=b,location_cordinate=a,query_time=e)
        mod.save()
        tokenized_tweet=live_dataset['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
        
        tokenized_tweet = live_dataset['clean_tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        for i in range(len(tokenized_tweet)):
            tokenized_tweet[i] = ''.join(tokenized_tweet[i])
        tokenized_tweet= tokenized_tweet.apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
        tokenized_tweet=  tokenized_tweet.str.replace('\d+', '')
        yyy=tokenized_tweet
        
        xxx= yyy.str.split(expand=True).stack().value_counts()
        # for data in live_dataset.columns:
            # #print(data)
        live_dataset =live_dataset.drop(['Date', 'User','IsVerified','Likes','RT',"User_location","Tweet"], axis = 1)
        live_dataset.to_csv('api/output.csv', mode='a', index=False, header=False)

        return JsonResponse({"data":{"date":df5['Date'].values.tolist(),'User':df5['User'].values.tolist(),
        "IsVerified":df5['IsVerified'].values.tolist(),"Tweet":df5['Tweet'].values.tolist(),"User_location":df5['User_location'].values.tolist()
        ,"label":live_dataset["Sentiment"].to_list()
        },"wordCounts":xxx.to_dict()})





@api_view(['GET', 'POST'])
def newsAnanlyserViewNew(request):


    url = 'https://newsapi.org/v2/everything'
    # api_key = 'ef935e216c3e404c869b01a9a0da76ee'
    api_key = 'f84317925d46427ab3903575e1d2260d'
    # api_key = '31fcde72f0bc42f2871df05c681f3117'

    if request.method == 'POST':
     
      
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
       
        b=body["topic"]  
        c=int(body["count"])  
        d=body["result_type"]  
        e=body["until_date"] 
        domain = "null"
       
        parameters_headlines = {
        'q': str(b),
        'sortBy':'popularity',
        'pageSize': 100,
        'apiKey': api_key,
        'language': 'en',
        'from' : e   
         }
        if 'domain' in body:
            parameters_headlines=''
            domain = body["domain"]
            url="https://newsapi.org/v2/everything?apiKey=f84317925d46427ab3903575e1d2260d&q={}&domains={}&to={}".format(str(b),domain.lower(),e)
          
       
        print(url)
               
        response_headline = requests.get(url, params = parameters_headlines)
        print(response_headline)
        if not response_headline.status_code == 200:
            return JsonResponse({"message":"you have exhausted your daily limit.","status_from_news":response_headline.status_code})
        # if response_headline.total_result==0:
        #     return JsonResponse("dfsdfsdfsdf")
        response_json_headline = response_headline.json()
        # print(response_json_headline)
        print("============================")
        responses = response_json_headline["articles"]
        if not int(response_json_headline['totalResults']) > 2:
            return JsonResponse({"message":"not enough result from news api"})

        news_articles_df = pd.DataFrame(get_articles(responses))
 
        news_articles_df.dropna(inplace=True)
        news_articles_df = news_articles_df[~news_articles_df['description'].isnull()]
        news_articles_df['combined_text'] = news_articles_df['title'].map(str) +" "+ news_articles_df['content'].map(str)
        live_dataset = news_articles_df['combined_text'].copy()
        live_dataset['combined_text'] = news_articles_df['combined_text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
        # live_dataset['combined_text']=live_dataset['combined_text'].apply(lambda x: [item for item in x if item not in stop])
        live_dataset['combined_text'] = live_dataset['combined_text'].str.replace("[^a-zA-Z#]", " ")
        live_dataset['combined_text'] = live_dataset['combined_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
        xxx= live_dataset['combined_text'].str.split(expand=True).stack().value_counts()
       
        tokenized_tweet1 = live_dataset['combined_text']
        live_dataset['combined_text'] = tokenized_tweet1.str.replace("[^a-zA-Z#]", " ")
        live_dataset_prepare = live_dataset['combined_text']
        
        
        live_dataset['label']=[]
        try:
       
            live_dataset["label"] = live_dataset_prepare.apply(lambda x : predict_class([x]))
         
            # print(live_dataset["label"])
            count= live_dataset["label"].value_counts()
            print("======================================")
            print(count.keys())
            if 0 not in count.keys():
                count[0]=0
            if 1 not in count.keys():
                count[1]=0
            if -1 not in count.keys():
                count[-1]=0
            # print(type(count))
            mod = newsModelPredictions(pred_type="news",positive_count=count[1],negetive_count=count[-1],neutral_count=count[0],total_result=len(live_dataset["label"])
            ,query_String=b,source = "all")
            mod.save()
        except ValueError as ve:
           
            return JsonResponse({"message":"count of words in dataset is not more than 100."})
        return JsonResponse({"source":news_articles_df['source'].values.tolist(),"pub_date":news_articles_df['pub_date'].values.tolist()
        ,"url":news_articles_df['url'].values.tolist(),"label":live_dataset["label"].values.tolist(),"wordCounts":xxx.to_dict()})

