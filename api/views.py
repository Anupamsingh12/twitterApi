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

ps = PorterStemmer()

def index(request):
    return render(request,'index.html')


st={ "age": "12","gender": "0",
"height": "168",
"weight": "72",
"ap_hi": "156",
"ap_low": "50",
"cholesterol": "1",
"gluc": "3",
"smoke": "0",
"alco": "1",
"active": "1" }

@api_view(['GET', 'POST'])
def cardiorisk(request):
   
    if request.method == 'POST':
        filename='api/finalized_model2.sav'  
        loaded_model = pickle.load(open(filename, 'rb'))
        p=JSONParser().parse(request)
        
        x=[]
        for a in p:
            x.append(int(p[a]))
        aa=np.array([x])
        aa.reshape(-1,1)
        xx=loaded_model.predict_proba(aa)
        return Response({"message": "cardio risk", "data": xx})
       
    return Response({"message": "please send data in format ",'data':st})

def get_tweets1(df5,Topic1, Count1, coordinates, result_type, until_date):
    i=0
    #for tweet in tweepy.Cursor(api.search_tweets, q=Topic,count=100, lang='en',exclude='retweets').items():
    #for tweet in tweepy.Cursor(api.search_tweets, geocode = coordinates, lang = 'en', result_type = result_type, until = until_date, count = 100).items(max_tweets)
    for tweet in tweepy.Cursor(api.search_tweets, q=Topic1, count=Count1, geocode = coordinates, lang = 'en', result_type = result_type, until = until_date).items():
    #for tweet in tweepy.Cursor(api.search_tweets, q=Topic,count=100, lang='en').items():
        # print(i, end='\r')
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
    # print(df5.head())
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
@api_view(['GET', 'POST'])
def cardiorisk2(request):
    
    
    if request.method == 'POST':
     
        # print("asdadasdasd")
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        a=body["coordinates"]  
        b=body["topic"]  
        c=int(body["count"])  
        d=body["result_type"]  
        e=body["until_date"] 
        


        filename='api/tfidf.sav'
        tfidf = pickle.load(open(filename, 'rb'))
        df5 = pd.DataFrame(columns=["Date","User","IsVerified","Tweet","Likes","RT",'User_location'])
        print("inside djkfsfaskj")
        # filename='api/LogRegModel.sav'  
        filename='api/logisticNew.sav'  
        getTweetFromData(a,b,c,d,e,df5)
        # print(df5.head())
        if df5.empty:
            return JsonResponse({"message":"no data returned from twitter"})
        live_dataset = df5.copy()
        live_dataset['Tidy_Tweets'] = np.vectorize(remove_pattern)(live_dataset['Tweet'], "@[\w]*")
        live_dataset['Tidy_Tweets'] = live_dataset['Tidy_Tweets'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
        live_dataset['Tidy_Tweets'] = live_dataset['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")
        live_dataset['Tidy_Tweets'] = live_dataset['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
        tokenized_tweet1 = live_dataset['Tidy_Tweets'].apply(lambda x: x.split())
        tokenized_tweet1 = tokenized_tweet1.apply(lambda x: [ps.stem(i) for i in x])
        for i in range(len(tokenized_tweet1)):
            tokenized_tweet1[i] = ' '.join(tokenized_tweet1[i])
        live_dataset['Tidy_Tweets'] = tokenized_tweet1
        live_dataset_prepare = live_dataset['Tidy_Tweets']
        # tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=100,stop_words='english')
        tfidf_matrix=tfidf.fit_transform(live_dataset['Tidy_Tweets'])
        Log_Reg = pickle.load(open(filename, 'rb'))
        try:
            prediction_live_tfidf = Log_Reg.predict(tfidf_matrix)
            
            # test_pred_int = prediction_live_tfidf[:,1] >= 0.3
            test_pred_int = prediction_live_tfidf.astype(np.int)
            df5['label'] = test_pred_int
        except ValueError as ve:
            return JsonResponse({"message":"count of words in dataset is not more than 100."})

        # tokenized_tweet = df5['Tweet'].apply(lambda x: x.split())
        # df5["Tweet"] = df5["Tweet"].str.replace("[^a-zA-Z#]", " ")
        # df5['Tweet'] = df5["Tweet"].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
        # tokenized_tweet=df5["Tweet"].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
        
        df5["Tweet"]= np.vectorize(remove_pattern)(df5['Tweet'], "@[\w]*")
        tokenized_tweet= df5["Tweet"].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
        tokenized_tweet=tokenized_tweet.str.replace("[^a-zA-Z#]", " ")
        
        tokenized_tweet=tokenized_tweet.apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

        tokenized_tweet = tokenized_tweet.apply(lambda x: x.split())
        tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])
        tokenized_tweet=tokenized_tweet.apply(lambda x: [item for item in x if item not in stop])
        for i in range(len(tokenized_tweet)):
            tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
        yyy=tokenized_tweet

        xxx= yyy.str.split(expand=True).stack().value_counts()
        # print("=====================================================================================")
        # print(xxx.to_dict())
        count = df5['label'].value_counts()
        print(count[-1])
        print(count[1])
        print(count[0])
        mod = ModelPredictions(pred_type="twitter",positive_count=count[1],negetive_count=count[-1],neutral_count=count[0],total_result=len(df5['label'])
        ,query_String=b,location_cordinate=a)
        mod.save()


        # return JsonResponse({"data":df5.to_json()})
        return JsonResponse({"data":{'date':df5['Date'].values.tolist(),'User':df5['User'].values.tolist(),"IsVerified":df5['IsVerified'].values.tolist(),"Tweet":df5['Tweet'].values.tolist(),"User_location":df5['User_location'].values.tolist(),"label":df5['label'].values.tolist()},"wordCounts":xxx.to_dict()})
    #     p=JSONParser().parse(request)
       
    #     x=[]
    #     for a in p:
    #         x.append(int(p[a]))
    #     aa=np.array([x])
    #     aa.reshape(-1,1)
    #     xx=loaded_model.predict_proba(aa)
    #     return Response({"message": "cardio risk", "data": xx})
       
    # return Response({"message": "please send data in format ",'data':st})

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



@api_view(['GET', 'POST'])
def newsAnanlyserView(request):


    url = 'https://newsapi.org/v2/everything'
    # api_key = 'ef935e216c3e404c869b01a9a0da76ee'
    api_key = 'f84317925d46427ab3903575e1d2260d'
    # api_key = '31fcde72f0bc42f2871df05c681f3117'

    if request.method == 'POST':
     
        # print("asdadasdasd")
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        # a=body["coordinates"]  
        b=body["topic"]  
        c=int(body["count"])  
        d=body["result_type"]  
        e=body["until_date"] 
        print(b)
        parameters_headlines = {
        'q': str(b),
        'sortBy':'popularity',
        'pageSize': c,
        'apiKey': api_key,
        'language': 'en',
        'from' : e   
    }
        print("==============================")
        print(parameters_headlines)
        filename='api/tfidf.sav'
        tfidf = pickle.load(open(filename, 'rb'))
        filename='api/logisticNew2.sav'
        Log_Reg = pickle.load(open(filename, 'rb'))

        response_headline = requests.get(url, params = parameters_headlines)
        print(response_headline)
        if not response_headline.status_code == 200:
            return JsonResponse({"message":"you have exhausted your daily limit.","status_from_news":response_headline.status_code})
        response_json_headline = response_headline.json()
        responses = response_json_headline["articles"]
        # transforming the data from JSON dictionary to a pandas data frame
        news_articles_df = pd.DataFrame(get_articles(responses))
        # printing the head to check the format and the working of the get_articles function
        news_articles_df.head()
        news_articles_df['pub_date'] = pd.to_datetime(news_articles_df['pub_date']).apply(lambda x: x.date())
        news_articles_df.dropna(inplace=True)
        news_articles_df = news_articles_df[~news_articles_df['description'].isnull()]
        news_articles_df['combined_text'] = news_articles_df['title'].map(str) +" "+ news_articles_df['content'].map(str)
        live_dataset = news_articles_df['combined_text'].copy()
        live_dataset['combined_text'] = news_articles_df['combined_text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
        live_dataset['combined_text'] = live_dataset['combined_text'].str.replace("[^a-zA-Z#]", " ")
        live_dataset['combined_text'] = live_dataset['combined_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
        # live_dataset['combined_text'] = np.vectorize(remove_pattern)(live_dataset['combined_text'], "@[\w]*")
        print(live_dataset['combined_text'])
        
        xxx= live_dataset['combined_text'].str.split(expand=True).stack().value_counts()
        print(xxx)
        tokenized_tweet1 = live_dataset['combined_text']
        live_dataset['combined_text'] = tokenized_tweet1.str.replace("[^a-zA-Z#]", " ")
        live_dataset_prepare = live_dataset['combined_text']
        
        tfidf_matrix=tfidf.fit_transform(tokenized_tweet1)
        live_dataset['label']=[]
        try:
            prediction_live_tfidf = Log_Reg.predict(tfidf_matrix)
                
            # test_pred_int = prediction_live_tfidf[:,1] >= 0.3
            test_pred_int = prediction_live_tfidf.astype(np.int)
            print(test_pred_int)
            live_dataset['label'] = test_pred_int
            print(test_pred_int)
            unique, counts = numpy.unique(test_pred_int, return_counts=True)
            count = dict(zip(unique, counts))
            # count = live_dataset['label'].value_counts()
            print(count)
            print(count[-1])
            print(count[1])
            print(count[0])
            mod = newsModelPredictions(pred_type="news",positive_count=count[1],negetive_count=count[-1],neutral_count=count[0],total_result=len(test_pred_int)
            ,query_String=b)
            mod.save()
        except ValueError as ve:
            return JsonResponse({"message":"count of words in dataset is not more than 100."})
        return JsonResponse({"source":news_articles_df['source'].values.tolist(),"pub_date":news_articles_df['pub_date'].values.tolist()
        ,"url":news_articles_df['url'].values.tolist(),"label":json.dumps(test_pred_int.tolist()),"wordCounts":xxx.to_dict()})


@api_view(['GET'])
def allStats(request):
    q=request.GET
    qq=q.get('search_string', 'null')
    # print(qq)
    # print(q['type'].split('?'))
    if q['type']== 'twitter':
        if not qq=='null' :
            pred = ModelPredictions.objects.filter(query_String=qq)
            return JsonResponse({'data':list(pred.values())})
        
        else:
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

   
    

# allStats(3)
    