import csv
import re
import string
import tweepy
import nltk
import Sastrawi
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from tweepy import OAuthHandler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from django.contrib.auth.models import User
from django.http import HttpResponse, Http404
from django.shortcuts import render
from app_as.config import *

def home(keyword):
    if keyword.method == 'POST':
        # You need to insert your own developer twitter credentials here
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)

        tweets = api.search(searchItem)

        users = [[tweet.created_at, tweet.full_text] for tweet in tweets]

        df = pd.DataFrame(data=users, columns=['Created_At','Tweets'])
        print (df.empty)
    return render(keyword, 'home.html')

def analisis(request):
    
    return render(request, 'analisis.html', {'report': '', 'user': user, })