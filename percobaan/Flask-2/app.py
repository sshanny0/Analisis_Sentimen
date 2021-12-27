import tweepy
import csv
import pandas as pd
import numpy as np
import string
import re
import nltk
import Sastrawi
import ast
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

from tweepy import OAuthHandler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn import metrics, preprocessing
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection

from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

set(stopwords.words('indonesian'))
app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('indonesian')
    stop_words.extend(["yg", "dg", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh',
                       '&amp', 'yah', 'klau', 'a', 'b', 'c', 'd', 'e', 'f',
                   'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 
                   's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])

    text1 = request.form['text1'].lower()
    text1 = re.sub(r"\b[a-zA-Z]\b", "", request.form['text1'])
    text1 = re.sub('[0-9]+', '', request.form['text1'])
    text1 = request.form['text1'].replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    text1 = request.form['text1'].encode('ascii', 'replace').decode('ascii')
    text1 = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", request.form['text1']).split())
    text1 = re.sub(r'#', '', request.form['text1'])
    text1 = request.form['text1'].translate(str.maketrans("","",string.punctuation))
    text1 = request.form['text1'].strip()
    text1 = re.sub('\s+',' ',request.form['text1'])

    processed_doc1 = ' '.join([word for word in text1.split() if word not in stop_words])

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    output = stemmer.stem(text1)

    negasi = ['bukan','tidak','ga','gk']
    lexicon = pd.read_csv('full_lexicon-Copy1.csv')
    lexicon = lexicon.drop(lexicon[(lexicon['word'] == 'bukan')
                               |(lexicon['word'] == 'tidak')
                               |(lexicon['word'] == 'ga')|(lexicon['word'] == 'gk') ].index,axis=0)
    lexicon = lexicon.reset_index(drop=True)

    sencol =[]
    senrow =np.array([])
    nsen = 0
    sentiment_list = []
    def found_word(ind,words,word,sen,sencol,sentiment,add):
    	if word in sencol:
    		sen[sencol.index(word)] += 1
    	else:
    		sencol.append(word)
    		sen.append(1)
    		add += 1
    		if (words[ind-1] in negasi):
    			sentiment += -lexicon['weight'][lexicon_word.index(word)]
    		else:
    			sentiment += lexicon['weight'][lexicon_word.index(word)]
    			return sen,sencol,sentiment,add

    			for i in range(len(df)):
    				nsen = senrow.shape[0]
    				words = output[i]
    				sentiment = 0 
    				add = 0
    				prev = [0 for ii in range(len(words))]
    				n_words = len(words)
    				if len(sencol)>0:
    					sen =[0 for j in range(len(sencol))]
    				else:
    					sen =[]

    					for word in words:
    						ind = words.index(word)
    						if word in lexicon_word :
    							sen,sencol,sentiment,add= found_word(ind,words,word,sen,sencol,sentiment,add)
    						else:
    							kata_dasar = stemmer.stem(word)
    							if kata_dasar in lexicon_word:
    								sen,sencol,sentiment,add= found_word(ind,words,kata_dasar,sen,sencol,sentiment,add)
    							elif(n_words>1):
    								if ind-1>-1:
    									back_1    = words[ind-1]+' '+word
    									if (back_1 in lexicon_word):
    										sen,sencol,sentiment,add= found_word(ind,words,back_1,sen,sencol,sentiment,add)
    									elif(ind-2>-1):
    										back_2    = words[ind-2]+' '+back_1
    										if back_2 in lexicon_word:
    											sen,sencol,sentiment,add= found_word(ind,words,back_2,sen,sencol,sentiment,add)
    											if add>0:  
    											 	if i>0:
    											 		if (nsen==0):
    											 			senrow = np.zeros([i,add],dtype=int)
    											 		elif(i!=nsen):
    											 			padding_h = np.zeros([nsen,add],dtype=int)
    											 			senrow = np.hstack((senrow,padding_h))
    											 			padding_v = np.zeros([(i-nsen),senrow.shape[1]],dtype=int)
    											 			senrow = np.vstack((senrow,padding_v))
    											 		else:
    											 			padding =np.zeros([nsen,add],dtype=int)
    											 			senrow = np.hstack((senrow,padding))
    											 			senrow = np.vstack((senrow,sen))
    											 			if i==0:
    											 				senrow = np.array(sen).reshape(1,len(sen))
    											 			elif(nsen>0):
    											 				senrow = np.vstack((senrow,sen))
    											 				sentiment_list.append(sentiment)

    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed_doc1)
    compound = round((1 + dd['compound'])/2, 2)

    return render_template('form.html', final=compound, text1=text1)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
