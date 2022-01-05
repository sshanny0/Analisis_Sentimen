import tweepy
import matplotlib.pyplot as plt
import string
import nltk
import Sastrawi
import numpy as np
import pandas as pd
import ast
import os
import re
import csv

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn import metrics
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


class SentimentAnalysis:

    def __init__(self):
        self.tweets = []
        self.tweetText = []

    def DownloadData(self,searchTerm, NoOfTerms):
        # authenticating
        print(searchTerm, NoOfTerms)
        client = tweepy.Client(consumer_key = "DUZVWmR3ufL1YhcloahR8Vihe",
            consumer_secret = "qxrPCMYfzf898OGyzuRa1Aw5b44ibANhEAumaXbsZgn05saQWH",
            access_token ="1333705882643775489-hBBZlViy8LtFj7Obx8q0s73hejYddm",
            access_token_secret = "m9WiiHTzdSizlFocK5nxdEx3zQoeFBBLS0rDau1vglvNG",
            bearer_token = "AAAAAAAAAAAAAAAAAAAAACRyXgEAAAAAQ%2F2fvclHvP2kqBDWNX6r03K1M9Y%3D6yyZ43tTYsbLvtSRdwboFX9owRgKkBPOPFvrtrbf0cAcpClVV5",
            wait_on_rate_limit = False) 

        # input for term to be searched and how many tweets to search
        # searchTerm = input("Enter Keyword/Tag to search about: ")
        # NoOfTerms = int(input("Enter how many tweets to search: "))

        # searching for tweets
        tweets = []
        for tweet in tweepy.Paginator(client.search_recent_tweets,
                                    query = searchTerm,                             
                                    tweet_fields = ['author_id', 'created_at'],
                                    max_results = 100).flatten(limit=NoOfTerms):
            tweets.append(tweet)

        result = []

        for tweet in tweets:
            result.append({'Created_At': tweet.created_at, 'Tweets': tweet.text})
            df = pd.DataFrame(result)

        #Preprocessing
        def remove(tweet):
            tweet = tweet.lower()
            tweet = re.sub(r"\b[a-zA-Z]\b", "", tweet)
            tweet = re.sub('[0-9]+', '', tweet)
            tweet = tweet.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
            tweet = tweet.encode('ascii', 'replace').decode('ascii')
            tweet = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", tweet).split())
            tweet = re.sub(r'#', '', tweet)
            tweet = tweet.translate(str.maketrans("","",string.punctuation))
            tweet = tweet.strip()
            tweet = re.sub('\s+',' ',tweet)
            return tweet
            
        df["Clean_Data"] = df["Tweets"].apply(remove)

        #Normalization
        normalized_word = pd.read_csv('kebutuhan/Kamus_Baku.csv')
        normalized_word_dict = {}

        for index, row in normalized_word.iterrows():
            if row[0] not in normalized_word_dict:
                normalized_word_dict[row[0]] = row[1] 

        def normalized_term(document):
            return ' '.join([normalized_word_dict[term] if term in normalized_word_dict else term for term in document.split()])

        df['Tweet_Normal'] = df['Clean_Data'].apply(lambda x: normalized_term(x))

        #Stopwords
        stop_words=stopwords.words('indonesian')
        stop_words.extend(['yg', 'rt', 'dg', 'dgn', 'ny', 'd', 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh',
                       '&amp', 'yah', 'klau', 'a', 'b', 'c', 'd', 'e', 'f',
                       'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 
                       's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
        df['Tanpa_StopWords'] = df['Tweet_Normal'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

        #Stemming
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        df['Stemming'] = df['Tanpa_StopWords'].apply(lambda x : ([stemmer.stem(y) for y in x.split()]))
        df.to_csv('hasilnya.csv')
        
        #LEXICON BASED
        df = pd.read_csv('hasilnya.csv')
        negasi = ['bukan','tidak','ga','gk']
        lexicon = pd.read_csv('kebutuhan/full_lexicon.csv')
        lexicon = lexicon.reset_index(drop=True)
        lexicon_word = lexicon['word'].to_list()
        lexicon_num_words = lexicon['number_of_words']

        ns_words = []
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
            words = word_tokenize(df['Stemming'][i])
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

        sencol.append('Bobot')
        sentiment_array = np.array(sentiment_list).reshape(senrow.shape[0],1)
        sentiment_data = np.hstack((senrow,sentiment_array))
        df_sen = pd.DataFrame(sentiment_data,columns = sencol)

        cek_df = pd.DataFrame([])
        cek_df['Tweetnya'] = df['Stemming'].copy()
        cek_df['Bobot']  = df_sen['Bobot'].copy()

        df['Bobot'] = df_sen['Bobot']

        df.loc[df['Bobot'] == 0, 'Label'] = 'neutral'
        df.loc[df['Bobot'] > 0, 'Label'] = 'positive'
        df.loc[df['Bobot'] < 0, 'Label'] = 'negative'

        df.loc[cek_df['Bobot'] == 0, 'Sentimen'] = 1 #netral
        df.loc[cek_df['Bobot'] > 0, 'Sentimen'] = 2 #positive
        df.loc[cek_df['Bobot'] < 0, 'Sentimen'] = 0 #negative


        # SUPPORT VECTOR MACHINE
        x_train, x_test, y_train, y_test = model_selection.train_test_split(df['Stemming'], df['Sentimen'], test_size=0.1, random_state=0)
        train_df = pd.DataFrame()
        train_df['Stemming'] = x_train
        train_df['Sentimen'] = y_train
        test_df = pd.DataFrame()
        test_df['Stemming'] = x_test
        test_df['Sentimen'] = y_test
        tfidf = TfidfVectorizer()  
        docs = df['Stemming']
        x = tfidf.fit(docs)
        x_train_tfidf = x.transform(x_train)
        x_test_tfidf = x.transform(x_test)
        SVM = svm.SVC(kernel='linear')
        y_pred = SVM.fit(x_train_tfidf, y_train).predict(x_test_tfidf)
        cm = metrics.confusion_matrix(y_test, y_pred)
        word_tfidf = tfidf.fit_transform(df['Stemming'].values)
        df['Sentimen'] = SVM.predict(word_tfidf)
        df.to_csv('hasilnya.csv')

        #PIE DIAGRAM
        df = pd.read_csv('hasilnya.csv')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        def make_autopct(values):
            def my_autopct(pct):
                total  = sum(values)
                val = int(round(pct*total/100.0))
                return '{p:.2f}%({v:d})'.format(p=pct,v=val)
            return my_autopct
        sentimen = df['Stemming'].groupby(df['Sentimen']).count().values
        labels = ['Positive', 'Negative', 'Neutral']
        colors = ['brown', 'lightcoral', 'beige']
        explode = [0.1, 0, 0]
        ax1.pie(sentimen, colors=colors, labels=labels, explode=explode, autopct=make_autopct(sentimen), shadow=True, startangle=90)
        ax1.legend(labels, loc="upper right")
        ax1.set_title('How people are reacting on "' + searchTerm + '" by analyzing ' + str(NoOfTerms) + ' Tweets.')
        #CONFUSION MATRIX
        
        ax2.matshow(cm, cmap=plt.cm.Reds, alpha=0.3)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(x=j, y=i,
                    s=cm[i, j], 
                    va='center', ha='center')
        ax2.set_title('Confusion Matrix on "' + searchTerm + '" by analyzing ' + str(NoOfTerms) + ' Tweets.')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.savefig("static/fig.png")

# if __name__== "__main__":
#     sa = SentimentAnalysis()
#     sa.DownloadData()
