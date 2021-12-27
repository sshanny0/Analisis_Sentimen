import sys,tweepy,csv,re
import matplotlib.pyplot as plt
import string
import nltk
import Sastrawi
import numpy as np
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
        consumerKey='qubRP6d5B5eCD3RHrgoMRkDjH'
        consumerSecret='AJlk1MiHfXqnhByWtCf1Mbi2T9WGW08hoLkkDIbks4ztUpFx99'
        accessToken='3180202308-DmrgiSoxUUoSRevLLPldgCE57M0eD117K0mhFG5'
        accessTokenSecret='fuY5Hkk838OnmGOZUb16KLkSDw3JZjflxXQ1NHP6pP9Jr'
        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth, wait_on_rate_limit = False)

        # input for term to be searched and how many tweets to search
        # searchTerm = input("Enter Keyword/Tag to search about: ")
        # NoOfTerms = int(input("Enter how many tweets to search: "))

        # searching for tweets
        self.tweets = tweepy.Cursor(api.search_tweets, q=searchTerm, lang = "in").items(NoOfTerms)
        users = [[tweet.created_at, tweet.full_text] for tweet in posts]
        df = pd.DataFrame(data=users, columns=['Created_At','Tweets'])
        # Open/create a file to append data to
        csvFile = df.to_csv('hasilnya.csv')

        csvWriter = csv.writer(csvFile)

        polarity = 0
        positive = 0
        negative = 0
        neutral = 0

        # iterating through tweets fetched
        for tweet in self.tweets:
            #Append to temp so that we can store in csv later. I use encode UTF-8
            self.tweetText.append(self.SVMmethod(tweet.text).encode('utf-8'))
            # print (tweet.text.translate(non_bmp_map))    #print tweet's text
            analysis = TextBlob(tweet.text)

            # print(analysis.sentiment)  # print tweet's polarity
            polarity += analysis.sentiment.polarity  # adding up polarities to find the average later

            if (analysis.sentiment.polarity == 0):  # adding reaction of how people are reacting to find average later
                neutral += 1
            
            elif (analysis.sentiment.polarity > 0 ):
                positive += 1
            
            elif (analysis.sentiment.polarity < 0):
                negative += 1

        # write csv
        csvWriter.writerow(self.tweetText)
        csvFile.close()

        # finding average reaction
        polarity = polarity / NoOfTerms

        # finding average of how people are reacting
        positive = self.percentage(positive, NoOfTerms)
       
        negative = self.percentage(negative, NoOfTerms)
        
        neutral = self.percentage(neutral, NoOfTerms)

        # finding average reaction
        polarity = polarity / NoOfTerms

        # printing out data
        print("How people are reacting on " + searchTerm + " by analyzing " + str(NoOfTerms) + " tweets.")
        print()
        print("General Report: ")

        if (polarity == 0):
            print("Neutral")
        
        elif (polarity > 0 ):
            print("Positive")
       
        elif (polarity < 0):
            print("Negative")
        
        print()
        print("Detailed Report: ")
        print(str(positive) + "% people thought it was positive")
        
        print(str(negative) + "% people thought it was negative")
        
        print(str(neutral) + "% people thought it was neutral")

        self.plotPieChart(positive, negative, neutral, searchTerm, NoOfTerms)

        print(classification_report(y_test, predictions_SVM))

        self.SVMmethod(positive, negative, neutral)

    def load_data():
        data = pd.read_csv('hasilnya.csv')
        return data

    tweet_df = load_data()
    df = pd.DataFrame(tweet_df[['Tweets']])

    def cleanTweet(self,tweet):
        tweet = tweet.lower()
        #remove one char
        tweet = re.sub(r"\b[a-zA-Z]\b", "", tweet)
        #remove angka
        tweet = re.sub('[0-9]+', '', tweet)
        # remove tab, new line, ans back slice
        tweet = tweet.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
        # remove non ASCII (emoticon, chinese word, .etc)
        tweet = tweet.encode('ascii', 'replace').decode('ascii')
        # remove mention, link, hashtag
        tweet = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", tweet).split())
        # remove hashtags #
        tweet = re.sub(r'#', '', tweet)
        #remove punctuation
        tweet = tweet.translate(str.maketrans("","",string.punctuation))
        #remove whitespace leading & trailing
        tweet = tweet.strip()
        #remove multiple whitespace into single whitespace
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

    df['Tweet_Normal'] = df['Clean_Data'].apply(normalized_term)

    #Stopwords
    stop_words=stopwords.words('indonesian')
    stop_words.extend(['yg', 'dg', 'dgn', 'ny', 'd', 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh',
                       '&amp', 'yah', 'klau', 'a', 'b', 'c', 'd', 'e', 'f',
                       'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 
                       's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
    df.columns = ['Tweets', 'Clean_Data', 'Tweet_Normal']
    df['Tanpa_StopWords'] = df['Tweet_Normal'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    df = pd.DataFrame(tweet_df[['Tweets', 'Clean_Data', 'Tweet_Normal', 'Tanpa_StopWords']])

    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    df['Stemming'] = df['Tanpa_StopWords'].apply(lambda x : ([stemmer.stem(y) for y in x.split()]))

    #Lexicon
    negasi = ['bukan','tidak','ga','gk']
    lexicon = pd.read_csv('kebutuhan/full_lexicon.csv')
    lexicon = lexicon.drop(lexicon[(lexicon['word'] == 'bukan')
                               |(lexicon['word'] == 'tidak')
                               |(lexicon['word'] == 'ga')|(lexicon['word'] == 'gk') ].index,axis=0)
    lexicon = lexicon.reset_index(drop=True)

    lexicon_word = lexicon['word'].to_list()
    lexicon_num_words = lexicon['number_of_words']
    ns_words = []

    for word in word_dict.keys():
        if word not in lexicon_word:
            kata_dasar = stemmer.stem(word)
            if kata_dasar not in lexicon_word:
                ns_words.append(word)

    sencol =[]
    senrow =np.array([])
    nsen = 0

    sentiment_list = []
    # function to write the word's sentiment if it is founded
    def found_word(ind,words,word,sen,sencol,sentiment,add):
    # if it is already included in the bag of words matrix, then just increase the value
        if word in sencol:
            sen[sencol.index(word)] += 1
        else:
            #if not, than add new word
            sencol.append(word)
            sen.append(1)
            add += 1

        #if there is a negation word before it, the sentiment would be the negation of it's sentiment
        if (words[ind-1] in negasi):
            sentiment += -lexicon['weight'][lexicon_word.index(word)]
        else:
            sentiment += lexicon['weight'][lexicon_word.index(word)]

        return sen,sencol,sentiment,add

        # checking every words, if they are appear in the lexicon, and then calculate their sentiment if they do
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
            # check whether they are included in the lexicon
            if word in lexicon_word :
                sen,sencol,sentiment,add= found_word(ind,words,word,sen,sencol,sentiment,add)
            else:
                # if not, then check the root word
                kata_dasar = stemmer.stem(word)
                if kata_dasar in lexicon_word:
                    sen,sencol,sentiment,add= found_word(ind,words,kata_dasar,sen,sencol,sentiment,add)
                    # if still negative, try to match the combination of words with the adjacent words
                elif(n_words>1):
                    if ind-1>-1:
                        back_1    = words[ind-1]+' '+word
                        if (back_1 in lexicon_word):
                            sen,sencol,sentiment,add= found_word(ind,words,back_1,sen,sencol,sentiment,add)
                        elif(ind-2>-1):
                            back_2    = words[ind-2]+' '+back_1
                            if back_2 in lexicon_word:
                                sen,sencol,sentiment,add= found_word(ind,words,back_2,sen,sencol,sentiment,add)

            # if there is new word founded, then expand the matrix
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
                    # if there isn't then just update the old matrix

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

    df.loc[df['Bobot'] == 0, 'Label'] = 'netral'
    df.loc[df['Bobot'] > 0, 'Label'] = 'positif'
    df.loc[df['Bobot'] < 0, 'Label'] = 'negatif'

    df.loc[cek_df['Bobot'] == 0, 'Sentimen'] = 2 #netral
    df.loc[cek_df['Bobot'] > 0, 'Sentimen'] = 1 #positive
    df.loc[cek_df['Bobot'] < 0, 'Sentimen'] = 0 #negative

def SVMmethod(self):
    from sklearn import model_selection
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = model_selection.train_test_split(df['Stemming'], df['Sentimen'], test_size=0.1, random_state=0)

    train_df = pd.DataFrame()
    train_df['Stemming'] = x_train
    train_df['Sentimen'] = y_train
    test_df = pd.DataFrame()
    test_df['Stemming'] = x_test
    test_df['Sentimen'] = y_test

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer()  
    docs = df['Stemming']
    x = tfidf.fit(docs)
    x_train_tfidf = x.transform(x_train)
    x_test_tfidf = x.transform(x_test)

    from sklearn import svm
    SVM = svm.SVC(kernel='linear')
    SVM.fit(x_train_tfidf, y_train)

    predictions_SVM = SVM.predict(x_test_tfidf)

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report

    confusion_matrix = confusion_matrix(y_test, predictions_SVM)

# function to calculate percentage
def percentage(self, part, whole):
    temp = 100 * float(part) / float(whole)
    return format(temp, '.2f')

def plotPieChart(self, positive, negative, neutral, searchTerm, noOfSearchTerms):
    labels = ['Positive [' + str(positive) + '%]', 'Neutral [' + str(neutral) + '%]',
              'Negative [' + str(negative) + '%]']
    sizes = [positive, neutral, negative]
    colors = ['brown', 'lightcoral', 'beige']
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.title('How people are reacting on "' + searchTerm + '" by analyzing ' + str(noOfSearchTerms) + ' Tweets.')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("static/fig.png")

