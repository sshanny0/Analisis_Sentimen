import sys, os
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
from matplotlib.figure import Figure
import matplotlib
import pandas as pd
import numpy as np
import re
import os
import tweepy
#import nltk
import re
import string
import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

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

    def clean_text(self, text):
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text

    cleantext = "(@[A-Za-z0-9_-]+)|([^A-Za-z \t\n])|(\w+:\/\/\S+)|(x[A-Za-z0-9]+)|(X[A-Za-z0-9]+)"  # regex untuk remove punctuation

    f = open("stopword_list_tala-1.txt", "r")
    isi = f.read()

    tempStoplist = []
    for tempstp in isi.split():
        tempStoplist.append(tempstp.lower())


    def preprocessing(self, teks):
        teks = teks.rstrip("\n")
        teks = re.sub(r'^https?:\/\/.*[\r\n]*', '', teks, flags=re.MULTILINE)
        teks = re.sub(cleantext, ' ', str(teks).lower()).strip()  # casefolding dan remove punctuation

        tokens = []
        for token in teks.split():
            if token not in tempStoplist:  # jika token tidak di stopword maka simpan
                token = stemmer.stem(token)  # lakukan stemming
                if len(token) > 2:
                    tokens.append(token)
        return " ".join(tokens)

    def runFunction(self):
        data = pd.read_csv(self.filename)
        # data = data.drop(columns=['Unnamed: 0'])
        data['cleantweet'] = data['isitweet'].apply(lambda x: clean_text(x))
        data['cleantweet2'] = data.isitweet.apply(lambda x: preprocessing(x))
        vectorizer = TfidfVectorizer()
        train, valid = train_test_split(data, test_size=0.2, random_state=0, stratify=data.label.values)
        X_train = vectorizer.fit_transform(train.isitweet.values)
        X_valid = vectorizer.transform(valid.isitweet.values)
        y_train = train.label.values
        y_valid = valid.label.values
        svc = SVC(kernel='rbf', C=1, random_state=42)
        model = svc.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        valid = valid.drop(columns=['label'])
        valid['label'] = y_pred
        valid = valid[['name', 'timetweet', 'isitweet', 'cleantweet', 'label']]
        self.result = valid
        self.tableView.setModel(PandasModel(self.result))
        self.posLabel.setText(str(len(valid[valid['label'] == 'positif'])))
        self.negLabel.setText(str(len(valid[valid['label'] == 'negatif'])))
        self.netLabel.setText(str(len(valid[valid['label'] == 'netral'])))
        print('Accuracy score: ', accuracy_score(y_valid, y_pred))

        # Plot Pie
        x = valid['label'].value_counts()
        colors = ['#c0c000', '#7a949c', '#52a7c1']
        labelnya = ['Netral', 'Positif', 'Negatif']
        self.matplotlibwidget.ax.pie(x, labels=labelnya, autopct='%.0f%%', colors=colors,
                                     wedgeprops=dict(width=0.6, edgecolor='w'));

        # Plot Bar
        self.matplotlibwidget2.ax.barh(valid['name'].value_counts().nlargest(5).index,
                                       data['name'].value_counts().nlargest(5), color=['#023e8a'])

    def saveFunction(self):
        outName, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Excel File (*.xlsx)")
        self.result.to_excel(outName, index=None, index_label=None)


#if __name__ == '__main__':
#    app = QApplication(sys.argv)
#    w = MainWidget()
#    w.show()
#    sys.exit(app.exec_())