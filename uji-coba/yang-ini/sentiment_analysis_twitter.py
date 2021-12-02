import sys,tweepy,csv,re
from textblob import TextBlob
import matplotlib.pyplot as plt
import string

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

        # Open/create a file to append data to
        csvFile = open('result.csv', 'a')

        # Use csv writer
        csvWriter = csv.writer(csvFile)


        # creating some variables to store info
        polarity = 0
        positive = 0
        negative = 0
        neutral = 0


        # iterating through tweets fetched
        for tweet in self.tweets:
            #Append to temp so that we can store in csv later. I use encode UTF-8
            self.tweetText.append(self.cleanTweet(tweet.text).encode('utf-8'))
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
            

        # Write to csv and close csv file
        csvWriter.writerow(self.tweetText)
        csvFile.close()

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


    def cleanTweet(self, tweet):
        tweet = re.sub(r"\b[a-zA-Z]\b", "", tweet)
        tweet = re.sub('[0-9]+', '', tweet)
        tweet = tweet.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
        tweet = tweet.encode('ascii', 'replace').decode('ascii')
        tweet = re.sub(r'#', '', tweet)
        tweet = tweet.translate(str.maketrans("","",string.punctuation))
        tweet = tweet.strip()
        tweet = re.sub('\s+',' ',tweet)
        tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())
        return tweet

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

# if __name__== "__main__":
#     sa = SentimentAnalysis()
#     sa.DownloadData()
