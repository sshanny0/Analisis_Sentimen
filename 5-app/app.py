from flask import Flask, render_template, request
import os
from sentiment_analysis_twitter import SentimentAnalysis
# ----------------------------------------------
#     DECLARATIONS
# ----------------------------------------------
static_folder = 'static'
template_folder = "templates"
app = Flask(__name__,
            static_folder=static_folder,
            template_folder=template_folder)
sa = SentimentAnalysis()

# ----------------------------------------------
#     ROUTINGS
# ----------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/sentiment', methods=["POST"])
def sentiment():
    try:
        searchTerm = str(request.form['term'])
        NoOfTerms = int(request.form['number'])
    except:
        searchTerm = "-filter:retweets"
    sa.DownloadData(searchTerm, NoOfTerms)
    return render_template("show.html")
# ----------------------------------------------
#     MAIN FUNTION
# ----------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)