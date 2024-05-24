from flask import Flask, request, render_template
import pickle as pkl
import nltk 
from nltk.stem import SnowballStemmer
import neattext.functions as nfx
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

model = pickle.load(open("voting_classifier.pkl", "rb"))
victorizer : TfidfVectorizer = pickle.load(open("victorizer.pkl", "rb"))

def pre_process(sentence):  
    sentence = nfx.remove_stopwords(sentence)
    Stemmer = SnowballStemmer(language = 'english')
    pre_stemming = sentence.split()
    post_stemming = []
    for word in pre_stemming:
        post_stemming.append(Stemmer.stem(word))
    sentence = ' '.join(post_stemming)
    
    return victorizer.transform([sentence])


app = Flask(__name__, static_url_path='/static')



@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/predict", methods = ["POST"])
def predict():
    text = request.form["sentence"]
    vector = pre_process(text)
    prediction = model.predict(vector)[0]
    map_feeling_to_emote = {
        'sadness' : '\U0001F614',
        'anger'   : '\U0001F92C',
        'love'    : '\U0001F63B',
        'surprise': '\U0001F62E',
        'fear'    : '\U0001F628',
        'joy'     : '\U0001F604'
    }
    #['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']
    return render_template("index.html", emotion = f"{prediction} {map_feeling_to_emote[prediction]}")

if __name__ == "__main__":
    app.run(debug=True, use_reloader = False)




