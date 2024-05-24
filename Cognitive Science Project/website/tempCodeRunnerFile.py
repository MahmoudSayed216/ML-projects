from flask import Flask, request, render_template
import pickle as pkl
import nltk 
# from nltk.stem import SnowballStemmer
# import neattext.functions as nfx
import pickle

# def stem(sentence):
#     Stemmer = SnowballStemmer(language = 'english')
#     pre_stemming = sentence.split()
#     post_stemming = []
#     for word in pre_stemming:
#         post_stemming.append(Stemmer.stem(word))
#     return ' '.join(post_stemming)

app = Flask(__name__, static_url_path='/static')



@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/predict", methods = ["POST"])
def predict():
  
    return render_template("index.html", emotion = "")

if __name__ == "__main__":
    app.run(debug=True, use_reloader = False)




