from flask import Flask, request, jsonify
import joblib
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

app = Flask(__name__)

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf.pkl")
encoder = joblib.load("label_encoder.pkl")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):

    text = text.lower()

    text = re.sub(r"[^a-z\s]", "", text)

    tokens = word_tokenize(text)

    tokens = [w for w in tokens if w not in stop_words]

    tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)


@app.route("/predict", methods=["POST"])

def predict():

    message = request.json["message"]

    processed = preprocess(message)

    vector = vectorizer.transform([processed])

    prediction = model.predict(vector)

    label = encoder.inverse_transform(prediction)[0]

    return jsonify({
        "message": message,
        "prediction": label
    })


if __name__ == "__main__":
    app.run(debug=True)