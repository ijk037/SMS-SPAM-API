from flask import Flask, request, jsonify
import joblib
import nltk
import re
import os

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

app = Flask(__name__)

model      = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf.pkl")
encoder    = joblib.load("label_encoder.pkl")

stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess(text):
    text   = text.lower()
    text   = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "SMS Spam API is running ✅"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Send JSON with a 'message' key"}), 400

    message = data["message"].strip()
    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    processed  = preprocess(message)
    vector     = vectorizer.transform([processed])
    prediction = model.predict(vector)
    proba      = model.predict_proba(vector)[0]
    label      = encoder.inverse_transform(prediction)[0]
    confidence = round(float(proba.max()) * 100, 1)

    return jsonify({
        "message":    message,
        "prediction": label,
        "confidence": f"{confidence}%",
        "is_spam":    label == "spam"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
