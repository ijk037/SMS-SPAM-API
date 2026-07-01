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

# Helper to dynamically find the latest model version
def get_model_path_and_version():
    requested_version = os.environ.get("MODEL_VERSION", "").strip().lower()
    
    # 1. If a specific version is requested (e.g. 'v1', 'v2')
    if requested_version:
        path = os.path.join("models", requested_version)
        if os.path.exists(path) and os.path.isdir(path):
            return path, requested_version
        # Fallback to root if the requested version is 'root' or doesn't exist
        if requested_version == "root":
            return "", "root"
        # If it doesn't exist in models/, try root or raise error
        return "", "root"
        
    # 2. Automatically detect latest version in models/
    if os.path.exists("models") and os.path.isdir("models"):
        versions = []
        for name in os.listdir("models"):
            if name.startswith("v") and os.path.isdir(os.path.join("models", name)):
                try:
                    v_num = int(name[1:])
                    versions.append((v_num, name))
                except ValueError:
                    continue
        if versions:
            versions.sort(reverse=True)
            latest_dir = versions[0][1]
            return os.path.join("models", latest_dir), latest_dir
            
    # 3. Fallback to root files
    return "", "v1"

# Load model, vectorizer, and label encoder
model_dir, active_version = get_model_path_and_version()

print(f"Loading model version '{active_version}' from '{model_dir or '.'}'...")

def load_asset(filename):
    path = os.path.join(model_dir, filename) if model_dir else filename
    if not os.path.exists(path):
        # Fallback to root files if not found in folder
        if os.path.exists(filename):
            return joblib.load(filename)
        raise FileNotFoundError(f"Model asset {filename} not found at {path} or root.")
    return joblib.load(path)

model      = load_asset("spam_model.pkl")
vectorizer = load_asset("tfidf.pkl")
encoder    = load_asset("label_encoder.pkl")

stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_v1(text):
    text   = text.lower()
    text   = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

def preprocess_v2(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' __url__ ', text)
    text = re.sub(r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b', ' __email__ ', text)
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', ' __phone__ ', text)
    text = re.sub(r'\b\d{5,6}\b', ' __phone__ ', text)
    text = re.sub(r'[\$\£\€\¥]\s?\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s?[\$\£\€\¥]', ' __money__ ', text)
    text = re.sub(r'\b[a-z]+\d+\w*|\b\d+[a-z]+\w*\b', ' __promo__ ', text)
    text = re.sub(r'[^a-z0-9\s_]', ' ', text)
    tokens = word_tokenize(text)
    processed_tokens = []
    for w in tokens:
        if w.startswith('__') and w.endswith('__'):
            processed_tokens.append(w)
        elif w not in stop_words and len(w) > 1:
            processed_tokens.append(stemmer.stem(w))
    return ' '.join(processed_tokens)

def preprocess(text):
    # Dynamically select preprocessing based on loaded vectorizer vocabulary
    has_special_tokens = False
    if hasattr(vectorizer, 'vocabulary_'):
        has_special_tokens = any(term in vectorizer.vocabulary_ for term in ["__url__", "__phone__", "__money__"])
    
    if has_special_tokens:
        return preprocess_v2(text)
    return preprocess_v1(text)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "SMS Spam API is running ✅",
        "active_model_version": active_version
    })

@app.route("/model-metadata", methods=["GET"])
def model_metadata():
    num_features = len(vectorizer.get_feature_names_out()) if hasattr(vectorizer, 'get_feature_names_out') else len(vectorizer.vocabulary_)
    
    # Check preprocessing features
    has_v2_features = hasattr(vectorizer, 'vocabulary_') and any(term in vectorizer.vocabulary_ for term in ["__url__", "__phone__"])
    
    return jsonify({
        "active_version": active_version,
        "model_type": type(model).__name__,
        "vectorizer_type": type(vectorizer).__name__,
        "num_features": num_features,
        "features_supported": {
            "robust_preprocessing_v2": has_v2_features,
            "url_handling": has_v2_features,
            "phone_handling": has_v2_features,
            "money_handling": has_v2_features,
            "email_handling": has_v2_features,
            "promo_code_handling": has_v2_features
        }
    })

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
        "is_spam":    label == "spam",
        "model_version": active_version
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
