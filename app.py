from flask import Flask, request, jsonify, render_template
import joblib
import nltk
import re
import os
import csv
import time
from collections import defaultdict
import json

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

app = Flask(__name__)

# Simple in-memory rate limiter database
rate_limit_records = defaultdict(list)
API_KEYS = {"sms-shield-secure-key-2026", "demo-key-123"}

def check_rate_limit(ip_address, limit=10, period=60):
    now = time.time()
    timestamps = rate_limit_records[ip_address]
    
    # Filter out timestamps older than the period
    active_timestamps = [t for t in timestamps if now - t < period]
    rate_limit_records[ip_address] = active_timestamps
    
    if len(active_timestamps) >= limit:
        return False
    
    rate_limit_records[ip_address].append(now)
    return True

@app.before_request
def enforce_security():
    # Only protect API endpoints (predict, feedback, model-metadata)
    if request.path in ["/predict", "/feedback", "/model-metadata"]:
        # 1. Rate Limiting
        # Extract client IP supporting proxy headers
        ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()
        
        # Limit to 15 requests per minute
        if not check_rate_limit(ip, limit=15, period=60):  # set to 15 to allow test cases + user trials comfortably
            return jsonify({"error": "Rate limit exceeded. Max 15 requests per minute."}), 429
            
        # 2. API Key Authentication
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            # Check query parameters
            api_key = request.args.get("api_key")
        if not api_key and request.is_json:
            # Check JSON payload
            try:
                api_key = request.get_json().get("api_key")
            except Exception:
                pass
                
        if api_key not in API_KEYS:
            return jsonify({"error": "Unauthorized. Invalid or missing X-API-Key header/parameter."}), 401

# Try to import PyTorch for TextCNN (V3)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class TextCNN(nn.Module):
        def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim=2):
            super(TextCNN, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.convs = nn.ModuleList([
                nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
                for fs in filter_sizes
            ])
            self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, text):
            embedded = self.embedding(text) # [batch_size, seq_len, embedding_dim]
            embedded = embedded.permute(0, 2, 1) # [batch_size, embedding_dim, seq_len]
            conved = [F.relu(conv(embedded)) for conv in self.convs]
            pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
            cat = self.dropout(torch.cat(pooled, dim=1))
            return self.fc(cat)
except ImportError:
    TextCNN = None

# Globals for loaded default assets
model = None
vectorizer = None
encoder = None
cnn_model = None
vocab = None
label_map = None

# Helper to dynamically find the latest model version
def get_model_path_and_version():
    requested_version = os.environ.get("MODEL_VERSION", "").strip().lower()
    
    # 1. If a specific version is requested (e.g. 'v1', 'v2', 'v3')
    if requested_version:
        path = os.path.join("models", requested_version)
        if os.path.exists(path) and os.path.isdir(path):
            return path, requested_version
        if requested_version == "root":
            return "", "root"
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

if active_version == "v3":
    if TextCNN is None:
        raise ImportError("PyTorch ('torch') is required to load version v3 model, but it is not installed.")
        
    # Load CNN assets
    vocab_path = os.path.join(model_dir, "vocab.json")
    model_path = os.path.join(model_dir, "cnn_model.pth")
    label_path = os.path.join(model_dir, "label_encoder.json")
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    with open(label_path, "r", encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
        
    # Load info file if present
    info_path = os.path.join(model_dir, "model_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
            vocab_size = info["vocab_size"]
            embedding_dim = info["embedding_dim"]
            num_filters = info["num_filters"]
            filter_sizes = info["filter_sizes"]
    else:
        vocab_size = len(vocab)
        embedding_dim = 100
        num_filters = 100
        filter_sizes = [3, 4, 5]
        
    cnn_model = TextCNN(vocab_size, embedding_dim, num_filters, filter_sizes)
    cnn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    cnn_model.eval()
else:
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

# Cache for dynamically loaded versions (scikit-learn models only)
model_cache = {}
if active_version != "v3":
    model_cache[active_version] = (model, vectorizer, encoder)

def get_cached_assets(version_name):
    version_name = version_name.strip().lower()
    if version_name in model_cache:
        return model_cache[version_name]
    
    # Load and cache
    if version_name == "v1" or version_name == "root":
        m = joblib.load("spam_model.pkl")
        v = joblib.load("tfidf.pkl")
        e = joblib.load("label_encoder.pkl")
    else:
        path = os.path.join("models", version_name)
        if not (os.path.exists(path) and os.path.isdir(path)):
            raise FileNotFoundError(f"Version '{version_name}' not found under models/")
        m = joblib.load(os.path.join(path, "spam_model.pkl"))
        v = joblib.load(os.path.join(path, "tfidf.pkl"))
        e = joblib.load(os.path.join(path, "label_encoder.pkl"))
        
    model_cache[version_name] = (m, v, e)
    return m, v, e

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

def preprocess(text, target_vectorizer=None):
    if active_version == "v3":
        return preprocess_v2(text)
        
    vec = target_vectorizer if target_vectorizer is not None else vectorizer
    has_special_tokens = False
    if vec and hasattr(vec, 'vocabulary_'):
        has_special_tokens = any(term in vec.vocabulary_ for term in ["__url__", "__phone__", "__money__"])
    
    if has_special_tokens:
        return preprocess_v2(text)
    return preprocess_v1(text)

@app.route("/", methods=["GET"])
def home():
    # Content negotiation: return JSON for API clients / test suite, HTML for browsers
    accept_header = request.headers.get("Accept", "")
    if "text/html" not in accept_header:
        return jsonify({
            "status": "SMS Spam API is running ✅",
            "active_model_version": active_version
        })
    return render_template("index.html")

@app.route("/model-metadata", methods=["GET"])
def model_metadata():
    requested_version = request.args.get("version", active_version).strip().lower()
    
    if requested_version == "v3" or (requested_version == "" and active_version == "v3"):
        if TextCNN is None or vocab is None:
            return jsonify({"error": "PyTorch model v3 is not supported or loaded."}), 500
        return jsonify({
            "active_version": "v3",
            "model_type": "TextCNN (PyTorch)",
            "vocab_size": len(vocab),
            "seq_len_limit": 50,
            "features_supported": {
                "robust_preprocessing_v2": True,
                "url_handling": True,
                "phone_handling": True,
                "money_handling": True,
                "email_handling": True,
                "promo_code_handling": True
            }
        })
        
    try:
        req_model, req_vectorizer, req_encoder = get_cached_assets(requested_version)
        version_used = requested_version
    except Exception:
        req_model, req_vectorizer, req_encoder = model, vectorizer, encoder
        version_used = active_version

    num_features = len(req_vectorizer.get_feature_names_out()) if hasattr(req_vectorizer, 'get_feature_names_out') else len(req_vectorizer.vocabulary_)
    has_v2_features = hasattr(req_vectorizer, 'vocabulary_') and any(term in req_vectorizer.vocabulary_ for term in ["__url__", "__phone__"])
    
    return jsonify({
        "active_version": version_used,
        "model_type": type(req_model).__name__,
        "vectorizer_type": type(req_vectorizer).__name__,
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

    requested_version = data.get("model_version", active_version).strip().lower()

    # Special case: requested version is v3
    if requested_version == "v3" or (requested_version == "" and active_version == "v3"):
        if TextCNN is None or cnn_model is None:
            return jsonify({"error": "PyTorch model v3 is not loaded or supported."}), 500
            
        processed = preprocess_v2(message)
        max_len = 50
        seq = [vocab.get(word, 1) for word in processed.split()]
        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
            
        seq_tensor = torch.tensor([seq], dtype=torch.long)
        
        with torch.no_grad():
            outputs = cnn_model(seq_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            
        label = label_map[pred_class]
        confidence = round(float(probs[pred_class].item()) * 100, 1)
        
        # Perturbation explanation for PyTorch CNN
        explanations = []
        orig_prob_spam = probs[1].item() if len(probs) > 1 else 0.0
        words = list(set(processed.split()))
        for w in words:
            modified_processed = " ".join([word for word in processed.split() if word != w])
            m_seq = [vocab.get(word, 1) for word in modified_processed.split()]
            if len(m_seq) < max_len:
                m_seq = m_seq + [0] * (max_len - len(m_seq))
            else:
                m_seq = m_seq[:max_len]
            m_tensor = torch.tensor([m_seq], dtype=torch.long)
            with torch.no_grad():
                m_outputs = cnn_model(m_tensor)
                m_probs = F.softmax(m_outputs, dim=1)[0]
            diff = orig_prob_spam - (m_probs[1].item() if len(m_probs) > 1 else 0.0)
            explanations.append({
                "word": w,
                "spam_score": round(float(diff), 4)
            })
        explanations.sort(key=lambda x: abs(x["spam_score"]), reverse=True)

        return jsonify({
            "message":           message,
            "processed_message": processed,
            "prediction":        label,
            "confidence":        f"{confidence}%",
            "is_spam":           label == "spam",
            "model_version":     "v3",
            "explanations":      explanations
        })

    # Standard scikit-learn paths (v1 / v2)
    try:
        req_model, req_vectorizer, req_encoder = get_cached_assets(requested_version)
        version_used = requested_version
    except Exception:
        req_model, req_vectorizer, req_encoder = model, vectorizer, encoder
        version_used = active_version

    processed  = preprocess(message, req_vectorizer)
    vector     = req_vectorizer.transform([processed])
    prediction = req_model.predict(vector)
    proba      = req_model.predict_proba(vector)[0]
    label      = req_encoder.inverse_transform(prediction)[0]
    confidence = round(float(proba.max()) * 100, 1)

    explanations = []
    if hasattr(req_model, "predict_proba"):
        words = list(set(processed.split()))
        if words:
            try:
                spam_index = list(req_encoder.classes_).index("spam")
            except ValueError:
                spam_index = 1
            
            orig_prob_spam = proba[spam_index]
            for w in words:
                modified_processed = " ".join([word for word in processed.split() if word != w])
                mod_vector = req_vectorizer.transform([modified_processed])
                mod_proba = req_model.predict_proba(mod_vector)[0]
                mod_prob_spam = mod_proba[spam_index]
                diff = orig_prob_spam - mod_prob_spam
                explanations.append({
                    "word": w,
                    "spam_score": round(float(diff), 4)
                })
            explanations.sort(key=lambda x: abs(x["spam_score"]), reverse=True)

    return jsonify({
        "message":           message,
        "processed_message": processed,
        "prediction":        label,
        "confidence":        f"{confidence}%",
        "is_spam":           label == "spam",
        "model_version":     version_used,
        "explanations":      explanations
    })

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    if not data or "message" not in data or "reported_label" not in data:
        return jsonify({"error": "Send JSON with 'message' and 'reported_label' keys"}), 400

    message = data["message"].strip()
    reported_label = data["reported_label"].strip().lower()

    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400
        
    if reported_label not in ["spam", "ham"]:
        return jsonify({"error": "reported_label must be 'spam' or 'ham'"}), 400

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    feedback_file = os.path.join("data", "feedback.tsv")

    try:
        # Append to TSV file
        with open(feedback_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([reported_label, message])
    except Exception as e:
        return jsonify({"error": f"Failed to save feedback: {str(e)}"}), 500

    return jsonify({
        "status": "success",
        "message": "Feedback saved successfully"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
