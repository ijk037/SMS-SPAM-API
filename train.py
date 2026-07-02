import os
import re
import urllib.request
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import csv

# Download NLTK requirements
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    if not isinstance(text, str):
        return ""
    # 1. Lowercase
    text = text.lower()
    
    # 2. Replace URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' __url__ ', text)
    
    # 3. Replace Emails
    text = re.sub(r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b', ' __email__ ', text)
    
    # 4. Replace Phone Numbers (typical format, and also 5-6 digit shortcodes)
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', ' __phone__ ', text)
    text = re.sub(r'\b\d{5,6}\b', ' __phone__ ', text)
    
    # 5. Replace Currency symbols followed by numbers or vice versa
    text = re.sub(r'[\$\£\€\¥]\s?\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s?[\$\£\€\¥]', ' __money__ ', text)
    
    # 6. Replace alphanumeric codes common in spam (promo codes like FREE100, WIN50)
    text = re.sub(r'\b[a-z]+\d+\w*|\b\d+[a-z]+\w*\b', ' __promo__ ', text)
    
    # 7. Remove general punctuation except letters, numbers, spaces, and underscores
    text = re.sub(r'[^a-z0-9\s_]', ' ', text)
    
    # 8. Tokenize
    tokens = word_tokenize(text)
    
    # 9. Stopwords and Stemming, keeping special tokens
    processed_tokens = []
    for w in tokens:
        if w.startswith('__') and w.endswith('__'):
            processed_tokens.append(w)
        elif w not in stop_words and len(w) > 1:
            processed_tokens.append(stemmer.stem(w))
            
    return ' '.join(processed_tokens)

def download_dataset():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "sms.tsv")
    
    if not os.path.exists(filepath):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")
    else:
        print("Dataset already exists locally.")
    return filepath

def load_data(filepath):
    messages = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                labels.append(row[0])
                messages.append(row[1])
    return messages, labels

def main():
    # 1. Download and load data
    filepath = download_dataset()
    messages, labels = load_data(filepath)
    
    total_samples = len(messages)
    spam_count = sum(1 for l in labels if l == "spam")
    ham_count = sum(1 for l in labels if l == "ham")
    
    print(f"Dataset loaded. Total samples: {total_samples}")
    print(f"Ham: {ham_count} ({ham_count/total_samples*100:.1f}%)")
    print(f"Spam: {spam_count} ({spam_count/total_samples*100:.1f}%)")
    
    # 2. Preprocess
    print("Preprocessing messages (this may take a moment)...")
    processed_messages = [preprocess(msg) for msg in messages]
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        processed_messages, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Encode labels
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    
    # 4. Vectorize (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 5. Define classifiers
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    trained_models = {}
    
    print("\nTraining and evaluating models...")
    spam_val = int(encoder.transform(["spam"])[0])
    
    for name, clf in models.items():
        clf.fit(X_train_vec, y_train_encoded)
        preds = clf.predict(X_test_vec)
        
        acc = accuracy_score(y_test_encoded, preds)
        prec = precision_score(y_test_encoded, preds, pos_label=spam_val)
        rec = recall_score(y_test_encoded, preds, pos_label=spam_val)
        f1 = f1_score(y_test_encoded, preds, pos_label=spam_val)
        
        print(f"\n{name} Results:")
        print(classification_report(y_test_encoded, preds, target_names=encoder.classes_))
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision (Spam)": prec,
            "Recall (Spam)": rec,
            "F1-Score (Spam)": f1
        })
        trained_models[name] = clf
        
    print("\nComparison:")
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)
    for res in results:
        print(f"{res['Model']:<25} | {res['Accuracy']:<10.4f} | {res['Precision (Spam)']:<10.4f} | {res['Recall (Spam)']:<10.4f} | {res['F1-Score (Spam)']:<10.4f}")
    
    # Find the best model based on F1-Score of Spam
    best_idx = 0
    best_f1 = -1
    for idx, res in enumerate(results):
        if res["F1-Score (Spam)"] > best_f1:
            best_f1 = res["F1-Score (Spam)"]
            best_idx = idx
            
    best_model_name = results[best_idx]["Model"]
    print(f"\nBest Model based on F1-Score (Spam): {best_model_name}")
    
    best_clf = trained_models[best_model_name]
    
    # Save the best model to models/v2/
    os.makedirs("models/v2", exist_ok=True)
    
    joblib.dump(best_clf, "models/v2/spam_model.pkl")
    joblib.dump(vectorizer, "models/v2/tfidf.pkl")
    joblib.dump(encoder, "models/v2/label_encoder.pkl")
    print("\nSaved best model, vectorizer, and label encoder to models/v2/")

if __name__ == "__main__":
    main()
