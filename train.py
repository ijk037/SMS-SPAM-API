import os
import re
import urllib.request
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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

def normalize_obfuscation(text):
    text = text.lower()
    replacements = {
        '@': 'a', '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '8': 'b', 
        '|': 'i', '$': 's', '£': 'l', '€': 'e', '¥': 'y', '!': 'i'
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
        
    text = re.sub(r'\b([a-z])(?:\s+([a-z]))+\b', lambda m: m.group(0).replace(" ", ""), text)
    return text

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = normalize_obfuscation(text)
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

def extract_meta_features(text_list):
    meta = []
    for text in text_list:
        length = len(text)
        cap_ratio = sum(1 for c in text if c.isupper()) / (length + 1)
        digit_count = sum(1 for c in text if c.isdigit())
        num_special = sum(1 for c in text if not c.isalnum() and not c.isspace())
        meta.append([length, cap_ratio, digit_count, num_special])
        
    return sp.csr_matrix(meta)

def augment_spam_obfuscations(messages, labels, ratio=0.25):
    augmented_messages = []
    augmented_labels = []
    
    obfuscations_map = {'a': '@', 'o': '0', 'i': '1', 'e': '3', 's': '$'}
    
    for msg, lbl in zip(messages, labels):
        if lbl == "spam" and np.random.rand() < ratio:
            chars = list(msg)
            mutated = False
            for idx, char in enumerate(chars):
                char_lower = char.lower()
                if char_lower in obfuscations_map and np.random.rand() < 0.4:
                    chars[idx] = obfuscations_map[char_lower]
                    mutated = True
            if mutated:
                augmented_messages.append("".join(chars))
                augmented_labels.append("spam")
                
    return augmented_messages, augmented_labels

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
    filepath = download_dataset()
    messages, labels = load_data(filepath)
    
    # Check for user reported feedback data and merge it
    feedback_path = os.path.join("data", "feedback.tsv")
    if os.path.exists(feedback_path):
        print(f"\nFeedback file found: {feedback_path}")
        try:
            fb_messages, fb_labels = load_data(feedback_path)
            fb_count = len(fb_messages)
            if fb_count > 0:
                print(f"Merging {fb_count} user-reported feedback samples into training set...")
                messages.extend(fb_messages)
                labels.extend(fb_labels)
        except Exception as e:
            print(f"Warning: Failed to load feedback data: {str(e)}")
            
    # Apply synthetic data augmentation to make models robust against obfuscation tricks
    print("\nApplying spelling-obfuscation data augmentation for SPAM class...")
    aug_msgs, aug_lbls = augment_spam_obfuscations(messages, labels)
    if aug_msgs:
        print(f"Generated {len(aug_msgs)} synthetic obfuscated SPAM samples.")
        messages.extend(aug_msgs)
        labels.extend(aug_lbls)
    
    total_samples = len(messages)
    spam_count = sum(1 for l in labels if l == "spam")
    ham_count = sum(1 for l in labels if l == "ham")
    
    print(f"Dataset loaded. Total samples: {total_samples}")
    print(f"Ham: {ham_count} ({ham_count/total_samples*100:.1f}%)")
    print(f"Spam: {spam_count} ({spam_count/total_samples*100:.1f}%)")
    
    # Keep copies of raw messages for metadata extraction *before* preprocessing
    raw_messages = list(messages)
    
    # 2. Preprocess text
    print("Preprocessing messages (this may take a moment)...")
    processed_messages = [preprocess(msg) for msg in messages]
    
    # 3. Split data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        raw_messages, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Apply preprocessing to training and testing sets
    X_train_processed = [preprocess(msg) for msg in X_train_raw]
    X_test_processed = [preprocess(msg) for msg in X_test_raw]
    
    # Encode labels
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    
    # 4. Vectorize text (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train_processed)
    X_test_vec = vectorizer.transform(X_test_processed)
    
    # 5. Extract structural meta features (length, caps, digits, special symbols)
    print("Extracting message metadata features...")
    X_train_meta = extract_meta_features(X_train_raw)
    X_test_meta = extract_meta_features(X_test_raw)
    
    # Combine TF-IDF features with meta features
    X_train_combined = sp.hstack([X_train_vec, X_train_meta])
    X_test_combined = sp.hstack([X_test_vec, X_test_meta])
    
    # 6. Define classifiers (Logistic Regression, Multinomial NB, Random Forest)
    clf1 = MultinomialNB()
    clf2 = LogisticRegression(max_iter=1000, random_state=42)
    clf3 = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create soft voting ensemble classifier
    ensemble = VotingClassifier(
        estimators=[
            ('Naive Bayes', clf1),
            ('Logistic Regression', clf2),
            ('Random Forest', clf3)
        ],
        voting='soft'
    )
    
    models = {
        "Multinomial Naive Bayes": clf1,
        "Logistic Regression": clf2,
        "Random Forest": clf3,
        "Voting Ensemble (Soft)": ensemble
    }
    
    results = []
    trained_models = {}
    
    print("\nTraining and evaluating models on combined text + metadata features...")
    spam_val = int(encoder.transform(["spam"])[0])
    
    for name, clf in models.items():
        # MultinomialNB does not support negative meta features (our cap ratio and counts are positive, so it fits perfectly!)
        clf.fit(X_train_combined, y_train_encoded)
        preds = clf.predict(X_test_combined)
        
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
    
    # Save the best model, vectorizer, and encoder to models/v2/
    os.makedirs("models/v2", exist_ok=True)
    joblib.dump(best_clf, "models/v2/spam_model.pkl")
    joblib.dump(vectorizer, "models/v2/tfidf.pkl")
    joblib.dump(encoder, "models/v2/label_encoder.pkl")
    print("\nSaved best model, vectorizer, and label encoder to models/v2/")

if __name__ == "__main__":
    main()
