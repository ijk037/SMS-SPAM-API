import os
import re
import json
import urllib.request
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# NLTK downloads
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
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

def download_dataset():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "sms.tsv")
    
    if not os.path.exists(filepath):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")
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

# CNN Model Definition
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
        # text: [batch_size, seq_len]
        embedded = self.embedding(text) # [batch_size, seq_len, embedding_dim]
        embedded = embedded.permute(0, 2, 1) # [batch_size, embedding_dim, seq_len]
        
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

def main():
    filepath = download_dataset()
    messages, labels = load_data(filepath)
    
    print(f"Dataset loaded. Total samples: {len(messages)}")
    
    # 1. Preprocess
    print("Preprocessing messages...")
    processed_messages = [preprocess(msg) for msg in messages]
    
    # 2. Train-test split
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        processed_messages, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Encode labels
    label_map = {"ham": 0, "spam": 1}
    y_train = np.array([label_map[l] for l in y_train_raw])
    y_test = np.array([label_map[l] for l in y_test_raw])
    
    # 3. Build Vocab
    vocab = {"<pad>": 0, "<unk>": 1}
    for text in X_train_raw:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)
                
    print(f"Vocabulary size: {len(vocab)}")
    
    # 4. Tokenize & Pad
    max_len = 50
    def text_to_sequence(texts):
        sequences = []
        for text in texts:
            seq = [vocab.get(word, 1) for word in text.split()]
            if len(seq) < max_len:
                seq = seq + [0] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            sequences.append(seq)
        return np.array(sequences)
        
    X_train = text_to_sequence(X_train_raw)
    X_test = text_to_sequence(X_test_raw)
    
    # 5. Create PyTorch datasets & dataloaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 6. Instantiate Model
    vocab_size = len(vocab)
    embedding_dim = 100
    num_filters = 100
    filter_sizes = [3, 4, 5]
    
    model = TextCNN(vocab_size, embedding_dim, num_filters, filter_sizes)
    
    # Cross Entropy with weights because classes are imbalanced
    ham_weight = 1.0
    spam_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    class_weights = torch.tensor([ham_weight, spam_weight], dtype=torch.float32)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 7. Training loop
    epochs = 10
    print("Training CNN model...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for texts, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}")
        
    # 8. Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for texts, targets in test_loader:
            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_targets.extend(targets.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, pos_label=1)
    rec = recall_score(all_targets, all_preds, pos_label=1)
    f1 = f1_score(all_targets, all_preds, pos_label=1)
    
    print("\nTextCNN Evaluation:")
    print(classification_report(all_targets, all_preds, target_names=["ham", "spam"]))
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (Spam): {prec:.4f}")
    print(f"Recall (Spam):    {rec:.4f}")
    print(f"F1-Score (Spam):  {f1:.4f}")
    
    # 9. Save
    os.makedirs("models/v3", exist_ok=True)
    torch.save(model.state_dict(), "models/v3/cnn_model.pth")
    with open("models/v3/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
        
    with open("models/v3/label_encoder.json", "w", encoding="utf-8") as f:
        json.dump({0: "ham", 1: "spam"}, f)
        
    # Save training metrics to a text file for metadata check
    metrics = {
        "model_type": "TextCNN",
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "num_filters": num_filters,
        "filter_sizes": filter_sizes,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }
    with open("models/v3/model_info.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f)
        
    print("\nSaved CNN model state dict, vocab.json, and metadata to models/v3/")

if __name__ == "__main__":
    main()
