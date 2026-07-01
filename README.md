# SMS Spam API

A Flask REST API that classifies SMS messages as **spam** or **ham** using Logistic Regression + TF-IDF.

## Endpoints

### `GET /`
Health check and active model version.

### `GET /model-metadata`
Retrieve active model metrics, classifier type, features count, and preprocessing flags.

### `POST /predict`
Classify a message. Includes `"model_version"` in the JSON response.

**Request:**
```json
{ "message": "You won a FREE prize! Click now!" }
```

**Response:**
```json
{
  "message": "You won a FREE prize! Click now!",
  "prediction": "spam",
  "confidence": "97.3%",
  "is_spam": true,
  "model_version": "v2"
}
```

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Test it:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Claim your free prize now!"}'
```

## Model Training & Versioning

To retrain the model and test multiple classifiers:
```bash
python train.py
```
This script will:
1. Download the UCI SMS Spam Collection dataset.
2. Apply **Robust Preprocessing v2** (capturing emails, phone numbers, URLs, currency, and promo codes).
3. Evaluate and compare **Multinomial Naive Bayes**, **Logistic Regression**, and **Random Forest**.
4. Automatically save the best performing model (based on Spam F1-Score) to `models/v2/`.

### Switching Versions
By default, the API automatically loads the latest model version found under `models/` (e.g., `v2`). You can lock the server to a specific version using the `MODEL_VERSION` environment variable:
```bash
# On Windows PowerShell
$env:MODEL_VERSION="v1"
python app.py

# On Linux/macOS
MODEL_VERSION=v1 python app.py
```

## Running Tests
Run the API integration tests:
```bash
python test_app.py
```

## Tech Stack (Updated)
- Python, Flask, scikit-learn, NLTK
- Active Model: Random Forest (v2) / Logistic Regression (v1)
- Vectorizer: TF-IDF (3000 features)
- Preprocessing: URL/Email/Phone/Money regex normalization + tokenization + stopword removal + stemming
