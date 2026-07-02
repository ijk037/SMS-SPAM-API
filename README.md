# SMS Spam API & Web Dashboard

A Flask REST API and interactive web dashboard that classifies SMS messages as **spam** or **ham** using advanced NLP preprocessing and machine learning (Random Forest & Logistic Regression).

---

## Features
- **Robust Preprocessing v2**: Normalizes URLs, emails, phone numbers, money/currency, and alphanumeric promo codes before tokenization.
- **Dynamic Model Versioning**: Automatically loads the latest model version from the `models/` directory or resolves a locked version via the `MODEL_VERSION` environment variable.
- **Dual Client Support (Content Negotiation)**: The root URL (`/`) serves a rich web dashboard when visited in a browser, and a JSON health check when accessed by API clients.
- **Interactive Web Dashboard**:
  - **Single Prediction**: Paste any message to get real-time verdicts (SPAM vs HAM), classification confidence, and a visual list of extracted features.
  - **Side-by-Side Model Comparison**: Compare predictions from Version 1 (Logistic Regression) and Version 2 (Random Forest) side-by-side, complete with details about active algorithms, feature sizes, and preprocessing rules.

---

## Run Locally

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server**:
   ```bash
   python app.py
   ```

3. **Access the application**:
   - Open a browser and visit: `http://localhost:5000/` to launch the **Interactive Web Dashboard**.
   - Make API requests directly to the endpoints (e.g. using `curl` or Postman).

### Using Docker

If you prefer running the application inside a container, you can build and run it using **Docker** and **Docker Compose**:

1. **Start the containerized API and Dashboard**:
   ```bash
   docker-compose up -d --build
   ```

2. **Check container logs**:
   ```bash
   docker-compose logs -f
   ```

3. **Stop the container**:
   ```bash
   docker-compose down
   ```

- **Persistence**: The `docker-compose.yml` mounts the `./data` and `./models` directories. Any reported feedback is saved to your host filesystem, and retraining models on the host automatically updates the container.
- **Health check**: The container includes an automatic health check querying `http://localhost:5000/` every 30 seconds.

---

## API Endpoints

### `GET /`
- **Browser Output**: Renders the HTML Web Dashboard.
- **API Output** (requests with headers `Accept: application/json`):
  ```json
  {
    "active_model_version": "v2",
    "status": "SMS Spam API is running ✅"
  }
  ```

### `GET /model-metadata`
Retrieve active model metrics, classifier type, features count, and preprocessing flags.
- **Optional Query Parameter**: `version` (e.g. `/model-metadata?version=v1`) to inspect a specific cached model version.
- **Response**:
  ```json
  {
    "active_version": "v2",
    "features_supported": {
      "email_handling": true,
      "money_handling": true,
      "phone_handling": true,
      "promo_code_handling": true,
      "robust_preprocessing_v2": true,
      "url_handling": true
    },
    "model_type": "RandomForestClassifier",
    "num_features": 3000,
    "vectorizer_type": "TfidfVectorizer"
  }
  ```

### `POST /predict`
Classify an SMS message.
- **Request Body**:
  ```json
  { 
    "message": "URGENT! Call 09061701461 to claim your GBP1000 prize now!",
    "model_version": "v2" 
  }
  ```
- **Response**:
  ```json
  {
    "message": "URGENT! Call 09061701461 to claim your GBP1000 prize now!",
    "processed_message": "urgent call __phone__ claim gbp1000 prize",
    "prediction": "spam",
    "confidence": "90.0%",
    "is_spam": true,
    "model_version": "v2",
    "explanations": [
      { "word": "urgent", "spam_score": 0.45 },
      { "word": "__phone__", "spam_score": 0.28 },
      { "word": "claim", "spam_score": 0.12 },
      { "word": "call", "spam_score": 0.05 }
    ]
  }
  ```

### `POST /feedback`
Submit user feedback for misclassified messages.
- **Request Body**:
  ```json
  {
    "message": "This is a reported spam message",
    "reported_label": "spam"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Feedback saved successfully"
  }
  ```
- **Behavior**: Appends the raw reported sample to `data/feedback.tsv` which will be automatically merged into the training dataset during the next model retraining cycle.

---

## API Security & Rate Limiting

To simulate a production-grade SaaS API, security features are built-in:

### 1. API Key Authentication
- Protected endpoints (`/predict`, `/feedback`, `/model-metadata`) require a valid API key.
- Provide the key using either:
  - Header: `X-API-Key: <your-key>`
  - Query Parameter: `?api_key=<your-key>`
  - JSON Body key: `"api_key": "<your-key>"`
- Valid local keys: `demo-key-123` (default in Dashboard) and `sms-shield-secure-key-2026`.

### 2. Rate Limiting
- Requests to protected API endpoints are limited to **15 requests per minute per IP address**.
- Exceeding this limit returns a `429 Too Many Requests` status code.
- Visited directly in the browser, the main HTML page rendering (`GET /`) is exempted to prevent lockouts.

---

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

### Switching Default Version
By default, the API automatically loads the latest model version found under `models/` (e.g., `v2`). You can lock the server's default startup version using the `MODEL_VERSION` environment variable:
```bash
# On Windows PowerShell
$env:MODEL_VERSION="v1"
python app.py

# On Linux/macOS
MODEL_VERSION=v1 python app.py
```

---

## Running Tests
Run the API integration tests:
```bash
python test_app.py
```

---

## Tech Stack
- **Backend**: Python, Flask, scikit-learn, NLTK
- **Frontend**: HTML5, Vanilla CSS3 (Custom Glassmorphism theme, dynamic animated components), Javascript (Async fetch)
- **Active Model (v2)**: Random Forest Classifier
- **Baseline Model (v1)**: Logistic Regression
- **Vectorizer**: TF-IDF Vectorizer (3000 features)


