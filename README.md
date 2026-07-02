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
Health check.

### `POST /predict`
Classify a message.

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
  "is_spam": true
}
```

## Run locally

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
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Claim your free prize now!"}'
```

## Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set **Start Command**: `gunicorn app:app`
5. Deploy — you'll get a public URL

## Tech Stack
- Python, Flask, scikit-learn, NLTK
- Model: Logistic Regression
- Vectorizer: TF-IDF (3000 features)
