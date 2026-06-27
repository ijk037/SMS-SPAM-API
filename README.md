# SMS Spam API

A Flask REST API that classifies SMS messages as **spam** or **ham** using Logistic Regression + TF-IDF.

## Endpoints

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
pip install -r requirements.txt
python app.py
```

Test it:
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
