import os
import json
import sys

# Ensure current directory is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import app, rate_limit_records

# Ensure console supports UTF-8 characters on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set up test client
client = app.test_client()
HEADERS = {"X-API-Key": "demo-key-123"}

def test_home():
    print("Testing GET /...")
    response = client.get("/")
    assert response.status_code == 200
    data = json.loads(response.data)
    print("Home response:", data)
    assert "active_model_version" in data
    print("GET / passed! ✅")

def test_metadata():
    print("\nTesting GET /model-metadata...")
    # Explicitly request version 'v2' metadata so the assertions are robust to newer default models (e.g. v3)
    response = client.get("/model-metadata?version=v2", headers=HEADERS)
    assert response.status_code == 200
    data = json.loads(response.data)
    print("Metadata response:", json.dumps(data, indent=2))
    assert data["active_version"] == "v2"
    assert data["features_supported"]["robust_preprocessing_v2"] is True
    print("GET /model-metadata passed! ✅")

def test_predictions():
    print("\nTesting POST /predict...")
    
    test_cases = [
        {"message": "Hello, how are you doing today?", "expected": "ham"},
        {"message": "URGENT! You have won a 1-week FREE membership to our prize draw! Call 09061701461 to claim your GBP1000 prize now!", "expected": "spam"},
        {"message": "Go to http://spamlink.com to win USD500 free cash! Reply STOP to cancel.", "expected": "spam"}
    ]
    
    for case in test_cases:
        print(f"Predicting for message: '{case['message']}'")
        # Explicitly request model_version 'v2' to ensure predictions match expected values
        response = client.post("/predict", 
                               data=json.dumps({
                                   "message": case["message"],
                                   "model_version": "v2"
                               }),
                               headers=HEADERS,
                               content_type="application/json")
        assert response.status_code == 200
        data = json.loads(response.data)
        print(f"Prediction: {data['prediction']} (Confidence: {data['confidence']})")
        assert data["prediction"] == case["expected"]
        print("Sub-test passed! ✅")
        
    print("POST /predict passed! ✅")

def test_feedback():
    print("\nTesting POST /feedback...")
    response = client.post("/feedback",
                           data=json.dumps({
                               "message": "This is a reported spam message",
                               "reported_label": "spam"
                           }),
                           headers=HEADERS,
                           content_type="application/json")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    print("POST /feedback passed! ✅")

def test_unauthorized():
    print("\nTesting API Key Authorization Security...")
    # Test missing key
    response = client.get("/model-metadata")
    assert response.status_code == 401
    
    # Test invalid key
    response = client.post("/predict",
                           data=json.dumps({"message": "Hello"}),
                           headers={"X-API-Key": "wrong-key"},
                           content_type="application/json")
    assert response.status_code == 401
    print("Authentication checks passed! ✅")

def test_rate_limiting():
    print("\nTesting Rate Limiting Security...")
    # Reset records to guarantee exactly 15 requests before lockout
    rate_limit_records.clear()
    
    # Since our limit is set to 15, let's fire 15 requests, which should succeed.
    for i in range(15):
        response = client.post("/predict",
                               data=json.dumps({"message": f"Test message {i}"}),
                               headers=HEADERS,
                               content_type="application/json")
        assert response.status_code == 200

    # The 16th request should fail with 429 Too Many Requests
    response = client.post("/predict",
                           data=json.dumps({"message": "Rate limited request"}),
                           headers=HEADERS,
                           content_type="application/json")
    assert response.status_code == 429
    data = json.loads(response.data)
    assert "Rate limit exceeded" in data["error"]
    print("Rate limiting checks passed! ✅")

if __name__ == "__main__":
    try:
        test_home()
        test_metadata()
        test_predictions()
        test_feedback()
        test_unauthorized()
        test_rate_limiting()
        print("\nAll API security and functional tests passed successfully! 🚀")
    except AssertionError as e:
        print("\nAssertion failed ❌", e)
        sys.exit(1)
