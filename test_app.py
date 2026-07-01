import os
import json
import sys
from app import app

# Ensure console supports UTF-8 characters on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set up test client
client = app.test_client()

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
    response = client.get("/model-metadata")
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
        response = client.post("/predict", 
                               data=json.dumps({"message": case["message"]}),
                               content_type="application/json")
        assert response.status_code == 200
        data = json.loads(response.data)
        print(f"Prediction: {data['prediction']} (Confidence: {data['confidence']})")
        assert data["prediction"] == case["expected"]
        print("Sub-test passed! ✅")
        
    print("POST /predict passed! ✅")

if __name__ == "__main__":
    try:
        test_home()
        test_metadata()
        test_predictions()
        print("\nAll API tests passed successfully! 🚀")
    except AssertionError as e:
        print("\nAssertion failed ❌", e)
