import os
import json
import sys
import importlib
import traceback

# Ensure current directory is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Ensure console supports UTF-8 characters on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {"X-API-Key": "demo-key-123"}

def run_tests_for_version(version):
    print(f"\n==================================================")
    print(f" TESTING MODEL VERSION: {version} ")
    print(f"==================================================")
    
    os.environ["MODEL_VERSION"] = version
    
    # Reload the app module so it re-runs model loading and loads the requested version
    import app
    importlib.reload(app)
    
    client = app.app.test_client()
    
    # 1. Test Home
    print("Testing GET /...")
    response = client.get("/")
    assert response.status_code == 200
    home_data = json.loads(response.data)
    print("Home response:", home_data)
    assert home_data["active_model_version"] == version
    print("GET / passed! ✅")
    
    # 2. Test Metadata
    print("\nTesting GET /model-metadata...")
    response = client.get("/model-metadata", headers=HEADERS)
    if response.status_code != 200:
        print("FAIL DATA:", response.data.decode('utf-8'))
    assert response.status_code == 200
    metadata_data = json.loads(response.data)
    print("Metadata response:", json.dumps(metadata_data, indent=2))
    assert metadata_data["active_version"] == version
    
    if version in ["v2", "v3"]:
        assert metadata_data["features_supported"]["robust_preprocessing_v2"] is True
    else:
        assert metadata_data["features_supported"]["robust_preprocessing_v2"] is False
    print("GET /model-metadata passed! ✅")
    
    # 3. Test Predictions
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
                               headers=HEADERS,
                               content_type="application/json")
        if response.status_code != 200:
            print("FAIL DATA:", response.data.decode('utf-8'))
        assert response.status_code == 200
        pred_data = json.loads(response.data)
        print(f"Prediction: {pred_data['prediction']} (Confidence: {pred_data['confidence']})")
        assert pred_data["prediction"] == case["expected"]
        assert pred_data["model_version"] == version
        print("Sub-test passed! ✅")
        
    print(f"POST /predict passed for {version}! ✅")

def test_feedback():
    import app
    client = app.app.test_client()
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
    import app
    client = app.app.test_client()
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
    import app
    client = app.app.test_client()
    print("\nTesting Rate Limiting Security...")
    # Reset records to guarantee exactly 15 requests before lockout
    app.rate_limit_records.clear()
    
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
    versions = ["v1", "v2", "v3"]
    success = True
    
    for v in versions:
        try:
            run_tests_for_version(v)
        except AssertionError as e:
            print(f"\n❌ Assertion failed for version {v}:", e)
            traceback.print_exc()
            success = False
        except Exception as e:
            print(f"\n❌ Unexpected error for version {v}:", e)
            traceback.print_exc()
            success = False
            
    if success:
        try:
            test_feedback()
            test_unauthorized()
            test_rate_limiting()
            print("\n🎉 All tests for ALL model versions (v1, v2, v3) & security gates passed successfully! 🚀")
            if "MODEL_VERSION" in os.environ:
                del os.environ["MODEL_VERSION"]
            sys.exit(0)
        except AssertionError as e:
            print("\n❌ Security test assertion failed:")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n❌ Some functional tests failed.")
        sys.exit(1)
