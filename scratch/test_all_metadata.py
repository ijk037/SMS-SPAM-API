import sys
import os
import json
import importlib

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

versions = ["v1", "v2", "v3"]

for v in versions:
    print(f"\nTargeting version: {v}")
    os.environ["MODEL_VERSION"] = v
    import app
    importlib.reload(app)
    client = app.app.test_client()
    HEADERS = {"X-API-Key": "demo-key-123"}
    
    response = client.get("/model-metadata", headers=HEADERS)
    print("Status:", response.status_code)
    print("Body:", response.data.decode('utf-8'))
