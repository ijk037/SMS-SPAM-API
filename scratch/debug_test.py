import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app import app
client = app.test_client()
HEADERS = {"X-API-Key": "demo-key-123"}

response = client.get("/model-metadata", headers=HEADERS)
print("Status Code:", response.status_code)
print("Data:", response.data.decode('utf-8'))
