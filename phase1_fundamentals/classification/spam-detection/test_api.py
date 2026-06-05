"""
Test script for FastAPI endpoints
Run: python test_api.py
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_root():
    """Test root endpoint"""
    print("\n1️⃣  Testing Root Endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200

def test_health():
    """Test health check endpoint"""
    print("\n2️⃣  Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200

def test_models_info():
    """Test models info endpoint"""
    print("\n3️⃣  Testing Models Info...")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200

def test_single_prediction_spam():
    """Test single prediction - SPAM message"""
    print("\n4️⃣  Testing Single Prediction (SPAM)...")

    data = {
        "message": "WINNER!! You have won $1000 cash prize! Call now 123-456-7890"
    }

    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()

    print(f"\n📧 Message: {data['message']}")
    print(f"🧹 Cleaned: {result['cleaned_message']}")
    print(f"⚖️  Verdict: {result['final_verdict']}")
    print(f"📊 Consensus: {result['consensus']}")
    print(f"⏱️  Processing Time: {result['processing_time_ms']}ms")
    print("\nModel Predictions:")
    for model, pred in result['predictions'].items():
        conf = f"{pred['confidence']:.1f}%" if pred['confidence'] else "N/A"
        print(f"  - {model.replace('_', ' ').title()}: {pred['prediction'].upper()} (confidence: {conf})")

    assert response.status_code == 200
    assert result['success'] == True

def test_single_prediction_ham():
    """Test single prediction - HAM message"""
    print("\n5️⃣  Testing Single Prediction (HAM)...")

    data = {
        "message": "Hey, can we meet for coffee tomorrow at 3pm?"
    }

    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()

    print(f"\n📧 Message: {data['message']}")
    print(f"🧹 Cleaned: {result['cleaned_message']}")
    print(f"⚖️  Verdict: {result['final_verdict']}")
    print(f"📊 Consensus: {result['consensus']}")
    print(f"⏱️  Processing Time: {result['processing_time_ms']}ms")

    assert response.status_code == 200
    assert result['success'] == True

def test_batch_prediction():
    """Test batch prediction"""
    print("\n6️⃣  Testing Batch Prediction...")

    messages = [
        "URGENT! Your account will be closed. Click here now!",
        "Hi, let's have lunch tomorrow",
        "Congratulations! You won a FREE cruise!",
        "Meeting rescheduled to 2pm"
    ]

    response = requests.post(f"{BASE_URL}/predict/batch", json=messages)
    print(f"Status: {response.status_code}")
    result = response.json()

    print(f"\n📊 Batch Results:")
    for i, res in enumerate(result['results'], 1):
        print(f"\n  {i}. Message: {res['message'][:50]}...")
        print(f"     Verdict: {res['verdict']}")
        print(f"     Consensus: {res['consensus']}")

    assert response.status_code == 200
    assert result['total_messages'] == len(messages)

def test_invalid_request():
    """Test with invalid input"""
    print("\n7️⃣  Testing Invalid Request...")

    # Empty message
    data = {"message": ""}
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 422  # Validation error

def test_batch_limit():
    """Test batch size limit"""
    print("\n8️⃣  Testing Batch Size Limit...")

    # Send more than 100 messages
    messages = ["test message"] * 101
    response = requests.post(f"{BASE_URL}/predict/batch", json=messages)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 400  # Bad request

def run_all_tests():
    """Run all tests"""
    print("="*70)
    print("🧪 TESTING SMS SPAM DETECTION API")
    print("="*70)

    try:
        test_root()
        test_health()
        test_models_info()
        test_single_prediction_spam()
        test_single_prediction_ham()
        test_batch_prediction()
        test_invalid_request()
        test_batch_limit()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the API is running: python api.py")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")

if __name__ == "__main__":
    run_all_tests()
