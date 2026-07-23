"""
Test script for FastAPI backend

Run this script to test your API endpoints locally.
Usage: python test_api.py
"""

import requests
import os
import sys
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "test_image.jpg"  # Replace with your test image


def test_health():
    """Test health endpoint"""
    print("\n1. Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Status: {data['status']}")
            print(f"   ✓ Model loaded: {data['model_loaded']}")
            return True
        else:
            print(f"   ✗ Error: Status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ✗ Cannot connect to {API_URL}")
        print("   Make sure the backend is running!")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_root():
    """Test root endpoint"""
    print("\n2. Testing / endpoint...")
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Message: {data['message']}")
            print(f"   ✓ Status: {data['status']}")
            return True
        else:
            print(f"   ✗ Error: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_model_info():
    """Test model info endpoint"""
    print("\n3. Testing /model/info endpoint...")
    try:
        response = requests.get(f"{API_URL}/model/info")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Model: {data['model_name']}")
            print(f"   ✓ Framework: {data['framework']}")
            print(f"   ✓ Loaded: {data['loaded']}")
            return True
        else:
            print(f"   ✗ Error: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_prediction(image_path):
    """Test prediction endpoint"""
    print("\n4. Testing /predict endpoint...")

    # Check if test image exists
    if not os.path.exists(image_path):
        print(f"   ✗ Test image not found: {image_path}")
        print(f"   Please provide a test image or update TEST_IMAGE_PATH")
        return False

    try:
        # Open and send image
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                timeout=30
            )

        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Success: {data['success']}")
            print(f"   ✓ Filename: {data['filename']}")
            print(f"\n   Top Predictions:")
            for i, pred in enumerate(data['predictions'][:3], 1):
                confidence = pred['confidence'] * 100
                print(f"   {i}. {pred['class']}: {confidence:.2f}%")
            return True
        else:
            print(f"   ✗ Error: Status code {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print(f"   ✗ Request timeout (>30s)")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("FastAPI Backend Test Suite")
    print("=" * 50)
    print(f"Testing API at: {API_URL}")

    # Run tests
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Root Endpoint", test_root()))
    results.append(("Model Info", test_model_info()))
    results.append(("Prediction", test_prediction(TEST_IMAGE_PATH)))

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{len(results)} tests passed")
    print("=" * 50)

    # Exit code
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
