#!/usr/bin/env python3
"""
Test script for API server functionality
"""

import sys
import time
import threading
import requests
from api_server import StockAnalyzerAPI

def start_test_server():
    """Start the API server in a separate thread for testing."""
    api = StockAnalyzerAPI()
    # Start server on a different port for testing
    api.run(host='127.0.0.1', port=5001, debug=False)

def test_api_endpoints():
    """Test the API endpoints."""
    base_url = "http://127.0.0.1:5001"

    try:
        print("Testing API Server Functionality")
        print("=" * 50)

        # Wait for server to start
        print("Waiting for server to start...")
        time.sleep(2)

        # Test health check
        print("Testing health check...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print(f"  OK Health check: {response.json()}")
        else:
            print(f"  ERROR Health check failed: {response.status_code}")
            return False

        # Test config endpoint
        print("Testing config endpoint...")
        response = requests.get(f"{base_url}/api/config", timeout=10)
        if response.status_code == 200:
            config_data = response.json()
            print(f"  OK Config: API version {config_data.get('api_version')}")
        else:
            print(f"  ERROR Config endpoint failed: {response.status_code}")
            return False

        # Test status endpoint with valid ticker
        print("Testing status endpoint...")
        response = requests.get(f"{base_url}/api/status/AAPL", timeout=10)
        if response.status_code == 200:
            status_data = response.json()
            print(f"  OK Status: {status_data.get('data_status')}")
        else:
            print(f"  ERROR Status endpoint failed: {response.status_code}")

        # Test status endpoint with invalid ticker
        print("Testing status endpoint with invalid ticker...")
        response = requests.get(f"{base_url}/api/status/INVALID", timeout=10)
        if response.status_code == 400:
            print("  OK Invalid ticker properly rejected")
        else:
            print(f"  WARNING Invalid ticker not properly rejected: {response.status_code}")

        # Test 404 endpoint
        print("Testing 404 handling...")
        response = requests.get(f"{base_url}/nonexistent", timeout=10)
        if response.status_code == 404:
            print("  OK 404 properly handled")
        else:
            print(f"  ERROR 404 not properly handled: {response.status_code}")

        print("\nAPI server tests completed successfully!")
        return True

    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API server")
        return False
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_tests():
    """Run the API tests."""
    try:
        # Start server in background thread
        print("Starting test server...")
        server_thread = threading.Thread(target=start_test_server, daemon=True)
        server_thread.start()

        # Run tests
        success = test_api_endpoints()

        return success

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return False
    except Exception as e:
        print(f"Test setup failed: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)