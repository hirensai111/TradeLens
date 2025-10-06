"""
Test script for TradeLens AI Backend
Tests the Flask API endpoints and integration
"""

import requests
import json
import time
from typing import Dict, Any

def test_ai_backend():
    """Test the AI backend functionality"""
    base_url = "http://localhost:5001"

    print("🧪 Testing TradeLens AI Backend")
    print("=" * 50)

    # Test data
    test_context = {
        'symbol': 'AAPL',
        'price': 220.45,
        'rsi': 65.2,
        'macd': 0.75,
        'dayChange': 2.1,
        'volume': 52000000,
        'market_status': 'open',
        'trend': 'bullish'
    }

    test_messages = [
        "What's the trend?",
        "Should I buy AAPL?",
        "Explain RSI in simple terms",
        "What does the 2.1% move mean?",
        "Risk analysis please"
    ]

    # Test 1: Status endpoint
    print("1. Testing status endpoint...")
    try:
        response = requests.get(f"{base_url}/api/chat/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"   [OK] Status: {status['status']}")
            print(f"   OpenAI Available: {status['openai_available']}")
            print(f"   Context Manager: {status['context_manager_available']}")
        else:
            print(f"   [ERROR] Status check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   [ERROR] Connection error: {e}")
        print("   Make sure the AI backend is running on port 5001")
        return False

    print()

    # Test 2: Chat endpoint with test messages
    print("2. Testing chat endpoint...")
    session = requests.Session()

    for i, message in enumerate(test_messages, 1):
        print(f"   Test {i}: '{message}'")

        try:
            chat_data = {
                'message': message,
                'context': test_context,
                'history': []
            }

            response = session.post(
                f"{base_url}/api/chat",
                json=chat_data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', 'No response')
                print(f"   [OK] AI: {ai_response}")

                # Check response length (should be under 3 sentences)
                sentence_count = len([s for s in ai_response.split('.') if s.strip()])
                if sentence_count <= 3:
                    print(f"   [OK] Response length OK ({sentence_count} sentences)")
                else:
                    print(f"   [WARNING]  Response too long ({sentence_count} sentences)")

            else:
                print(f"   [ERROR] Request failed: {response.status_code}")
                print(f"   Response: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"   [ERROR] Request error: {e}")

        print()
        time.sleep(1)  # Avoid overwhelming the API

    # Test 3: Chat history
    print("3. Testing chat history...")
    try:
        response = session.get(f"{base_url}/api/chat/history")
        if response.status_code == 200:
            history = response.json()
            print(f"   [OK] Retrieved {len(history.get('history', []))} messages")
            print(f"   Message count: {history.get('message_count', 0)}")
        else:
            print(f"   [ERROR] History request failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   [ERROR] History error: {e}")

    print()

    # Test 4: Clear chat
    print("4. Testing clear chat...")
    try:
        response = session.post(f"{base_url}/api/chat/clear")
        if response.status_code == 200:
            print("   [OK] Chat cleared successfully")
        else:
            print(f"   [ERROR] Clear failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   [ERROR] Clear error: {e}")

    print()
    print("🎉 Testing completed!")
    return True

def test_quick_actions():
    """Test quick action responses"""
    base_url = "http://localhost:5001"

    print("[ROCKET] Testing Quick Actions")
    print("=" * 30)

    test_context = {
        'symbol': 'TSLA',
        'price': 850.30,
        'rsi': 72.5,
        'macd': 1.25,
        'dayChange': -1.8,
        'volume': 25000000,
        'market_status': 'open',
        'trend': 'bearish'
    }

    quick_actions = [
        "What's the trend?",
        "Should I buy?",
        "Risk analysis",
        "Compare with sector"
    ]

    session = requests.Session()

    for action in quick_actions:
        print(f"Testing: {action}")

        try:
            chat_data = {
                'message': action,
                'context': test_context,
                'history': []
            }

            response = session.post(
                f"{base_url}/api/chat",
                json=chat_data,
                timeout=15
            )

            if response.status_code == 200:
                result = response.json()
                print(f"[OK] {result.get('response', 'No response')}")
            else:
                print(f"[ERROR] Failed: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Error: {e}")

        print()

if __name__ == "__main__":
    import sys

    print("TradeLens AI Backend Test Suite")
    print("Make sure the AI backend is running first:")
    print("python ai_assistant/ai_backend.py")
    print()

    input("Press Enter when the backend is running...")

    # Run tests
    success = test_ai_backend()

    if success:
        print()
        test_quick_actions()
    else:
        print("Basic tests failed. Please check the backend setup.")
        sys.exit(1)