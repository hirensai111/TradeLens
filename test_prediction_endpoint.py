#!/usr/bin/env python3
"""
Test script to verify the prediction endpoint works
"""
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stock_analyzer'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'prediction_engine'))

from api_server import StockAnalyzerAPI
from flask import Flask

def test_prediction_endpoint():
    """Test if prediction endpoint is registered"""
    print("Creating API instance...")
    api = StockAnalyzerAPI()

    print("\nRegistered routes:")
    for rule in api.app.url_map.iter_rules():
        print(f"  {rule.methods} {rule.rule}")

    # Check if predict route exists
    predict_routes = [r for r in api.app.url_map.iter_rules() if 'predict' in str(r.rule)]

    if predict_routes:
        print(f"\n✓ Prediction endpoint found: {predict_routes[0]}")
        print("\nTo test the endpoint:")
        print("1. Make sure the API server is running: python stock_analyzer/api_server.py")
        print("2. In another terminal, run:")
        print("   curl http://localhost:5000/api/predict/AAPL")
        return True
    else:
        print("\n✗ Prediction endpoint NOT found!")
        return False

if __name__ == "__main__":
    test_prediction_endpoint()
