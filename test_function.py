#!/usr/bin/env python3
"""Test the Netlify function locally"""
import sys
from pathlib import Path

# Add path for imports
sys.path.append(str(Path(__file__).resolve().parent / "Qna chatbot"))

from core.single_product import answer

# Test the core function
def test_core():
    print("Testing core function...")
    result = answer("Is it transfer-proof?")
    print(f"Result: {result}")

# Test the Netlify handler
def test_netlify_handler():
    print("\nTesting Netlify handler...")
    sys.path.append("netlify/functions/chat")
    from main import handler
    
    # Mock event
    event = {
        "httpMethod": "POST",
        "body": '{"question": "What finish does it have?"}'
    }
    
    result = handler(event, {})
    print(f"Status: {result['statusCode']}")
    print(f"Response: {result['body']}")

if __name__ == "__main__":
    test_core()
    test_netlify_handler()
