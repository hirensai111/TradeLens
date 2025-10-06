#!/usr/bin/env python3
"""
Debug script to check import issues
"""

import os
import sys

print("🔍 Debugging Import Issues")
print("-" * 50)

# Check current directory
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path[0]}")

# List all Python files in current directory
print("\n📁 Python files in current directory:")
for file in os.listdir('.'):
    if file.endswith('.py'):
        print(f"  - {file}")

# Check specific file
print("\n🔎 Checking for telegram_bot.py:")
if os.path.exists('telegram_bot.py'):
    print("✅ telegram_bot.py exists")
    
    # Check if it's readable
    try:
        with open('telegram_bot.py', 'r') as f:
            first_line = f.readline()
            print(f"✅ File is readable")
            print(f"First line: {first_line[:50]}...")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
else:
    print("❌ telegram_bot.py NOT found")

# Try different import methods
print("\n🔧 Testing imports:")

# Method 1: Direct import
try:
    import telegram_bot
    print("✅ Method 1: 'import telegram_bot' worked")
except ImportError as e:
    print(f"❌ Method 1 failed: {e}")

# Method 2: From import
try:
    from telegram_bot import TelegramSignalBot
    print("✅ Method 2: 'from telegram_bot import TelegramSignalBot' worked")
except ImportError as e:
    print(f"❌ Method 2 failed: {e}")

# Method 3: Check if it's a package issue
print("\n📦 Checking for __init__.py:")
if os.path.exists('__init__.py'):
    print("Found __init__.py - this directory is a package")
else:
    print("No __init__.py - this is a regular directory")

# Check file encoding
print("\n🔤 Checking file details:")
if os.path.exists('telegram_bot.py'):
    stat = os.stat('telegram_bot.py')
    print(f"File size: {stat.st_size} bytes")
    
    # Try to detect any hidden characters in filename
    files = os.listdir('.')
    for f in files:
        if 'telegram_bot' in f:
            print(f"Found file: '{f}' (length: {len(f)})")
            print(f"Bytes: {f.encode()}")

print("\n💡 Suggestions:")
print("1. Make sure there are no hidden characters in the filename")
print("2. Try renaming the file: rename telegram_bot.py telegram_bot_new.py")
print("3. Then rename it back: rename telegram_bot_new.py telegram_bot.py")
print("4. Check if the file has the correct Python code")