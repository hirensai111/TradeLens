#!/usr/bin/env python3
"""
Fix encoding issues in telegram_bot.py
"""

import os

print("🔧 Fixing telegram_bot.py encoding...")

# Read the file with different encodings
content = None
original_file = 'telegram_bot.py'
backup_file = 'telegram_bot_backup.py'

# Try different encodings
encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

for encoding in encodings:
    try:
        print(f"Trying {encoding}...")
        with open(original_file, 'r', encoding=encoding) as f:
            content = f.read()
        print(f"✅ Successfully read with {encoding}")
        break
    except Exception as e:
        print(f"❌ Failed with {encoding}: {e}")

if content is None:
    print("❌ Could not read file with any encoding")
    print("Trying binary mode...")
    
    # Read as binary and decode with error handling
    with open(original_file, 'rb') as f:
        binary_content = f.read()
    
    # Try to decode, replacing bad characters
    content = binary_content.decode('utf-8', errors='replace')
    print("✅ Read file in binary mode with error replacement")

# Backup original file
print(f"\n📁 Creating backup: {backup_file}")
try:
    with open(original_file, 'rb') as src:
        with open(backup_file, 'wb') as dst:
            dst.write(src.read())
    print("✅ Backup created")
except Exception as e:
    print(f"❌ Backup failed: {e}")

# Write cleaned content back
print(f"\n📝 Writing cleaned content to {original_file}")
try:
    # Remove any problematic characters
    # Common problematic characters: smart quotes, em dashes, etc.
    replacements = {
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2026': '...', # Ellipsis
        '\u00a0': ' ',  # Non-breaking space
    }
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    # Write with UTF-8 encoding
    with open(original_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ File fixed and saved with UTF-8 encoding")
    
    # Verify we can import it now
    print("\n🔍 Testing import...")
    try:
        import telegram_bot
        print("✅ Import successful!")
    except ImportError as e:
        print(f"❌ Import still failing: {e}")
        print("This might be due to the missing python-telegram-bot package")
    
except Exception as e:
    print(f"❌ Failed to write cleaned file: {e}")

print("\n✅ Done! Now run:")
print("1. pip install python-telegram-bot==13.15")
print("2. python test_tg_bot.py quick")