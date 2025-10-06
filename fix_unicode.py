"""
Fix all Unicode/Emoji characters that cause encoding errors on Windows
"""

import re
from pathlib import Path

# Emoji/Unicode replacements
UNICODE_REPLACEMENTS = {
    '🚀': '[ROCKET]',
    '✓': '[OK]',
    '✅': '[OK]',
    '❌': '[ERROR]',
    '⚠️': '[WARNING]',
    '⚠': '[WARNING]',
    '📊': '[CHART]',
    '🔔': '[BELL]',
    '💰': '[MONEY]',
    '📈': '[UP]',
    '📉': '[DOWN]',
    '🎯': '[TARGET]',
    '💎': '[GEM]',
    '⭐': '[STAR]',
    '❓': '[?]',
    '📡': '[SIGNAL]',
    '🇮🇳': '[IN]',
    'ℹ️': '[INFO]',
    'ℹ': '[INFO]',
    '💡': '[BULB]',
    '\U0001f4a1': '[BULB]',  # Lightbulb emoji
    '\U0001f680': '[ROCKET]',  # Rocket emoji
    '\U0001f527': '[WRENCH]',  # Wrench emoji
    '🔧': '[WRENCH]',
    '🔑': '[KEY]',
    '\U0001f511': '[KEY]',  # Key emoji
}

def fix_unicode_in_file(file_path: Path) -> bool:
    """Fix unicode characters in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        for emoji, replacement in UNICODE_REPLACEMENTS.items():
            content = content.replace(emoji, replacement)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[OK] Fixed: {file_path}")
            return True

        return False
    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        return False

def fix_all_unicode():
    """Fix unicode in all Python files."""

    directories = [
        'news_system',
        'prediction_engine',
        'options_analyzer',
        'core',
        'api',
        'output',
        'ai_assistant',
        'tests'
    ]

    fixed_count = 0
    total_count = 0

    print("Fixing Unicode/Emoji characters...")
    print("=" * 60)

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue

        for py_file in dir_path.rglob('*.py'):
            if '__pycache__' in str(py_file):
                continue

            total_count += 1
            if fix_unicode_in_file(py_file):
                fixed_count += 1

    # Also fix main.py
    if Path('main.py').exists():
        total_count += 1
        if fix_unicode_in_file(Path('main.py')):
            fixed_count += 1

    print("=" * 60)
    print(f"\nTotal files scanned: {total_count}")
    print(f"Files fixed: {fixed_count}")
    print(f"Files unchanged: {total_count - fixed_count}")
    print("\n[OK] Unicode fix complete!")

if __name__ == '__main__':
    fix_all_unicode()
