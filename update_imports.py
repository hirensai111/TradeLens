"""
Import Path Update Helper
Automatically updates import statements to match the new project structure.
"""

import os
import re
from pathlib import Path

# Mapping of old import paths to new import paths
IMPORT_MAPPINGS = {
    # News System
    r'from stock_analyzer\.news_api_client': 'from news_system.collectors.news_api_client',
    r'from stock_analyzer\.sentiment_analyzer': 'from news_system.analyzers.sentiment_analyzer',
    r'from stock_analyzer\.sentiment_integration': 'from news_system.analyzers.sentiment_integration',
    r'from stock_analyzer\.event_analyzer': 'from news_system.analyzers.event_analyzer',
    r'from stock_analyzer\.api_adapters': 'from news_system.collectors.adapters',
    r'from src\.collectors': 'from news_system.collectors',
    r'from src\.processors': 'from news_system.processors',
    r'from src\.intelligence': 'from news_system.analyzers',
    r'from src\.database': 'from news_system.database',

    # Prediction Engine
    r'from stock_analyzer\.data_processor': 'from prediction_engine.technical_analysis.data_processor',
    r'from stock_analyzer\.alpha_vantage_client': 'from prediction_engine.data_loaders.alpha_vantage_client',
    r'from phase4_ml_engine\.prediction_engine': 'from prediction_engine.predictors.prediction_engine',
    r'from phase4_ml_engine\.src\.features': 'from prediction_engine.features',
    r'from phase4_ml_engine\.src\.data_loaders': 'from prediction_engine.data_loaders',

    # Options Analyzer
    r'from phase4_ml_engine\.options_analyzer': 'from options_analyzer.analyzers.options_analyzer',
    r'from phase4_ml_engine\.automated_options_bot': 'from options_analyzer.bots.automated_options_bot',
    r'from phase4_ml_engine\.indian_trading_bot': 'from options_analyzer.bots.indian_trading_bot',
    r'from phase4_ml_engine\.zerodha_api_client': 'from options_analyzer.brokers.zerodha_api_client',
    r'from phase4_ml_engine\.groww_api_client': 'from options_analyzer.brokers.groww_api_client',
    r'from phase4_ml_engine\.ind_': 'from options_analyzer.indian_market.ind_',

    # Core
    r'from stock_analyzer\.utils': 'from core.utils.utils',
    r'from stock_analyzer\.config': 'from core.config.config',
    r'from stock_analyzer\.validators': 'from core.validators.validators',
    r'from src\.utils\.config': 'from core.config',

    # API
    r'from stock_analyzer\.api_server': 'from api.rest.api_server',
    r'from phase4_ml_engine\.telegram_bot': 'from api.telegram.telegram_bot',

    # Output
    r'from stock_analyzer\.excel_generator': 'from output.excel.excel_generator',
    r'from stock_analyzer\.json_exporter': 'from output.json.json_exporter',

    # AI Assistant
    r'from stock_analyzer\.ai_assistant': 'from ai_assistant',
}

def update_imports_in_file(file_path):
    """Update import statements in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        updated = False

        for old_pattern, new_import in IMPORT_MAPPINGS.items():
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_import, content)
                updated = True

        if updated and content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[OK] Updated: {file_path}")
            return True

        return False

    except Exception as e:
        print(f"[ERROR] Error updating {file_path}: {e}")
        return False

def update_all_imports(root_dir='.'):
    """Update imports in all Python files in the new structure."""
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

    updated_count = 0
    total_files = 0

    for directory in directories:
        dir_path = Path(root_dir) / directory
        if not dir_path.exists():
            continue

        for py_file in dir_path.rglob('*.py'):
            total_files += 1
            if update_imports_in_file(py_file):
                updated_count += 1

    # Also update main.py
    main_file = Path(root_dir) / 'main.py'
    if main_file.exists():
        total_files += 1
        if update_imports_in_file(main_file):
            updated_count += 1

    print(f"\n{'='*60}")
    print(f"Import Update Summary:")
    print(f"  Total files processed: {total_files}")
    print(f"  Files updated: {updated_count}")
    print(f"  Files unchanged: {total_files - updated_count}")
    print(f"{'='*60}")

def verify_structure():
    """Verify that the new directory structure exists."""
    required_dirs = [
        'news_system/collectors',
        'news_system/processors',
        'news_system/analyzers',
        'news_system/database',
        'prediction_engine/models',
        'prediction_engine/features',
        'prediction_engine/data_loaders',
        'prediction_engine/technical_analysis',
        'prediction_engine/predictors',
        'options_analyzer/analyzers',
        'options_analyzer/bots',
        'options_analyzer/brokers',
        'options_analyzer/indian_market',
        'core/config',
        'core/utils',
        'core/validators',
        'api/rest',
        'api/telegram',
        'output/excel',
        'output/json',
        'tests/news_system',
        'tests/prediction_engine',
        'tests/options_analyzer',
        'tests/integration',
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)

    if missing_dirs:
        print("Warning: Missing directories:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        return False

    print("[OK] All required directories exist")
    return True

if __name__ == '__main__':
    print("TradeLens Import Path Updater")
    print("=" * 60)

    # Verify structure
    if not verify_structure():
        print("\nPlease create the directory structure first!")
        exit(1)

    print("\nUpdating import paths...")
    update_all_imports()

    print("\n[OK] Import update complete!")
    print("\nNext steps:")
    print("1. Review the updated files")
    print("2. Run tests: python -m pytest tests/")
    print("3. Fix any remaining import issues manually")
    print("4. Delete old directories: stock_analyzer/ and stock-prediction-engine/")
