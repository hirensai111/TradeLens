"""
Comprehensive Import Fixer for TradeLens Project
Fixes all import statements to match the new reorganized structure.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

# Comprehensive mapping of old import paths to new import paths
IMPORT_MAPPINGS = {
    # ============ NEWS SYSTEM ============

    # News collectors
    r'from src\.collectors\.newsapi_collector': 'from news_system.collectors.newsapi_collector',
    r'from src\.collectors\.reddit_collector': 'from news_system.collectors.reddit_collector',
    r'from src\.collectors\.hybrid_collector': 'from news_system.collectors.hybrid_collector',
    r'from src\.collectors\.base_collector': 'from news_system.collectors.base_collector',
    r'from src\.collectors\.scheduler': 'from news_system.collectors.scheduler',
    r'from src\.collectors': 'from news_system.collectors',

    r'from stock_analyzer\.news_api_client': 'from news_system.collectors.news_api_client',
    r'from stock_analyzer\.news_integration_bridge': 'from news_system.collectors.news_integration_bridge',
    r'from stock_analyzer\.api_config': 'from news_system.collectors.api_config',
    r'from stock_analyzer\.api_adapters': 'from news_system.collectors.adapters',

    # News processors
    r'from src\.processors\.sentiment_analyzer': 'from news_system.processors.sentiment_analyzer',
    r'from src\.processors\.text_processor': 'from news_system.processors.text_processor',
    r'from src\.processors\.deduplicator': 'from news_system.processors.deduplicator',
    r'from src\.processors\.event_extractor': 'from news_system.processors.event_extractor',
    r'from src\.processors': 'from news_system.processors',

    # News analyzers
    r'from stock_analyzer\.sentiment_analyzer': 'from news_system.analyzers.sentiment_analyzer',
    r'from stock_analyzer\.sentiment_integration': 'from news_system.analyzers.sentiment_integration',
    r'from stock_analyzer\.event_analyzer': 'from news_system.analyzers.event_analyzer',
    r'from src\.intelligence\.correlation_analyzer': 'from news_system.analyzers.correlation_analyzer',
    r'from src\.intelligence': 'from news_system.analyzers',

    # News database
    r'from src\.database\.models\.news_models': 'from news_system.database.models.news_models',
    r'from src\.database\.connection': 'from news_system.database.connection',
    r'from src\.database\.session': 'from news_system.database.session',
    r'from src\.database\.migrations': 'from news_system.database.migrations',
    r'from src\.database': 'from news_system.database',

    # ============ PREDICTION ENGINE ============

    # Technical analysis
    r'from stock_analyzer\.data_processor': 'from prediction_engine.technical_analysis.data_processor',

    # Data loaders
    r'from stock_analyzer\.alpha_vantage_client': 'from prediction_engine.data_loaders.alpha_vantage_client',
    r'from alpha_vantage_config': 'from prediction_engine.data_loaders.alpha_vantage_config',
    r'from phase4_ml_engine\.src\.data_loaders\.excel_loader': 'from prediction_engine.data_loaders.excel_loader',
    r'from phase4_ml_engine\.src\.data_loaders\.phase3_connector': 'from prediction_engine.data_loaders.phase3_connector',
    r'from phase4_ml_engine\.src\.data_loaders': 'from prediction_engine.data_loaders',
    r'from src\.data_loaders': 'from prediction_engine.data_loaders',

    # Features
    r'from phase4_ml_engine\.src\.features\.feature_engineering': 'from prediction_engine.features.feature_engineering',
    r'from phase4_ml_engine\.src\.features': 'from prediction_engine.features',
    r'from src\.features': 'from prediction_engine.features',

    # Predictors
    r'from phase4_ml_engine\.prediction_engine': 'from prediction_engine.predictors.prediction_engine',
    r'from phase4_ml_engine\.predict': 'from prediction_engine.predictors.predict',
    r'from phase4_ml_engine\.ultimate_ai_predictor': 'from prediction_engine.predictors.ultimate_ai_predictor',
    r'from phase4_ml_engine\.graph_analyzer': 'from prediction_engine.predictors.graph_analyzer',
    r'from phase4_ml_engine\.report_generator': 'from prediction_engine.predictors.report_generator',

    # Models
    r'from phase4_ml_engine\.src\.models': 'from prediction_engine.models',
    r'from src\.models': 'from prediction_engine.models',

    # ============ OPTIONS ANALYZER ============

    # Analyzers
    r'from phase4_ml_engine\.options_analyzer': 'from options_analyzer.analyzers.options_analyzer',
    r'from phase4_ml_engine\.check_nifty_options': 'from options_analyzer.analyzers.check_nifty_options',

    # Bots
    r'from phase4_ml_engine\.automated_options_bot': 'from options_analyzer.bots.automated_options_bot',
    r'from phase4_ml_engine\.indian_trading_bot': 'from options_analyzer.bots.indian_trading_bot',

    # Brokers
    r'from phase4_ml_engine\.zerodha_api_client': 'from options_analyzer.brokers.zerodha_api_client',
    r'from phase4_ml_engine\.zerodha_technical_analyzer': 'from options_analyzer.brokers.zerodha_technical_analyzer',
    r'from phase4_ml_engine\.groww_api_client': 'from options_analyzer.brokers.groww_api_client',
    r'from phase4_ml_engine\.get_access_token': 'from options_analyzer.brokers.get_access_token',

    # Indian market
    r'from phase4_ml_engine\.ind_trade_logger': 'from options_analyzer.indian_market.ind_trade_logger',
    r'from phase4_ml_engine\.ind_data_processor': 'from options_analyzer.indian_market.ind_data_processor',

    # ============ CORE ============

    # Config
    r'from stock_analyzer\.config import Config': 'from core.config.config import Config',
    r'from stock_analyzer\.config': 'from core.config.config',
    r'from src\.utils\.config\.settings': 'from core.config.settings',
    r'from src\.utils\.config': 'from core.config',

    # Utils
    r'from stock_analyzer\.utils': 'from core.utils.utils',

    # Validators
    r'from stock_analyzer\.validators': 'from core.validators.validators',

    # ============ API ============

    # REST API
    r'from stock_analyzer\.api_server': 'from api.rest.api_server',
    r'from stock_analyzer\.stock_analyzer_handler': 'from api.rest.stock_analyzer_handler',

    # Telegram
    r'from phase4_ml_engine\.telegram_bot': 'from api.telegram.telegram_bot',

    # ============ OUTPUT ============

    # Excel
    r'from stock_analyzer\.excel_generator': 'from output.excel.excel_generator',

    # JSON
    r'from stock_analyzer\.json_exporter': 'from output.json.json_exporter',

    # Prompts
    r'from stock_analyzer\.corresponding_prompts': 'from output.corresponding_prompts',

    # ============ AI ASSISTANT ============

    r'from stock_analyzer\.ai_assistant\.ai_backend': 'from ai_assistant.ai_backend',
    r'from stock_analyzer\.ai_assistant\.context_manager': 'from ai_assistant.context_manager',
    r'from stock_analyzer\.ai_assistant\.prompt_templates': 'from ai_assistant.prompt_templates',
    r'from stock_analyzer\.ai_assistant': 'from ai_assistant',

    # ============ MONITORING ============

    r'from src\.monitoring\.alert_system': 'from news_system.monitoring.alert_system',
    r'from src\.monitoring': 'from news_system.monitoring',
}

# Additional patterns for import statements
IMPORT_AS_PATTERNS = [
    (r'import stock_analyzer\.config as', 'import core.config.config as'),
    (r'import stock_analyzer\.utils as', 'import core.utils.utils as'),
    (r'import stock_analyzer\.data_processor as', 'import prediction_engine.technical_analysis.data_processor as'),
]

def update_imports_in_file(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Update import statements in a single file.
    Returns (was_updated, list_of_changes)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return False, [f"Error reading file: {e}"]

    original_content = content
    changes = []

    # Apply all import mappings
    for old_pattern, new_import in IMPORT_MAPPINGS.items():
        matches = re.findall(old_pattern, content)
        if matches:
            content = re.sub(old_pattern, new_import, content)
            changes.append(f"  {old_pattern} -> {new_import}")

    # Apply import as patterns
    for old_pattern, new_pattern in IMPORT_AS_PATTERNS:
        if re.search(old_pattern, content):
            content = re.sub(old_pattern, new_pattern, content)
            changes.append(f"  {old_pattern} -> {new_pattern}")

    # Write back if changed
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes
        except Exception as e:
            return False, [f"Error writing file: {e}"]

    return False, []

def fix_all_imports(root_dir: str = '.') -> dict:
    """Fix imports in all Python files in the new structure."""

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

    results = {
        'total_files': 0,
        'updated_files': 0,
        'unchanged_files': 0,
        'errors': 0,
        'details': {}
    }

    print("=" * 80)
    print("FIXING ALL IMPORTS IN REORGANIZED PROJECT")
    print("=" * 80)
    print()

    for directory in directories:
        dir_path = Path(root_dir) / directory
        if not dir_path.exists():
            print(f"[SKIP] Directory not found: {directory}")
            continue

        print(f"\n[PROCESSING] {directory}/")
        print("-" * 80)

        module_stats = {'files': 0, 'updated': 0}

        for py_file in dir_path.rglob('*.py'):
            # Skip __pycache__ and other generated files
            if '__pycache__' in str(py_file):
                continue

            results['total_files'] += 1
            module_stats['files'] += 1

            was_updated, changes = update_imports_in_file(py_file)

            if was_updated:
                results['updated_files'] += 1
                module_stats['updated'] += 1
                rel_path = py_file.relative_to(root_dir)
                print(f"  [OK] {rel_path}")
                if changes:
                    for change in changes[:3]:  # Show first 3 changes
                        print(f"    {change}")
                    if len(changes) > 3:
                        print(f"    ... and {len(changes) - 3} more changes")
            elif changes and not was_updated:
                results['errors'] += 1
                rel_path = py_file.relative_to(root_dir)
                print(f"  [ERROR] {rel_path}")
                for change in changes:
                    print(f"    {change}")

        results['details'][directory] = module_stats
        if module_stats['updated'] > 0:
            print(f"\n  Summary: {module_stats['updated']}/{module_stats['files']} files updated")

    # Also update main.py
    main_file = Path(root_dir) / 'main.py'
    if main_file.exists():
        results['total_files'] += 1
        was_updated, changes = update_imports_in_file(main_file)
        if was_updated:
            results['updated_files'] += 1
            print(f"\n[OK] main.py")
            for change in changes[:5]:
                print(f"  {change}")

    results['unchanged_files'] = results['total_files'] - results['updated_files'] - results['errors']

    return results

def print_summary(results: dict):
    """Print summary of import fixes."""
    print("\n" + "=" * 80)
    print("IMPORT FIX SUMMARY")
    print("=" * 80)
    print(f"\nTotal files processed:  {results['total_files']}")
    print(f"Files updated:          {results['updated_files']}")
    print(f"Files unchanged:        {results['unchanged_files']}")
    print(f"Errors:                 {results['errors']}")

    if results['details']:
        print("\nPer-module breakdown:")
        print("-" * 80)
        for module, stats in results['details'].items():
            if stats['updated'] > 0:
                print(f"  {module:25} {stats['updated']:3}/{stats['files']:3} files updated")

    print("\n" + "=" * 80)

    if results['updated_files'] > 0:
        print("\n[OK] Import fixing complete!")
        print("\nNext steps:")
        print("1. Test the imports: python -c 'import news_system, prediction_engine, options_analyzer'")
        print("2. Run tests: python -m pytest tests/ -v")
        print("3. Fix any remaining import issues manually")
    else:
        print("\nNo imports needed updating (or all were already updated)")

if __name__ == '__main__':
    results = fix_all_imports()
    print_summary(results)
