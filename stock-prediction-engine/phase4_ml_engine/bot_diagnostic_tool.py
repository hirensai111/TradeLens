#!/usr/bin/env python3
"""
Trading Bot Diagnostic Tool
Identifies and fixes integration issues in the trading bot
"""

import os
import sys
import traceback
import importlib.util
from pathlib import Path

class TradingBotDiagnostic:
    """Comprehensive diagnostic tool for trading bot issues"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.issues = []
        self.fixes = []
        
    def run_full_diagnostic(self):
        """Run complete diagnostic"""
        print("🔍 TRADING BOT DIAGNOSTIC TOOL")
        print("=" * 50)
        
        self.check_file_structure()
        self.check_imports()
        self.check_environment_variables()
        self.analyze_import_errors()
        self.suggest_fixes()
        
        print("\n" + "=" * 50)
        print("📋 DIAGNOSTIC SUMMARY")
        print("=" * 50)
        print(f"Issues Found: {len(self.issues)}")
        print(f"Fixes Available: {len(self.fixes)}")
        
        if self.issues:
            print("\n❌ ISSUES FOUND:")
            for i, issue in enumerate(self.issues, 1):
                print(f"{i}. {issue}")
        
        if self.fixes:
            print("\n🔧 SUGGESTED FIXES:")
            for i, fix in enumerate(self.fixes, 1):
                print(f"{i}. {fix}")
    
    def check_file_structure(self):
        """Check if all required files exist"""
        print("\n📁 Checking File Structure...")
        
        required_files = {
            'indian_trading_bot.py': 'Main trading bot file',
            'options_analyzer.py': 'Enhanced options analyzer', 
            'zerodha_technical_analyzer.py': 'Technical analysis engine',
            'zerodha_api_client.py': 'Zerodha API client',
            'telegram_bot.py': 'Telegram integration',
            'ind_trade_logger.py': 'Trade logging',
            'ind_data_processor.py': 'Market data processor'
        }
        
        missing_files = []
        existing_files = []
        
        for filename, description in required_files.items():
            filepath = self.project_root / filename
            if filepath.exists():
                print(f"✅ {filename} - {description}")
                existing_files.append(filename)
            else:
                print(f"❌ {filename} - {description} (MISSING)")
                missing_files.append(filename)
                self.issues.append(f"Missing file: {filename}")
        
        # Check for alternative file names
        if missing_files:
            print("\n🔍 Checking for alternative file names...")
            all_py_files = list(self.project_root.glob("*.py"))
            
            for py_file in all_py_files:
                if py_file.name not in existing_files:
                    print(f"   📄 Found: {py_file.name}")
            
            # Suggest file creation
            for missing in missing_files:
                if missing == 'indian_trading_bot.py':
                    self.fixes.append(f"indian_trading_bot.py not found - rename from trading_bot.py or check file name")
                elif missing == 'options_analyzer.py':
                    self.fixes.append(f"Create {missing} or rename existing enhanced analyzer file")
                elif missing == 'zerodha_technical_analyzer.py':
                    self.fixes.append(f"Create {missing} or rename existing technical analyzer file")
                else:
                    self.fixes.append(f"Create missing file: {missing}")
    
    def check_imports(self):
        """Check import statements in indian_trading_bot.py"""
        print("\n📦 Checking Import Statements...")
        
        # Check both possible bot file names
        bot_files = ['indian_trading_bot.py', 'trading_bot.py']
        bot_file = None
        
        for filename in bot_files:
            filepath = self.project_root / filename
            if filepath.exists():
                bot_file = filepath
                print(f"✅ Found bot file: {filename}")
                break
        
        if not bot_file:
            self.issues.append("No trading bot file found (indian_trading_bot.py or trading_bot.py)")
            return
        
        try:
            with open(bot_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for import statements
            import_lines = [line.strip() for line in content.split('\n') if line.strip().startswith('from ') or line.strip().startswith('import ')]
            
            problematic_imports = []
            
            for line in import_lines:
                if 'options_analyzer' in line:
                    print(f"📦 Found: {line}")
                    # Try to import
                    try:
                        exec(line)
                        print("   ✅ Import successful")
                    except ImportError as e:
                        print(f"   ❌ Import failed: {e}")
                        problematic_imports.append(line)
                        self.issues.append(f"Import error: {line}")
                
                elif any(module in line for module in ['zerodha_technical_analyzer', 'zerodha_api_client', 'telegram_bot', 'ind_trade_logger', 'ind_data_processor']):
                    print(f"📦 Found: {line}")
                    try:
                        exec(line)
                        print("   ✅ Import successful")
                    except ImportError as e:
                        print(f"   ❌ Import failed: {e}")
                        problematic_imports.append(line)
                        self.issues.append(f"Import error: {line}")
            
            if problematic_imports:
                self.fixes.append("Fix import statements in the trading bot file")
                self.fixes.append("Ensure all imported modules exist and are properly named")
        
        except Exception as e:
            self.issues.append(f"Error reading bot file: {e}")
    
    def check_environment_variables(self):
        """Check required environment variables"""
        print("\n🌍 Checking Environment Variables...")
        
        required_vars = {
            'ZERODHA_API_KEY': 'Zerodha API key',
            'ZERODHA_ACCESS_TOKEN': 'Zerodha access token (or API secret)',
            'TELEGRAM_BOT_TOKEN': 'Telegram bot token',
            'TELEGRAM_CHAT_ID': 'Telegram chat ID'
        }
        
        missing_vars = []
        
        for var, description in required_vars.items():
            value = os.getenv(var)
            if value:
                print(f"✅ {var} - {description} (Set)")
            else:
                print(f"❌ {var} - {description} (Missing)")
                missing_vars.append(var)
                self.issues.append(f"Missing environment variable: {var}")
        
        if missing_vars:
            self.fixes.append("Create .env file with required environment variables")
            self.fixes.append("Set missing environment variables")
    
    def analyze_import_errors(self):
        """Analyze specific import errors"""
        print("\n🔬 Analyzing Import Errors...")
        
        # Test each component individually
        components = {
            'options_analyzer': 'ZerodhaEnhancedOptionsAnalyzer',
            'zerodha_technical_analyzer': 'ZerodhaTechnicalAnalyzer', 
            'zerodha_api_client': 'ZerodhaAPIClient',
            'telegram_bot': 'TelegramSignalBot',
            'ind_trade_logger': 'IndianTradeLogger',
            'ind_data_processor': 'IndianMarketProcessor'
        }
        
        for module_name, class_name in components.items():
            try:
                print(f"Testing {module_name}...")
                
                # Try to import the module
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    print(f"   ❌ Module {module_name} not found")
                    self.issues.append(f"Module not found: {module_name}")
                    continue
                
                # Try to load the module
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check if class exists
                if hasattr(module, class_name):
                    print(f"   ✅ {class_name} found in {module_name}")
                else:
                    print(f"   ❌ {class_name} not found in {module_name}")
                    self.issues.append(f"Class {class_name} not found in {module_name}")
                
            except Exception as e:
                print(f"   ❌ Error importing {module_name}: {e}")
                self.issues.append(f"Import error in {module_name}: {str(e)}")
    
    def suggest_fixes(self):
        """Suggest specific fixes for identified issues"""
        print("\n🔧 Generating Specific Fixes...")
        
        # File-specific fixes
        if any('options_analyzer' in issue for issue in self.issues):
            self.fixes.append("Create options_analyzer.py with ZerodhaEnhancedOptionsAnalyzer class")
            self.fixes.append("Or rename your enhanced options analyzer file to options_analyzer.py")
        
        if any('zerodha_technical_analyzer' in issue for issue in self.issues):
            self.fixes.append("Create zerodha_technical_analyzer.py with ZerodhaTechnicalAnalyzer class")
            self.fixes.append("Or rename your technical analyzer file to zerodha_technical_analyzer.py")
        
        # Method-specific fixes
        if any('analyze_symbol_for_options_enhanced' in issue for issue in self.issues):
            self.fixes.append("Rename method to analyze_symbol_for_options_enhanced or update bot calls")
        
        # Environment fixes
        if any('environment variable' in issue for issue in self.issues):
            self.fixes.append("Create .env file with required variables")
    
    def create_minimal_missing_files(self):
        """Create minimal versions of missing files for testing"""
        print("\n🏗️ Creating Minimal Missing Files...")
        
        templates = {
            'zerodha_api_client.py': '''"""
Minimal Zerodha API Client for testing
"""

class ZerodhaAPIClient:
    def __init__(self, api_key=None, access_token=None):
        self.api_key = api_key
        self.access_token = access_token
    
    def get_market_status(self):
        return {'status': 'open'}
    
    def get_live_quotes(self, symbols):
        return {symbol: {'price': 100.0} for symbol in symbols}
    
    def get_historical_data(self, symbol, timeframe, days):
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        return pd.DataFrame({
            'open': np.random.uniform(95, 105, days),
            'high': np.random.uniform(100, 110, days),
            'low': np.random.uniform(90, 100, days),
            'close': np.random.uniform(95, 105, days),
            'volume': np.random.randint(10000, 100000, days)
        }, index=dates)
''',
            
            'telegram_bot.py': '''"""
Minimal Telegram Bot for testing
"""

class TelegramSignalBot:
    def __init__(self, bot_token=None, chat_id=None):
        self.bot_token = bot_token
        self.chat_id = chat_id
    
    def send_message(self, message, silent=False):
        print(f"[TELEGRAM] {message}")
        return True
    
    def send_trade_signal(self, signal_message):
        print(f"[SIGNAL] {signal_message}")
        return True
    
    def send_error_alert(self, error_message):
        print(f"[ERROR] {error_message}")
        return True
''',
            
            'ind_trade_logger.py': '''"""
Minimal Trade Logger for testing
"""

class IndianTradeLogger:
    def __init__(self):
        self.signals = []
    
    def log_signal(self, ticker, analysis_result, source='test'):
        signal_id = len(self.signals) + 1
        self.signals.append({
            'id': signal_id,
            'ticker': ticker,
            'analysis_result': analysis_result,
            'source': source
        })
        return signal_id
    
    def save_automation_signal(self, signal_data):
        return True
    
    def get_daily_summary(self):
        return {'total_signals': len(self.signals)}
    
    def get_automation_stats(self):
        return {'total_signals': 0, 'processed_signals': 0}
    
    def test_automation_integration(self):
        return {'status': 'success'}
''',
            
            'ind_data_processor.py': '''"""
Minimal Data Processor for testing
"""

class IndianMarketProcessor:
    def __init__(self):
        pass
    
    def process_market_data(self, data):
        return data
''',
            
            'options_analyzer.py': '''"""
Minimal Options Analyzer for testing
This should be replaced with your actual enhanced analyzer
"""

class ZerodhaEnhancedOptionsAnalyzer:
    def __init__(self, zerodha_client=None, claude_api_key=None):
        self.zerodha = zerodha_client
        self.claude_api_key = claude_api_key
    
    async def analyze_trade(self, symbol, trading_style='swing', 
                          prediction_days=14, risk_tolerance='medium',
                          capital=100000, execute_trades=False):
        return {
            'trade_recommendation': {
                'strategy': 'BULLISH_CALL',
                'confidence': 0.7,
                'option_legs': []
            },
            'technical_analysis': {
                'market_bias': 'BULLISH',
                'confidence_score': 0.7
            },
            'market_data': {
                'current_price': 100.0
            },
            'zerodha_integration': {
                'live_data_available': True
            }
        }
    
    async def monitor_positions(self):
        return {'positions': {'net_positions': []}}
''',
            
            'zerodha_technical_analyzer.py': '''"""
Minimal Technical Analyzer for testing
This should be replaced with your actual technical analyzer
"""

class ZerodhaTechnicalAnalyzer:
    def __init__(self, zerodha_client):
        self.zerodha = zerodha_client
    
    async def analyze_symbol_for_options(self, symbol, current_price, 
                                       market_data, trading_style='swing'):
        return {
            'market_bias': 'BULLISH',
            'confidence_score': 0.7,
            'entry_signal': {
                'signal_type': 'BUY',
                'strength': 0.7,
                'reason': 'Test signal'
            },
            'support_resistance': {
                'nearest_support': current_price * 0.95,
                'nearest_resistance': current_price * 1.05
            }
        }
    
    async def quick_intraday_analysis(self, symbol, current_price, market_data):
        return {
            'confidence': 0.7,
            'gap_analysis': {
                'gap_type': 'NONE',
                'gap_percent': 0
            }
        }
'''
        }
        
        created_files = []
        for filename, content in templates.items():
            filepath = self.project_root / filename
            if not filepath.exists():
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    created_files.append(filename)
                    print(f"✅ Created: {filename}")
                except Exception as e:
                    print(f"❌ Failed to create {filename}: {e}")
        
        if created_files:
            print(f"\n📁 Created {len(created_files)} minimal files for testing")
            print("⚠️ Replace these with your actual implementation files")
            return True
        
        return False
    
    def create_env_template(self):
        """Create .env template file"""
        print("\n📝 Creating .env Template...")
        
        env_template = '''# Zerodha API Configuration
ZERODHA_API_KEY=your_api_key_here
ZERODHA_ACCESS_TOKEN=your_access_token_here
ZERODHA_API_SECRET=your_api_secret_here

# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Trading Configuration
INDIAN_WATCHLIST=NIFTY,BANKNIFTY,RELIANCE,TCS,HDFCBANK
SCAN_INTERVAL_MINUTES=5
ACCOUNT_SIZE=100000
RISK_TOLERANCE=medium

# Optional
CLAUDE_API_KEY=your_claude_api_key_here
'''
        
        env_file = self.project_root / '.env.template'
        try:
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(env_template)
            print(f"✅ Created: .env.template")
            print("📋 Copy to .env and fill in your actual values")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env.template: {e}")
            return False
    
    def fix_trading_bot_imports(self):
        """Fix import statements in the trading bot file"""
        print("\n🔧 Fixing Trading Bot Imports...")
        
        # Check both possible bot file names
        bot_files = ['indian_trading_bot.py', 'trading_bot.py']
        bot_file = None
        
        for filename in bot_files:
            filepath = self.project_root / filename
            if filepath.exists():
                bot_file = filepath
                print(f"✅ Found bot file: {filename}")
                break
        
        if not bot_file:
            print("❌ No trading bot file found")
            return False
        
        try:
            with open(bot_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Common import fixes
            fixes = [
                # Fix method names
                ('analyze_symbol_for_options_enhanced', 'analyze_symbol_for_options'),
                # Add error handling
                ('from options_analyzer import', 'try:\n    from options_analyzer import'),
                # Fix missing components
                ('self.market_data_provider', 'self.market_data_provider = ZerodhaMarketDataProvider(self.zerodha)'),
            ]
            
            modified = False
            for old, new in fixes:
                if old in content and new not in content:
                    content = content.replace(old, new)
                    modified = True
                    print(f"✅ Fixed: {old} → {new}")
            
            if modified:
                # Create backup
                backup_file = self.project_root / f'{bot_file.name}.backup'
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"💾 Backup created: {bot_file.name}.backup")
                print("⚠️ Review changes before applying")
                return True
            else:
                print("ℹ️ No import fixes needed")
                return False
                
        except Exception as e:
            print(f"❌ Error fixing imports: {e}")
            return False
    
    def fix_test_file(self):
        """Fix test_bot.py to import from indian_trading_bot.py"""
        print("\n🧪 Fixing Test File...")
        
        test_file = self.project_root / 'test_bot.py'
        if not test_file.exists():
            print("❌ test_bot.py not found")
            return False
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Fixes to make test work with indian_trading_bot.py
            fixes = [
                # Fix the main import
                ("from trading_bot import", "from indian_trading_bot import"),
                ("'trading_bot'", "'indian_trading_bot'"),
                ("trading_bot.", "indian_trading_bot."),
            ]
            
            original_content = content
            modified = False
            
            for old, new in fixes:
                if old in content:
                    content = content.replace(old, new)
                    modified = True
                    print(f"✅ Fixed: {old} → {new}")
            
            if modified:
                # Create backup
                backup_file = self.project_root / 'test_bot.py.backup'
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # Write fixed version
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"💾 Backup created: test_bot.py.backup")
                print("✅ test_bot.py updated to use indian_trading_bot.py")
                return True
            else:
                print("ℹ️ No fixes needed in test file")
                return False
                
        except Exception as e:
            print(f"❌ Error fixing test file: {e}")
            return False

def main():
    """Main diagnostic function"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        diagnostic = TradingBotDiagnostic()
        
        if command == "check":
            diagnostic.run_full_diagnostic()
        elif command == "create-files":
            diagnostic.create_minimal_missing_files()
        elif command == "create-env":
            diagnostic.create_env_template()
        elif command == "fix-imports":
            diagnostic.fix_trading_bot_imports()
        elif command == "fix-test":
            print("🔧 Fixing Test File...")
            diagnostic.fix_test_file()
        elif command == "quick-fix":
            print("🚀 Running Quick Fix...")
            diagnostic.run_full_diagnostic()
            if diagnostic.issues:
                print("\n🔧 Applying Quick Fixes...")
                diagnostic.create_env_template()
                diagnostic.fix_test_file()
                print("\n✅ Quick fixes applied. Re-run tests to check improvements.")
        else:
            print("Available commands:")
            print("  check        - Run full diagnostic")
            print("  create-files - Create missing minimal files")
            print("  create-env   - Create .env template")
            print("  fix-imports  - Fix import issues")
            print("  fix-test     - Fix test file to use indian_trading_bot.py")
            print("  quick-fix    - Apply all quick fixes")
    else:
        diagnostic = TradingBotDiagnostic()
        diagnostic.run_full_diagnostic()

if __name__ == "__main__":
    main()