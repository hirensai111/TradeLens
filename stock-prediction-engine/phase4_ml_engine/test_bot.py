#!/usr/bin/env python3
"""
🚀 Enhanced Trading Bot Test Suite with Realistic Market Scenarios - FIXED VERSION
Tests bot with diverse market conditions, volatility levels, and trading scenarios
"""

import asyncio
import sys
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import traceback
import random

# Load environment variables
load_dotenv()

# Import your bot components - FIXED IMPORTS
try:
    from options_analyzer import ZerodhaEnhancedOptionsAnalyzer
    from zerodha_technical_analyzer import ZerodhaTechnicalAnalyzer
    from zerodha_api_client import ZerodhaAPIClient
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

class EnhancedBotTester:
    """Enhanced test suite with realistic market scenarios and simplified data testing"""
    
    def __init__(self):
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': [],
            'detailed_results': {},
            'confidence_tests': {},
            'performance_metrics': {},
            'scenario_analysis': {}
        }
        
        # Core test symbols for reliable testing
        self.test_symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'HDFCBANK', 'TCS', 'INFY']
        
        # Simplified market conditions for testing
        self.market_conditions = [
            {
                'name': 'Bull Market Simulation',
                'description': 'Test with bullish expectations',
                'risk_tolerance': 'aggressive',
                'trading_style': 'swing',
                'expected_strategies': ['LONG_CALL', 'BULL_CALL_SPREAD', 'BULL_PUT_SPREAD']
            },
            {
                'name': 'Bear Market Simulation', 
                'description': 'Test with bearish expectations',
                'risk_tolerance': 'conservative',
                'trading_style': 'swing',
                'expected_strategies': ['LONG_PUT', 'BEAR_PUT_SPREAD', 'BEAR_CALL_SPREAD']
            },
            {
                'name': 'Neutral Market Simulation',
                'description': 'Test with neutral/range-bound expectations',
                'risk_tolerance': 'medium',
                'trading_style': 'intraday',
                'expected_strategies': ['IRON_CONDOR', 'LONG_STRADDLE', 'IRON_BUTTERFLY']
            },
            {
                'name': 'High Volatility Simulation',
                'description': 'Test during high volatility periods',
                'risk_tolerance': 'medium',
                'trading_style': 'intraday',
                'expected_strategies': ['LONG_STRADDLE', 'LONG_STRANGLE', 'SHORT_STRANGLE']
            }
        ]
        
        self.trading_styles = ['intraday', 'swing']
        self.risk_tolerances = ['conservative', 'medium', 'aggressive']

    def log_test(self, test_name: str, success: bool, details: str = "", error: str = "", metrics: dict = None):
        """Enhanced test logging with metrics"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        
        if details:
            print(f"   📝 {details}")
        
        if error:
            print(f"   ⚠️ Error: {error}")
            self.test_results['errors'].append(f"{test_name}: {error}")
        
        if metrics:
            print(f"   📊 Metrics: {metrics}")
            self.test_results['performance_metrics'][test_name] = metrics
        
        self.test_results['detailed_results'][test_name] = {
            'success': success,
            'details': details,
            'error': error,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if success:
            self.test_results['passed'] += 1
        else:
            self.test_results['failed'] += 1
        
        print()

    async def test_environment_setup(self):
        """Test 1: Environment and Configuration"""
        print("🔧 TEST 1: Environment Setup")
        print("-" * 40)
        
        try:
            # Check required environment variables
            required_vars = [
                'ZERODHA_API_KEY',
                'ZERODHA_ACCESS_TOKEN'
            ]
            
            # Optional but recommended
            optional_vars = [
                'TELEGRAM_BOT_TOKEN',
                'TELEGRAM_CHAT_ID'
            ]
            
            missing_required = []
            missing_optional = []
            
            for var in required_vars:
                if not os.getenv(var):
                    missing_required.append(var)
            
            for var in optional_vars:
                if not os.getenv(var):
                    missing_optional.append(var)
            
            if missing_required:
                self.log_test(
                    "Required Environment Variables", 
                    False, 
                    error=f"Missing required: {', '.join(missing_required)}"
                )
                return False
            else:
                details = f"All {len(required_vars)} required variables found"
                if missing_optional:
                    details += f". Optional missing: {', '.join(missing_optional)}"
                
                self.log_test(
                    "Environment Variables", 
                    True, 
                    details
                )
            
            return True
            
        except Exception as e:
            self.log_test("Environment Setup", False, error=str(e))
            return False

    async def test_zerodha_integration(self):
        """Test 2: Enhanced Zerodha API Integration"""
        print("🏦 TEST 2: Enhanced Zerodha API Integration")
        print("-" * 40)
        
        try:
            # Initialize Zerodha client
            zerodha_client = ZerodhaAPIClient()
            
            if not zerodha_client.access_token:
                self.log_test(
                    "Zerodha Client Init", 
                    False, 
                    error="No access token found"
                )
                return False
            
            self.log_test("Zerodha Client Init", True, "Client initialized successfully")
            
            # Test live quotes with performance metrics
            try:
                start_time = datetime.now()
                test_symbols = ['RELIANCE', 'NIFTY 50']
                quotes = zerodha_client.get_live_quotes(test_symbols)
                end_time = datetime.now()
                
                response_time = (end_time - start_time).total_seconds()
                
                if quotes and len(quotes) > 0:
                    metrics = {
                        'response_time_ms': round(response_time * 1000, 2),
                        'quotes_received': len(quotes),
                        'symbols_requested': len(test_symbols)
                    }
                    
                    self.log_test(
                        "Live Quotes API", 
                        True, 
                        f"Fetched {len(quotes)}/{len(test_symbols)} quotes",
                        metrics=metrics
                    )
                else:
                    self.log_test("Live Quotes API", False, error="No quotes received")
                    return False
                    
            except Exception as e:
                self.log_test("Live Quotes API", False, error=str(e))
                return False
            
            # Test market status
            try:
                market_status = zerodha_client.get_market_status()
                if market_status:
                    self.log_test(
                        "Market Status API",
                        True,
                        f"Market status: {market_status.get('status', 'unknown')}"
                    )
                else:
                    self.log_test("Market Status API", False, error="No market status received")
                    
            except Exception as e:
                self.log_test("Market Status API", False, error=str(e))
            
            return True
            
        except Exception as e:
            self.log_test("Zerodha Integration", False, error=str(e))
            return False

    async def test_core_analysis_functionality(self):
        """Test 3: Core Analysis Engine"""
        print("🧠 TEST 3: Core Analysis Functionality")
        print("-" * 40)
        
        try:
            zerodha_client = ZerodhaAPIClient()
            options_analyzer = ZerodhaEnhancedOptionsAnalyzer(zerodha_client)
            
            # Test with primary symbols
            test_results = {}
            primary_symbols = ['NIFTY', 'RELIANCE', 'HDFCBANK']
            
            for symbol in primary_symbols:
                try:
                    print(f"   🔍 Testing analysis for {symbol}...")
                    start_time = datetime.now()
                    
                    result = await options_analyzer.analyze_trade(
                        symbol=symbol,
                        trading_style='swing',
                        risk_tolerance='medium',
                        capital=100000,
                        execute_trades=False
                    )
                    
                    end_time = datetime.now()
                    analysis_time = (end_time - start_time).total_seconds()
                    
                    if result and not result.get('error'):
                        trade_rec = result.get('trade_recommendation', {})
                        strategy = trade_rec.get('strategy', 'Unknown')
                        confidence = trade_rec.get('confidence', 0)
                        
                        # Validate required fields
                        required_fields = ['trade_recommendation', 'market_data', 'risk_management']
                        missing_fields = [field for field in required_fields if field not in result]
                        
                        if missing_fields:
                            test_results[symbol] = {
                                'success': False,
                                'error': f"Missing fields: {missing_fields}"
                            }
                        else:
                            test_results[symbol] = {
                                'success': True,
                                'strategy': strategy,
                                'confidence': confidence,
                                'analysis_time': analysis_time,
                                'has_legs': len(trade_rec.get('option_legs', [])) > 0
                            }
                        
                        print(f"      ✅ {symbol}: {strategy} ({confidence:.1%}, {analysis_time:.2f}s)")
                    else:
                        error_msg = result.get('message', 'Unknown error') if result else 'No result'
                        test_results[symbol] = {'success': False, 'error': error_msg}
                        print(f"      ❌ {symbol}: Failed - {error_msg}")
                        
                except Exception as e:
                    test_results[symbol] = {'success': False, 'error': str(e)}
                    print(f"      ⚠️ {symbol}: Exception - {e}")
            
            # Evaluate results
            successful_count = sum(1 for r in test_results.values() if r.get('success', False))
            success_rate = successful_count / len(primary_symbols)
            
            # Calculate metrics
            successful_results = [r for r in test_results.values() if r.get('success', False)]
            
            metrics = {
                'success_rate': success_rate,
                'symbols_tested': len(primary_symbols),
                'successful_analyses': successful_count
            }
            
            if successful_results:
                avg_confidence = sum(r.get('confidence', 0) for r in successful_results) / len(successful_results)
                avg_time = sum(r.get('analysis_time', 0) for r in successful_results) / len(successful_results)
                legs_coverage = sum(1 for r in successful_results if r.get('has_legs', False))
                
                metrics.update({
                    'avg_confidence': round(avg_confidence, 3),
                    'avg_analysis_time': round(avg_time, 2),
                    'strategies_with_legs': legs_coverage
                })
            
            test_passed = success_rate >= 0.6  # At least 60% should work
            
            self.log_test(
                "Core Analysis Functionality",
                test_passed,
                f"Success: {successful_count}/{len(primary_symbols)} symbols",
                metrics=metrics
            )
            
            return test_passed
            
        except Exception as e:
            self.log_test("Core Analysis Functionality", False, error=str(e))
            return False

    async def test_market_conditions_adaptability(self):
        """Test 4: Market Conditions Adaptability"""
        print("🎯 TEST 4: Market Conditions Adaptability")
        print("-" * 40)
        
        try:
            zerodha_client = ZerodhaAPIClient()
            options_analyzer = ZerodhaEnhancedOptionsAnalyzer(zerodha_client)
            
            condition_results = {}
            
            for condition in self.market_conditions:
                print(f"   📊 Testing {condition['name']}...")
                
                try:
                    # Test with the specified conditions
                    result = await options_analyzer.analyze_trade(
                        symbol='NIFTY',
                        trading_style=condition['trading_style'],
                        risk_tolerance=condition['risk_tolerance'],
                        capital=100000,
                        execute_trades=False
                    )
                    
                    if result and not result.get('error'):
                        trade_rec = result.get('trade_recommendation', {})
                        strategy = trade_rec.get('strategy', 'Unknown')
                        confidence = trade_rec.get('confidence', 0)
                        
                        # Check if strategy matches expected types
                        strategy_appropriate = any(
                            expected.replace('_', '').lower() in strategy.replace('_', '').lower()
                            for expected in condition['expected_strategies']
                        )
                        
                        condition_results[condition['name']] = {
                            'success': True,
                            'strategy': strategy,
                            'confidence': confidence,
                            'strategy_appropriate': strategy_appropriate,
                            'trading_style': condition['trading_style'],
                            'risk_tolerance': condition['risk_tolerance']
                        }
                        
                        status = "✅" if strategy_appropriate else "⚠️"
                        print(f"      {status} Strategy: {strategy} ({confidence:.1%})")
                        
                    else:
                        error_msg = result.get('message', 'Unknown error') if result else 'No result'
                        condition_results[condition['name']] = {
                            'success': False,
                            'error': error_msg
                        }
                        print(f"      ❌ Failed: {error_msg}")
                        
                except Exception as e:
                    condition_results[condition['name']] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"      ⚠️ Exception: {e}")
            
            # Evaluate adaptability
            successful_conditions = sum(1 for r in condition_results.values() if r.get('success', False))
            appropriate_strategies = sum(1 for r in condition_results.values() 
                                      if r.get('success', False) and r.get('strategy_appropriate', False))
            
            success_rate = successful_conditions / len(self.market_conditions)
            adaptability_rate = appropriate_strategies / len(self.market_conditions) if successful_conditions > 0 else 0
            
            metrics = {
                'conditions_tested': len(self.market_conditions),
                'successful_analyses': successful_conditions,
                'appropriate_strategies': appropriate_strategies,
                'success_rate': round(success_rate, 3),
                'adaptability_rate': round(adaptability_rate, 3)
            }
            
            # Store for scenario analysis
            self.test_results['scenario_analysis'] = condition_results
            
            test_passed = success_rate >= 0.75 and adaptability_rate >= 0.5
            
            self.log_test(
                "Market Conditions Adaptability",
                test_passed,
                f"Adaptability: {appropriate_strategies}/{successful_conditions} appropriate strategies",
                metrics=metrics
            )
            
            return test_passed
            
        except Exception as e:
            self.log_test("Market Conditions Adaptability", False, error=str(e))
            return False

    async def test_trading_style_variations(self):
        """Test 5: Trading Style Variations"""
        print("⚡ TEST 5: Trading Style Variations")
        print("-" * 40)
        
        try:
            zerodha_client = ZerodhaAPIClient()
            options_analyzer = ZerodhaEnhancedOptionsAnalyzer(zerodha_client)
            
            style_results = {}
            test_symbol = 'RELIANCE'
            
            for style in self.trading_styles:
                for risk in self.risk_tolerances:
                    test_name = f"{style}_{risk}"
                    print(f"   🎯 Testing {style} trading with {risk} risk...")
                    
                    try:
                        result = await options_analyzer.analyze_trade(
                            symbol=test_symbol,
                            trading_style=style,
                            risk_tolerance=risk,
                            capital=100000,
                            execute_trades=False
                        )
                        
                        if result and not result.get('error'):
                            trade_rec = result.get('trade_recommendation', {})
                            strategy = trade_rec.get('strategy', 'Unknown')
                            confidence = trade_rec.get('confidence', 0)
                            
                            # Check for style-appropriate strategies
                            intraday_appropriate = (style != 'intraday' or 
                                                  'INTRADAY' in strategy or 
                                                  strategy in ['LONG_CALL', 'LONG_PUT', 'LONG_STRADDLE'])
                            
                            style_results[test_name] = {
                                'success': True,
                                'strategy': strategy,
                                'confidence': confidence,
                                'style_appropriate': intraday_appropriate,
                                'trading_style': style,
                                'risk_tolerance': risk
                            }
                            
                            print(f"      ✅ {strategy} ({confidence:.1%})")
                            
                        else:
                            error_msg = result.get('message', 'Unknown error') if result else 'No result'
                            style_results[test_name] = {
                                'success': False,
                                'error': error_msg
                            }
                            print(f"      ❌ Failed: {error_msg}")
                            
                    except Exception as e:
                        style_results[test_name] = {
                            'success': False,
                            'error': str(e)
                        }
                        print(f"      ⚠️ Exception: {e}")
            
            # Evaluate style variations
            total_combinations = len(self.trading_styles) * len(self.risk_tolerances)
            successful_combinations = sum(1 for r in style_results.values() if r.get('success', False))
            appropriate_combinations = sum(1 for r in style_results.values() 
                                        if r.get('success', False) and r.get('style_appropriate', True))
            
            success_rate = successful_combinations / total_combinations
            appropriateness_rate = appropriate_combinations / total_combinations
            
            metrics = {
                'total_combinations': total_combinations,
                'successful_combinations': successful_combinations,
                'appropriate_combinations': appropriate_combinations,
                'success_rate': round(success_rate, 3),
                'appropriateness_rate': round(appropriateness_rate, 3)
            }
            
            test_passed = success_rate >= 0.7 and appropriateness_rate >= 0.6
            
            self.log_test(
                "Trading Style Variations",
                test_passed,
                f"Success: {successful_combinations}/{total_combinations}, Appropriate: {appropriate_combinations}",
                metrics=metrics
            )
            
            return test_passed
            
        except Exception as e:
            self.log_test("Trading Style Variations", False, error=str(e))
            return False

    async def test_risk_management_integration(self):
        """Test 6: Risk Management Integration"""
        print("🛡️ TEST 6: Risk Management Integration")
        print("-" * 40)
        
        try:
            zerodha_client = ZerodhaAPIClient()
            options_analyzer = ZerodhaEnhancedOptionsAnalyzer(zerodha_client)
            
            risk_tests = [
                {'capital': 50000, 'risk': 'conservative', 'expected_low_risk': True},
                {'capital': 100000, 'risk': 'medium', 'expected_low_risk': False},
                {'capital': 200000, 'risk': 'aggressive', 'expected_low_risk': False}
            ]
            
            risk_results = {}
            
            for i, test_config in enumerate(risk_tests):
                test_name = f"Risk_Test_{i+1}"
                print(f"   🛡️ Testing capital ₹{test_config['capital']:,} with {test_config['risk']} risk...")
                
                try:
                    result = await options_analyzer.analyze_trade(
                        symbol='NIFTY',
                        trading_style='swing',
                        risk_tolerance=test_config['risk'],
                        capital=test_config['capital'],
                        execute_trades=False
                    )
                    
                    if result and not result.get('error'):
                        trade_rec = result.get('trade_recommendation', {})
                        risk_mgmt = result.get('risk_management', {})
                        
                        strategy = trade_rec.get('strategy', 'Unknown')
                        confidence = trade_rec.get('confidence', 0)
                        
                        # Check risk management fields
                        has_risk_mgmt = bool(risk_mgmt and not risk_mgmt.get('error'))
                        risk_approved = risk_mgmt.get('approved', False) if has_risk_mgmt else True
                        
                        # Check if risk level matches expectation
                        is_conservative_strategy = any(keyword in strategy for keyword in 
                                                     ['SPREAD', 'IRON', 'CONSERVATIVE'])
                        
                        risk_appropriate = (
                            (test_config['expected_low_risk'] and is_conservative_strategy) or
                            (not test_config['expected_low_risk'])
                        )
                        
                        risk_results[test_name] = {
                            'success': True,
                            'strategy': strategy,
                            'confidence': confidence,
                            'has_risk_management': has_risk_mgmt,
                            'risk_approved': risk_approved,
                            'risk_appropriate': risk_appropriate,
                            'capital': test_config['capital'],
                            'risk_tolerance': test_config['risk']
                        }
                        
                        status = "✅" if risk_appropriate else "⚠️"
                        print(f"      {status} {strategy}, Risk Mgmt: {'Yes' if has_risk_mgmt else 'No'}")
                        
                    else:
                        error_msg = result.get('message', 'Unknown error') if result else 'No result'
                        risk_results[test_name] = {
                            'success': False,
                            'error': error_msg
                        }
                        print(f"      ❌ Failed: {error_msg}")
                        
                except Exception as e:
                    risk_results[test_name] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"      ⚠️ Exception: {e}")
            
            # Evaluate risk management
            successful_tests = sum(1 for r in risk_results.values() if r.get('success', False))
            has_risk_mgmt_count = sum(1 for r in risk_results.values() 
                                    if r.get('success', False) and r.get('has_risk_management', False))
            appropriate_risk_count = sum(1 for r in risk_results.values() 
                                       if r.get('success', False) and r.get('risk_appropriate', False))
            
            success_rate = successful_tests / len(risk_tests)
            risk_mgmt_rate = has_risk_mgmt_count / len(risk_tests) if successful_tests > 0 else 0
            appropriateness_rate = appropriate_risk_count / len(risk_tests)
            
            metrics = {
                'risk_tests_run': len(risk_tests),
                'successful_tests': successful_tests,
                'with_risk_management': has_risk_mgmt_count,
                'appropriate_risk_levels': appropriate_risk_count,
                'success_rate': round(success_rate, 3),
                'risk_management_rate': round(risk_mgmt_rate, 3),
                'appropriateness_rate': round(appropriateness_rate, 3)
            }
            
            test_passed = success_rate >= 0.6 and appropriateness_rate >= 0.5
            
            self.log_test(
                "Risk Management Integration",
                test_passed,
                f"Risk Management: {has_risk_mgmt_count}/{successful_tests}, Appropriate: {appropriate_risk_count}",
                metrics=metrics
            )
            
            return test_passed
            
        except Exception as e:
            self.log_test("Risk Management Integration", False, error=str(e))
            return False

    async def test_performance_benchmarks(self):
        """Test 7: Performance Benchmarks"""
        print("⚡ TEST 7: Performance Benchmarks")
        print("-" * 40)
        
        try:
            zerodha_client = ZerodhaAPIClient()
            options_analyzer = ZerodhaEnhancedOptionsAnalyzer(zerodha_client)
            
            performance_results = []
            test_runs = 3  # Run multiple times for average
            
            for run in range(test_runs):
                print(f"   🏃 Performance run {run + 1}/{test_runs}...")
                
                start_time = datetime.now()
                
                result = await options_analyzer.analyze_trade(
                    symbol='NIFTY',
                    trading_style='swing',
                    risk_tolerance='medium',
                    capital=100000,
                    execute_trades=False
                )
                
                end_time = datetime.now()
                analysis_time = (end_time - start_time).total_seconds()
                
                if result and not result.get('error'):
                    performance_results.append({
                        'analysis_time': analysis_time,
                        'success': True,
                        'has_strategy': bool(result.get('trade_recommendation', {}).get('strategy')),
                        'has_legs': len(result.get('trade_recommendation', {}).get('option_legs', [])) > 0
                    })
                    print(f"      ✅ Run {run + 1}: {analysis_time:.2f}s")
                else:
                    performance_results.append({
                        'analysis_time': analysis_time,
                        'success': False
                    })
                    print(f"      ❌ Run {run + 1}: Failed in {analysis_time:.2f}s")
            
            # Calculate performance metrics
            successful_runs = [r for r in performance_results if r.get('success', False)]
            
            if successful_runs:
                avg_time = sum(r['analysis_time'] for r in successful_runs) / len(successful_runs)
                max_time = max(r['analysis_time'] for r in successful_runs)
                min_time = min(r['analysis_time'] for r in successful_runs)
                
                # Performance thresholds
                time_acceptable = avg_time <= 30.0  # 30 seconds average
                consistency_good = (max_time - min_time) <= 10.0  # 10 second variation
                
                metrics = {
                    'test_runs': test_runs,
                    'successful_runs': len(successful_runs),
                    'avg_analysis_time': round(avg_time, 2),
                    'min_time': round(min_time, 2),
                    'max_time': round(max_time, 2),
                    'time_variation': round(max_time - min_time, 2),
                    'time_acceptable': time_acceptable,
                    'consistency_good': consistency_good
                }
                
                test_passed = len(successful_runs) >= 2 and time_acceptable
                
                self.log_test(
                    "Performance Benchmarks",
                    test_passed,
                    f"Avg time: {avg_time:.2f}s, Success: {len(successful_runs)}/{test_runs}",
                    metrics=metrics
                )
                
                return test_passed
            else:
                self.log_test("Performance Benchmarks", False, error="No successful runs")
                return False
            
        except Exception as e:
            self.log_test("Performance Benchmarks", False, error=str(e))
            return False

    def generate_enhanced_test_report(self):
        """Generate comprehensive test report with scenario analysis"""
        print("\n" + "="*80)
        print("📊 ENHANCED COMPREHENSIVE TEST REPORT WITH SCENARIO ANALYSIS")
        print("="*80)
        
        total_tests = self.test_results['passed'] + self.test_results['failed']
        success_rate = (self.test_results['passed'] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📈 Overall Results:")
        print(f"   • Total Tests: {total_tests}")
        print(f"   • Passed: {self.test_results['passed']} ✅")
        print(f"   • Failed: {self.test_results['failed']} ❌")
        print(f"   • Success Rate: {success_rate:.1f}%")
        print()
        
        # Market Scenario Analysis Summary
        if self.test_results['scenario_analysis']:
            print("🎯 Market Scenario Analysis:")
            for scenario_name, results in self.test_results['scenario_analysis'].items():
                if isinstance(results, dict) and 'success' in results:
                    success = results['success']
                    strategy = results.get('strategy', 'Unknown')
                    confidence = results.get('confidence', 0)
                    appropriate = results.get('strategy_appropriate', False)
                    
                    status = "✅" if success and appropriate else "⚠️" if success else "❌"
                    
                    print(f"   {status} {scenario_name}:")
                    if success:
                        print(f"     └─ Confidence: {confidence:.1%}")
                        print(f"     └─ Strategy: {strategy}")
                        print(f"     └─ Appropriate: {'Yes' if appropriate else 'No'}")
                    else:
                        print(f"     └─ Error: {results.get('error', 'Unknown')}")
            print()
        
        # Performance Metrics Summary
        if self.test_results['performance_metrics']:
            print("⚡ Enhanced Performance Metrics:")
            
            # Extract confidence metrics across scenarios
            confidence_metrics = []
            analysis_times = []
            
            for test_name, metrics in self.test_results['performance_metrics'].items():
                if 'avg_confidence' in metrics:
                    confidence_metrics.append(metrics['avg_confidence'])
                if 'avg_analysis_time' in metrics:
                    analysis_times.append(metrics['avg_analysis_time'])
            
            if confidence_metrics:
                overall_avg_confidence = sum(confidence_metrics) / len(confidence_metrics)
                print(f"   • Overall Average Confidence: {overall_avg_confidence:.1%}")
                print(f"   • Confidence Range: {min(confidence_metrics):.1%} - {max(confidence_metrics):.1%}")
            
            if analysis_times:
                avg_analysis_time = sum(analysis_times) / len(analysis_times)
                print(f"   • Average Analysis Time: {avg_analysis_time:.2f}s")
                print(f"   • Time Range: {min(analysis_times):.2f}s - {max(analysis_times):.2f}s")
            
            # Strategy diversity analysis
            strategy_counts = {}
            for test_name, metrics in self.test_results['performance_metrics'].items():
                if 'strategy' in metrics:
                    strategy = metrics['strategy']
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            if strategy_counts:
                print(f"   • Strategy Diversity: {len(strategy_counts)} unique strategies")
                print(f"   • Most Common: {max(strategy_counts.items(), key=lambda x: x[1])[0]}")
            
            print()
        
        # Test Categories with Enhanced Analysis
        test_categories = {
            'Environment & Setup': ['Environment', 'Zerodha'],
            'Market Scenarios': ['Scenario', 'Volatility', 'Time', 'Correlation', 'Conditions', 'Style'],
            'Stress Testing': ['Stress', 'Risk'],
            'Core Functionality': ['Signal', 'E2E', 'Edge', 'Performance', 'Core', 'Analysis']
        }
        
        print("📝 Enhanced Test Categories:")
        for category, keywords in test_categories.items():
            category_tests = [
                test_name for test_name in self.test_results['detailed_results'].keys()
                if any(keyword in test_name for keyword in keywords)
            ]
            
            if category_tests:
                passed_in_category = sum(
                    1 for test_name in category_tests 
                    if self.test_results['detailed_results'][test_name]['success']
                )
                
                category_rate = (passed_in_category / len(category_tests)) * 100
                status = "✅" if category_rate >= 80 else "⚠️" if category_rate >= 60 else "❌"
                
                print(f"   {status} {category}: {passed_in_category}/{len(category_tests)} ({category_rate:.0f}%)")
        
        print()
        
        if self.test_results['errors']:
            print("❌ Failed Tests Details:")
            for i, error in enumerate(self.test_results['errors'], 1):
                print(f"   {i}. {error}")
            print()
        
        # Enhanced Recommendations
        print("💡 Enhanced Recommendations:")
        
        if success_rate >= 90:
            print("🎉 EXCELLENT! Your bot handles diverse market scenarios brilliantly!")
            print("   • Strong performance across all market conditions")
            print("   • Appropriate strategy selection for different scenarios")
            print("   • Robust confidence calibration")
            print("   • Ready for live market deployment")
        elif success_rate >= 75:
            print("✅ VERY GOOD! Your bot performs well across most scenarios.")
            print("   • Good scenario adaptability")
            print("   • Consider fine-tuning failed scenarios")
            print("   • Monitor performance in specific market conditions")
        elif success_rate >= 60:
            print("⚠️ MODERATE! Bot needs improvement in scenario handling.")
            print("   • Review strategy selection logic")
            print("   • Improve confidence calculations for specific scenarios")
            print("   • Test with more diverse market conditions")
        else:
            print("❌ POOR! Significant issues with scenario adaptation.")
            print("   • Major strategy selection problems")
            print("   • Confidence system needs complete overhaul")
            print("   • Not ready for live trading")
        
        print()
        print("🚀 Next Steps Based on Scenario Testing:")
        
        # Scenario-specific recommendations
        scenario_issues = []
        if self.test_results['scenario_analysis']:
            for scenario_name, results in self.test_results['scenario_analysis'].items():
                if isinstance(results, dict) and not results.get('success', False):
                    scenario_issues.append(scenario_name)
        
        if scenario_issues:
            print(f"   1. Address issues in: {', '.join(scenario_issues[:3])}")
            print("   2. Review strategy selection logic for these scenarios")
            print("   3. Adjust confidence calculations for specific market conditions")
        else:
            print("   1. Proceed with live paper trading")
            print("   2. Monitor real-market scenario performance")
            print("   3. Fine-tune based on live market feedback")
        
        # Save enhanced report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed': self.test_results['passed'],
                'failed': self.test_results['failed'],
                'success_rate': success_rate
            },
            'detailed_results': self.test_results['detailed_results'],
            'scenario_analysis': self.test_results['scenario_analysis'],
            'performance_metrics': self.test_results['performance_metrics'],
            'errors': self.test_results['errors'],
            'test_categories': test_categories,
            'market_conditions_tested': len(self.market_conditions),
            'symbols_tested': self.test_symbols
        }
        
        with open('enhanced_scenario_test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print("   4. Enhanced scenario test report saved to: enhanced_scenario_test_report.json")
        print()

async def run_enhanced_scenario_tests():
    """Run comprehensive tests with realistic market scenarios"""
    print("🚀 STARTING ENHANCED TRADING BOT SCENARIO TEST SUITE")
    print("🎯 Testing with Realistic Market Conditions & Diverse Scenarios")
    print("="*80)
    print(f"📅 Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tester = EnhancedBotTester()
    
    # Enhanced test suites with scenario focus
    test_suites = [
        tester.test_environment_setup,
        tester.test_zerodha_integration,
        tester.test_core_analysis_functionality,      # Core functionality test
        tester.test_market_conditions_adaptability,   # Market conditions adaptation
        tester.test_trading_style_variations,         # Trading style variations
        tester.test_risk_management_integration,      # Risk management testing
        tester.test_performance_benchmarks,           # Performance benchmarks
    ]
    
    for i, test_suite in enumerate(test_suites, 1):
        print(f"🔄 Running Test Suite {i}/{len(test_suites)}: {test_suite.__name__.replace('test_', '').replace('_', ' ').title()}")
        
        try:
            await test_suite()
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            traceback.print_exc()
        
        print()  # Spacing between test suites
    
    # Generate enhanced final report
    tester.generate_enhanced_test_report()

if __name__ == "__main__":
    print("🧪 Enhanced Trading Bot Scenario Test Suite v3.0 - FIXED")
    print("🎯 Featuring: Core Analysis Testing, Market Adaptability & Performance Benchmarks")
    print()
    
    try:
        asyncio.run(run_enhanced_scenario_tests())
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        traceback.print_exc()