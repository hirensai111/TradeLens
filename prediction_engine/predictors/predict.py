#!/usr/bin/env python3
"""
Phase 4 ML Prediction Engine - Command Line Tool
Usage: python predict.py [TICKER] [TIMEFRAME]

Examples:
  python predict.py MSFT 7 days
  python predict.py AAPL 2025-07-16 2025-07-30
  python predict.py GOOGL next week
  python predict.py TSLA tomorrow
"""

import sys
import os
sys.path.append('src')

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our Phase 4 components
from prediction_engine.features.feature_engineering import FeatureEngineer
from sklearn.ensemble import RandomForestRegressor
import joblib

class PredictionTool:
    """
    Main command-line prediction tool for Phase 4
    Integrates Excel data + Phase 3 news intelligence + ML models
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.is_trained = False
        self.model_path = "models/trained_model.joblib"
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        print("[ROCKET] Phase 4 ML Prediction Engine")
        print("=" * 50)
        
    def train_model(self, ticker: str, days_back: int = 60) -> bool:
        """Train or load ML model for predictions"""
        try:
            # Try to load existing model
            if os.path.exists(self.model_path):
                print("📚 Loading existing trained model...")
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                print("[OK] Model loaded successfully!")
                return True
            
            # Train new model
            print(f"🧠 Training new model on {days_back} days of data...")
            
            end_date = datetime.now() - timedelta(days=2)
            start_date = end_date - timedelta(days=days_back)
            
            # Create training dataset
            X, y = self.feature_engineer.create_training_dataset(ticker, start_date, end_date)
            
            if len(X) < 10:
                print(f"[ERROR] Insufficient training data: {len(X)} samples")
                return False
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X, y)
            self.is_trained = True
            
            # Save model
            joblib.dump(self.model, self.model_path)
            
            print(f"[OK] Model trained successfully on {len(X)} samples!")
            return True
            
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            return False
    
    def predict_single_day(self, ticker: str, target_date: datetime) -> Dict:
        """Make prediction for a single day"""
        if not self.is_trained:
            print("[ERROR] Model not trained yet!")
            return None
        
        try:
            # Get features for target date
            features = self.feature_engineer.create_features_for_date(ticker, target_date)
            
            # Convert to model input format
            feature_dict = {}
            feature_dict.update(features.price_features)
            feature_dict.update(features.technical_features)
            feature_dict.update(features.sentiment_features)
            feature_dict.update(features.time_features)
            feature_dict.update(features.derived_features)
            
            # Create DataFrame and align with training features
            X_pred = pd.DataFrame([feature_dict])
            
            # Get feature names from a sample training set to align columns
            sample_X, _ = self.feature_engineer.create_training_dataset(
                ticker, 
                datetime.now() - timedelta(days=10),
                datetime.now() - timedelta(days=5)
            )
            
            # Align features
            X_pred = X_pred.reindex(columns=sample_X.columns, fill_value=0)
            
            # Make prediction
            predicted_return = self.model.predict(X_pred)[0]
            
            # Calculate confidence based on prediction variance
            # Use model's estimators to get prediction variance
            predictions = [tree.predict(X_pred)[0] for tree in self.model.estimators_]
            prediction_std = np.std(predictions)
            confidence = max(0.1, min(0.95, 1.0 - (prediction_std * 10)))
            
            # Get current price and calculate predicted price
            current_price = feature_dict.get('close', 400)
            predicted_close = current_price * (1 + predicted_return)
            
            # Generate OHLC predictions (simplified approach)
            volatility = abs(predicted_return) + 0.01  # Base volatility
            predicted_open = current_price * (1 + predicted_return * 0.3)  # Partial gap
            predicted_high = predicted_close * (1 + volatility)
            predicted_low = predicted_close * (1 - volatility)
            
            return {
                'date': target_date,
                'ticker': ticker,
                'current_price': current_price,
                'predicted_return': predicted_return,
                'predicted_open': predicted_open,
                'predicted_high': predicted_high,
                'predicted_low': predicted_low,
                'predicted_close': predicted_close,
                'confidence': confidence,
                'key_factors': self._get_key_factors(features, predicted_return)
            }
            
        except Exception as e:
            print(f"[ERROR] Prediction failed for {target_date.date()}: {e}")
            return None
    
    def _get_key_factors(self, features, predicted_return) -> List[str]:
        """Generate key factors driving the prediction"""
        factors = []
        
        # Technical factors
        tech_features = features.technical_features
        if 'tech_rsi' in tech_features:
            rsi = tech_features['tech_rsi']
            if rsi > 70:
                factors.append("RSI indicates overbought conditions")
            elif rsi < 30:
                factors.append("RSI shows oversold bounce potential")
        
        if 'macd_bullish' in tech_features and tech_features['macd_bullish'] == 1:
            factors.append("MACD shows bullish momentum")
        
        # Sentiment factors
        sentiment_features = features.sentiment_features
        sentiment_1d = sentiment_features.get('sentiment_1d', 0)
        if abs(sentiment_1d) > 0.1:
            direction = "positive" if sentiment_1d > 0 else "negative"
            factors.append(f"Recent news sentiment is {direction}")
        
        news_volume = sentiment_features.get('news_volume_1d', 0)
        if news_volume > 5:
            factors.append(f"High news activity ({int(news_volume)} articles)")
        
        # Price factors
        price_features = features.price_features
        if 'volume_ratio' in price_features and price_features['volume_ratio'] > 1.5:
            factors.append("Above-average trading volume")
        
        # Direction factors
        if predicted_return > 0.01:
            factors.append("Technical indicators suggest upward momentum")
        elif predicted_return < -0.01:
            factors.append("Bearish signals detected")
        else:
            factors.append("Neutral price action expected")
        
        return factors[:5]  # Return top 5 factors
    
    def predict_multiple_days(self, ticker: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Make predictions for multiple days"""
        predictions = []
        current_date = start_date
        
        print(f"[UP] Generating predictions from {start_date.date()} to {end_date.date()}...")
        
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                prediction = self.predict_single_day(ticker, current_date)
                if prediction:
                    predictions.append(prediction)
                    print(f"   [OK] {current_date.date()}: ${prediction['predicted_close']:.2f} ({prediction['confidence']:.0%} confidence)")
                else:
                    print(f"   [ERROR] {current_date.date()}: Prediction failed")
            
            current_date += timedelta(days=1)
        
        return predictions
    
    def format_prediction_report(self, predictions: List[Dict]) -> str:
        """Generate formatted prediction report"""
        if not predictions:
            return "[ERROR] No predictions available"
        
        ticker = predictions[0]['ticker']
        start_date = predictions[0]['date']
        end_date = predictions[-1]['date']
        
        # Calculate summary statistics
        starting_price = predictions[0]['current_price']
        ending_price = predictions[-1]['predicted_close']
        total_change = (ending_price - starting_price) / starting_price
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        
        # Determine risk level
        volatility = np.std([p['predicted_return'] for p in predictions])
        if volatility < 0.02:
            risk_level = "Low"
        elif volatility < 0.04:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Generate report
        report = f"""
[TARGET] STOCK PREDICTION REPORT - {ticker}
{'=' * 60}
Prediction Period: {start_date.date()} to {end_date.date()} ({len(predictions)} days)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Confidence: {avg_confidence:.0%} | Risk Level: {risk_level}

[CHART] DAILY PREDICTIONS:
{'=' * 60}"""
        
        for pred in predictions:
            direction = "[UP]" if pred['predicted_return'] > 0 else "[DOWN]"
            change_pct = pred['predicted_return'] * 100
            
            report += f"""
📅 {pred['date'].strftime('%Y-%m-%d (%A)')}
   Open: ${pred['predicted_open']:.2f}  High: ${pred['predicted_high']:.2f}  Low: ${pred['predicted_low']:.2f}  Close: ${pred['predicted_close']:.2f}
   Confidence: {pred['confidence']:.0%} | Change: {direction} {change_pct:+.2f}%
   Key Drivers: {', '.join(pred['key_factors'][:2])}"""
        
        report += f"""

[UP] SUMMARY ANALYSIS:
{'=' * 60}
Starting Price: ${starting_price:.2f}
Ending Price: ${ending_price:.2f}
Total Change: {total_change:+.1%} (${ending_price - starting_price:+.2f})
Average Confidence: {avg_confidence:.0%}
Risk Level: {risk_level}

[TARGET] KEY INSIGHTS:
"""
        
        # Generate insights
        bullish_days = sum(1 for p in predictions if p['predicted_return'] > 0)
        trend = "Bullish" if bullish_days > len(predictions) / 2 else "Bearish"
        
        report += f"• Overall trend: {trend} ({bullish_days}/{len(predictions)} positive days)\n"
        report += f"• Price volatility: {volatility:.1%} (indicating {risk_level.lower()} risk)\n"
        report += f"• Confidence trend: {'Stable' if max([p['confidence'] for p in predictions]) - min([p['confidence'] for p in predictions]) < 0.2 else 'Variable'}\n"
        
        return report
    
    def parse_date_input(self, date_input: str) -> Tuple[datetime, datetime]:
        """Parse various date input formats"""
        today = datetime.now()
        
        if date_input.lower() == "tomorrow":
            start = today + timedelta(days=1)
            return start, start
        
        elif "days" in date_input.lower():
            try:
                days = int(date_input.split()[0])
                start = today + timedelta(days=1)
                end = start + timedelta(days=days-1)
                return start, end
            except:
                raise ValueError(f"Invalid days format: {date_input}")
        
        elif "week" in date_input.lower():
            if "next" in date_input.lower():
                start = today + timedelta(days=1)
                end = start + timedelta(days=6)
                return start, end
            else:
                raise ValueError(f"Invalid week format: {date_input}")
        
        else:
            # Try to parse as date range
            try:
                if " " in date_input:
                    dates = date_input.split()
                    if len(dates) == 2:
                        start = datetime.strptime(dates[0], "%Y-%m-%d")
                        end = datetime.strptime(dates[1], "%Y-%m-%d")
                        return start, end
                
                # Single date
                date = datetime.strptime(date_input, "%Y-%m-%d")
                return date, date
            except:
                raise ValueError(f"Invalid date format: {date_input}. Use YYYY-MM-DD or 'X days' or 'tomorrow'")


def main():
    """Main command line interface"""
    parser = argparse.ArgumentParser(
        description="Phase 4 ML Stock Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py MSFT 7 days
  python predict.py AAPL 2025-07-16 2025-07-30
  python predict.py GOOGL next week
  python predict.py TSLA tomorrow
        """
    )
    
    parser.add_argument('ticker', help='Stock ticker symbol (e.g., MSFT, AAPL)')
    parser.add_argument('timeframe', nargs='+', help='Timeframe (e.g., "7 days", "tomorrow", "2025-07-16 2025-07-30")')
    parser.add_argument('--retrain', action='store_true', help='Force retrain the model')
    
    args = parser.parse_args()
    
    # Initialize prediction tool
    tool = PredictionTool()
    
    ticker = args.ticker.upper()
    timeframe = ' '.join(args.timeframe)
    
    try:
        # Parse timeframe
        start_date, end_date = tool.parse_date_input(timeframe)
        
        print(f"[CHART] Ticker: {ticker}")
        print(f"📅 Period: {start_date.date()} to {end_date.date()}")
        print()
        
        # Train or load model
        if args.retrain or not tool.is_trained:
            success = tool.train_model(ticker)
            if not success:
                print("[ERROR] Failed to train model. Exiting.")
                return
        else:
            tool.train_model(ticker)  # This will load existing model
        
        # Make predictions
        if start_date == end_date:
            # Single day prediction
            prediction = tool.predict_single_day(ticker, start_date)
            if prediction:
                print(f"[TARGET] Prediction for {ticker} on {start_date.date()}:")
                print(f"   Current Price: ${prediction['current_price']:.2f}")
                print(f"   Predicted Close: ${prediction['predicted_close']:.2f}")
                print(f"   Expected Change: {prediction['predicted_return']:+.2%}")
                print(f"   Confidence: {prediction['confidence']:.0%}")
                print(f"   Key Factors: {', '.join(prediction['key_factors'][:3])}")
        else:
            # Multiple day predictions
            predictions = tool.predict_multiple_days(ticker, start_date, end_date)
            if predictions:
                report = tool.format_prediction_report(predictions)
                print(report)
                
                # Save report to file
                filename = f"prediction_report_{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.txt"
                with open(filename, 'w') as f:
                    f.write(report)
                print(f"\n📄 Report saved to: {filename}")
        
    except ValueError as e:
        print(f"[ERROR] Error: {e}")
        print("\nValid formats:")
        print("  • 7 days")
        print("  • tomorrow") 
        print("  • next week")
        print("  • 2025-07-16 2025-07-30")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Prediction cancelled by user")
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()