# debug_contract_selection.py
from zerodha_api_client import ZerodhaAPIClient
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_zerodha_option_symbol(symbol: str, strike: float, expiry: str, option_type: str) -> str:
    """Create Zerodha option symbol format"""
    try:
        expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
        year_suffix = expiry_date.strftime('%y')
        month_str = expiry_date.strftime('%b').upper()
        option_suffix = 'CE' if option_type.upper() in ['CALL', 'CE'] else 'PE'
        strike_str = str(int(strike))
        
        if symbol.upper() in ['NIFTY', 'NIFTY50', 'NIFTY 50']:
            symbol_prefix = 'NIFTY'
        elif symbol.upper() in ['BANKNIFTY', 'NIFTY BANK', 'BANK NIFTY']:
            symbol_prefix = 'BANKNIFTY'
        else:
            symbol_prefix = symbol.upper()
        
        return f"{symbol_prefix}{year_suffix}{month_str}{strike_str}{option_suffix}"
        
    except Exception as e:
        logger.error(f"Error creating symbol: {e}")
        return None

def debug_contract_selection():
    """Debug why contract selection is failing"""
    
    print("🔍 Debugging Contract Selection for NIFTY")
    print("=" * 60)
    
    # Initialize Zerodha client
    try:
        zerodha = ZerodhaAPIClient()
        print(f"[OK] Zerodha initialized with {len(zerodha.symbol_map)} instruments")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Zerodha: {e}")
        return
    
    # Get current NIFTY price
    try:
        quotes = zerodha.get_live_quotes(['NIFTY 50'])
        current_price = quotes['NIFTY 50']['price']
        print(f"[UP] Current NIFTY Price: ₹{current_price}")
    except Exception as e:
        print(f"[ERROR] Failed to get NIFTY price: {e}")
        current_price = 24574.2  # Fallback from your logs
        print(f"[UP] Using fallback NIFTY Price: ₹{current_price}")
    
    # Generate expiry (October monthly expiry)
    expiry = "2025-10-28"
    print(f"📅 Using expiry: {expiry}")
    
    # Generate strikes around current price
    atm_strike = round(current_price / 50) * 50  # Round to nearest 50
    strikes = []
    for i in range(-5, 6):  # 11 strikes total
        strikes.append(atm_strike + (i * 50))
    
    print(f"[TARGET] ATM Strike: {atm_strike}")
    print(f"[CHART] Testing {len(strikes)} strikes: {strikes[0]} to {strikes[-1]}")
    
    # Generate option symbols
    option_symbols = []
    for strike in strikes:
        for option_type in ['CE', 'PE']:
            symbol = create_zerodha_option_symbol('NIFTY', strike, expiry, option_type)
            if symbol:
                option_symbols.append(symbol)
    
    print(f"🏷️  Generated {len(option_symbols)} option symbols")
    
    # Test if symbols exist
    print("\n📋 Checking if symbols exist in instruments:")
    existing_symbols = []
    for symbol in option_symbols[:10]:  # Check first 10
        if symbol in zerodha.symbol_map:
            existing_symbols.append(symbol)
            print(f"[OK] {symbol}")
        else:
            print(f"[ERROR] {symbol}")
    
    if not existing_symbols:
        print("[WARNING]  No symbols found in instruments!")
        return
    
    # Try to get live quotes for existing symbols
    print(f"\n[SIGNAL] Fetching live quotes for {len(existing_symbols)} existing symbols...")
    try:
        live_quotes = zerodha.get_live_quotes(existing_symbols)
        print(f"[OK] Successfully fetched {len(live_quotes)} quotes")
        
        # Analyze the quote data
        print("\n[CHART] Quote Analysis:")
        print("-" * 50)
        
        valid_contracts = []
        for symbol, quote_data in live_quotes.items():
            price = quote_data.get('price', 0)
            bid = quote_data.get('bid', 0)
            ask = quote_data.get('ask', 0)
            volume = quote_data.get('volume', 0)
            oi = quote_data.get('oi', 0)
            
            print(f"{symbol:<20} Price: ₹{price:<8.2f} Bid: ₹{bid:<8.2f} Ask: ₹{ask:<8.2f} Vol: {volume:<8} OI: {oi}")
            
            # Check what makes a contract "suitable"
            # Common criteria:
            # 1. Price > 0
            # 2. Reasonable bid-ask spread
            # 3. Some liquidity (volume/OI)
            
            is_suitable = True
            reasons = []
            
            if price <= 0:
                is_suitable = False
                reasons.append("No price")
            
            if bid <= 0 or ask <= 0:
                is_suitable = False
                reasons.append("No bid/ask")
            
            if ask > 0 and bid > 0:
                spread = ((ask - bid) / ask) * 100
                if spread > 50:  # More than 50% spread
                    is_suitable = False
                    reasons.append(f"Wide spread ({spread:.1f}%)")
            
            if price > 0 and price < 1:  # Very cheap options might be filtered
                reasons.append("Very cheap option")
            
            if volume == 0 and oi == 0:
                is_suitable = False
                reasons.append("No liquidity")
            
            if is_suitable:
                valid_contracts.append(symbol)
                print(f"  [OK] SUITABLE")
            else:
                print(f"  [ERROR] NOT SUITABLE: {', '.join(reasons)}")
        
        print(f"\n[UP] Summary:")
        print(f"  Total symbols checked: {len(live_quotes)}")
        print(f"  Suitable contracts: {len(valid_contracts)}")
        
        if len(valid_contracts) == 0:
            print("🔍 ISSUE FOUND: No contracts meet suitability criteria!")
            print("\nPossible causes:")
            print("1. All options have zero price/bid/ask")
            print("2. Spread filters are too strict") 
            print("3. Liquidity filters are too strict")
            print("4. Price filters are eliminating valid options")
            
            # Show the actual filtering criteria being used
            print("\n[WRENCH] Suggested fixes:")
            print("1. Relax liquidity requirements for test mode")
            print("2. Allow wider bid-ask spreads")
            print("3. Accept options with zero volume but non-zero OI")
            print("4. Lower minimum price thresholds")
        else:
            print(f"[OK] Found {len(valid_contracts)} suitable contracts")
            for contract in valid_contracts[:3]:
                print(f"  📄 {contract}")
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch live quotes: {e}")
        logger.error("Quote fetching failed", exc_info=True)

if __name__ == "__main__":
    debug_contract_selection()