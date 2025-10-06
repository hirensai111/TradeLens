#!/usr/bin/env python3
"""
Helper script to get Zerodha access token
Zerodha only provides API Key and Secret - access token must be generated daily
"""

import os
from kiteconnect import KiteConnect

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("📄 Loaded environment variables from .env file")
except ImportError:
    print("💡 python-dotenv not installed. Using system environment variables")

def get_access_token():
    """Get access token for Zerodha API"""
    
    print("\n🔑 ZERODHA ACCESS TOKEN GENERATOR")
    print("=" * 50)
    print("ℹ️  Note: Zerodha access tokens are valid for 1 day only")
    print("ℹ️  You'll need to regenerate this token daily")
    print()
    
    # Get API credentials
    api_key = os.getenv("ZERODHA_API_KEY")
    api_secret = os.getenv("ZERODHA_API_SECRET")
    
    print("📋 Current .env file status:")
    print(f"   API Key: {'✅ Found' if api_key else '❌ Not found'}")
    print(f"   API Secret: {'✅ Found' if api_secret else '❌ Not found'}")
    print()
    
    if not api_key:
        api_key = input("Enter your Zerodha API Key: ").strip()
    else:
        print(f"🔑 Using API Key from .env: {api_key}")
    
    if not api_secret:
        api_secret = input("Enter your Zerodha API Secret: ").strip()
    else:
        print(f"🔐 Using API Secret from .env: {api_secret[:8]}...")
    
    if not api_key or not api_secret:
        print("❌ Both API Key and API Secret are required")
        return
    
    try:
        # Initialize KiteConnect
        kite = KiteConnect(api_key=api_key)
        
        # Generate login URL
        login_url = f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"
        
        print(f"\n📱 Step 1: Visit this URL to login:")
        print(f"   {login_url}")
        print(f"\n📋 Step 2: After login, you'll be redirected to a URL like:")
        print(f"   https://127.0.0.1:5000/?request_token=XXXXXX&action=login&status=success")
        print(f"\n🔍 Step 3: Copy the 'request_token' value from the URL")
        
        # Get request token from user
        request_token = input(f"\n✍️  Enter the request_token: ").strip()
        
        if not request_token:
            print("❌ Request token is required")
            return
        
        # Generate session
        print(f"\n🔄 Generating access token...")
        data = kite.generate_session(request_token, api_secret=api_secret)
        
        access_token = data["access_token"]
        user_id = data.get("user_id", "N/A")
        user_name = data.get("user_name", "N/A")
        
        print(f"\n✅ SUCCESS! Access token generated")
        print("=" * 50)
        print(f"🔑 Access Token: {access_token}")
        print(f"👤 User ID: {user_id}")
        print(f"👤 User Name: {user_name}")
        
        print(f"\n📝 Add this to your .env file:")
        print(f"ZERODHA_ACCESS_TOKEN={access_token}")
        
        # Try to update .env file automatically
        try:
            env_file_path = ".env"
            
            # Read existing .env content or create new
            env_lines = []
            if os.path.exists(env_file_path):
                with open(env_file_path, 'r') as f:
                    env_lines = f.read().strip().split('\n')
            
            # Update or add credentials
            api_key_found = False
            api_secret_found = False
            access_token_found = False
            
            for i, line in enumerate(env_lines):
                if line.startswith('ZERODHA_API_KEY='):
                    env_lines[i] = f'ZERODHA_API_KEY={api_key}'
                    api_key_found = True
                elif line.startswith('ZERODHA_API_SECRET='):
                    env_lines[i] = f'ZERODHA_API_SECRET={api_secret}'
                    api_secret_found = True
                elif line.startswith('ZERODHA_ACCESS_TOKEN='):
                    env_lines[i] = f'ZERODHA_ACCESS_TOKEN={access_token}'
                    access_token_found = True
            
            # Add missing entries
            if not api_key_found:
                env_lines.append(f'ZERODHA_API_KEY={api_key}')
            if not api_secret_found:
                env_lines.append(f'ZERODHA_API_SECRET={api_secret}')
            if not access_token_found:
                env_lines.append(f'ZERODHA_ACCESS_TOKEN={access_token}')
            
            # Write back to .env file
            with open(env_file_path, 'w') as f:
                f.write('\n'.join(filter(None, env_lines)) + '\n')
            
            print(f"✅ Updated {env_file_path} with all credentials")
            
        except Exception as e:
            print(f"⚠️  Could not update .env file automatically: {e}")
            print("💡 Please add the credentials to your .env file manually")
        
        print(f"\n🎉 You can now run the test scripts!")
        print(f"⏰ Remember: This access token expires daily at 3:30 PM IST")
        
    except Exception as e:
        print(f"❌ Error generating access token: {e}")
        print(f"\n💡 Common issues:")
        print(f"   - Make sure you completed the login in browser")
        print(f"   - Check that request_token is copied correctly (no extra spaces)")
        print(f"   - Verify your API secret is correct")
        print(f"   - Make sure you're using the correct Zerodha account")
        print(f"   - Ensure your app is approved and activated on Zerodha Developer Console")

if __name__ == "__main__":
    get_access_token()