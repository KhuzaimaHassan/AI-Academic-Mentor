"""
Setup script for Groq API key configuration
"""

import os
from pathlib import Path

def setup_groq_api():
    """Interactive setup for Groq API key"""
    print("🎓 AI Academic Mentor - Groq API Setup")
    print("=" * 50)
    
    print("\n📋 To get your Groq API key:")
    print("1. Go to: https://console.groq.com/keys")
    print("2. Sign up or log in to your Groq account")
    print("3. Create a new API key")
    print("4. Copy the API key (starts with 'gsk_')")
    
    print("\n🔑 Enter your Groq API key:")
    api_key = input("API Key: ").strip()
    
    if api_key and api_key.startswith('gsk_'):
        # Update config.py
        config_path = Path("config.py")
        if config_path.exists():
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Replace the placeholder
            updated_content = content.replace(
                'GROQ_API_KEY = "your_groq_api_key_here"',
                f'GROQ_API_KEY = "{api_key}"'
            )
            
            with open(config_path, 'w') as f:
                f.write(updated_content)
            
            print("✅ API key saved to config.py")
            print("🚀 You can now run the Streamlit app!")
            
        else:
            print("❌ config.py not found")
    else:
        print("❌ Invalid API key format. Should start with 'gsk_'")

if __name__ == "__main__":
    setup_groq_api()

