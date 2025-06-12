#!/usr/bin/env python3
"""
Setup script for Google Gemini integration with GRC Automation.
"""

import os
import sys

def check_gemini_setup():
    """Check if Gemini is properly configured."""
    print("🔧 Checking Gemini Setup...")
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("❌ No Gemini API key found!")
        print("💡 Please set one of these environment variables:")
        print("   export GEMINI_API_KEY=your_api_key_here")
        print("   export GOOGLE_AI_API_KEY=your_api_key_here")
        print("\n🌐 Get your API key from: https://makersuite.google.com/app/apikey")
        return False
    else:
        print("✅ Gemini API key found!")
    
    # Check for google-generativeai package
    try:
        import google.generativeai
        print("✅ google-generativeai package is installed!")
    except ImportError:
        print("❌ google-generativeai package not found!")
        print("💡 Install it with: pip install google-generativeai")
        return False
    
    # Check data files
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_files = [
        "preprocessing/policies/satim_chunks_cleaned.json",
        "preprocessing/norms/pci_dss_chunks.json"
    ]
    
    for file_path in data_files:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            print(f"✅ Found data file: {file_path}")
        else:
            print(f"⚠️  Data file not found: {file_path}")
    
    print("\n🚀 Setup check complete!")
    return True

def test_gemini_connection():
    """Test connection to Gemini API."""
    print("\n🤖 Testing Gemini connection...")
    
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            print("❌ No API key available for testing!")
            return False
        
        genai.configure(api_key=api_key)
        
        # Test with a simple model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Hello! Please respond with 'Gemini is working!'")
        
        if response and response.text:
            print(f"✅ Gemini responded: {response.text.strip()}")
            return True
        else:
            print("❌ No response from Gemini!")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Gemini: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Google Gemini Flash 2.0 Setup for GRC Automation")
    print("="*55)
    
    if check_gemini_setup():
        if test_gemini_connection():
            print("\n🎉 All systems go! You can now run:")
            print("   python test/test_policy_comparison_gemini.py")
        else:
            print("\n⚠️  Setup complete but connection test failed.")
            print("   Please check your API key and try again.")
    else:
        print("\n❌ Setup incomplete. Please fix the issues above.")