import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("🔧 SETUP DIAGNOSTIC")
print("=" * 50)

# Check .env file
env_file = ".env"
if os.path.exists(env_file):
    print(f"✅ .env file exists at: {os.path.abspath(env_file)}")
    with open(env_file, 'r') as f:
        content = f.read()
        if 'GEMINI_API_KEY' in content:
            print("✅ GEMINI_API_KEY found in .env file")
        else:
            print("❌ GEMINI_API_KEY not found in .env file")
else:
    print("❌ .env file not found")

# Check environment variables
print("\n🔍 Environment Variables:")
for key in ["GEMINI_API_KEY", "GOOGLE_AI_API_KEY"]:
    value = os.getenv(key)
    if value:
        print(f"✅ {key}: {value[:10]}...{value[-4:]}")
    else:
        print(f"❌ {key}: Not set")

# Check packages
print("\n📦 Package Check:")
try:
    import google.generativeai as genai
    print("✅ google-generativeai: Installed")
except ImportError:
    print("❌ google-generativeai: Not installed")
    print("   Install with: pip install google-generativeai")

try:
    from dotenv import load_dotenv
    print("✅ python-dotenv: Installed")
except ImportError:
    print("❌ python-dotenv: Not installed")
    print("   Install with: pip install python-dotenv")

print("\n🚀 If all checks pass, run your enhanced analysis!")