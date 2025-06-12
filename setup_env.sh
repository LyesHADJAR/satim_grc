#!/bin/bash

echo "🔧 Setting up environment for GRC Automation with Gemini"
echo "========================================================="

# Install required packages
echo "📦 Installing required packages..."
pip install google-generativeai>=0.3.0

# Create .env file template
echo "📝 Creating .env template..."
cat > .env.example << EOF

# Alternative name (both work)
GOOGLE_AI_API_KEY=AIzaSyCObwH3FxHQt3HSPII0fnidnDCnqA6ufJ0

# Model Configuration
GEMINI_MODEL=gemini-2.0-flash-exp
GEMINI_TEMPERATURE=0.2
GEMINI_MAX_TOKENS=2000
EOF

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Get your API key from: https://makersuite.google.com/app/apikey"
echo "2. Copy .env.example to .env and add your API key"
echo "3. Run: python setup_gemini.py to test your setup"
echo "4. Run: python test/test_policy_comparison_gemini.py"