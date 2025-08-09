#!/bin/bash

# Instructions for running with real LLM
echo "🚀 LLM-Guided Optuna Setup Instructions"
echo "========================================"
echo ""
echo "To run with a real LLM, you have several options:"
echo ""

echo "1. 🔑 OpenAI (GPT-4):"
echo "   export OPENAI_API_KEY='your-openai-api-key'"
echo "   python real_llm_demo.py"
echo ""

echo "2. 🤖 Anthropic Claude:"
echo "   export ANTHROPIC_API_KEY='your-claude-api-key'"
echo "   # Modify real_llm_demo.py to use 'claude-3-sonnet-20240229'"
echo "   python real_llm_demo.py"
echo ""

echo "3. 🔓 Ollama (Local):"
echo "   # Install ollama and run: ollama pull llama3"
echo "   # Modify real_llm_demo.py to use 'ollama/llama3'"
echo "   python real_llm_demo.py"
echo ""

echo "4. ☁️  Other providers:"
echo "   # LiteLLM supports 100+ models!"
echo "   # Examples: 'gemini-pro', 'cohere/command', 'replicate/meta/llama-2-70b'"
echo ""

echo "🎯 Quick Test (no API key needed):"
echo "   python demo.py"
echo ""

echo "📚 Example API Key Usage:"
echo '   OPENAI_API_KEY="sk-..." python real_llm_demo.py'
echo ""

# Check if we're in the right directory
if [ ! -f "real_llm_demo.py" ]; then
    echo "❌ Please run this from the llm_guided directory!"
    exit 1
fi

# Check if API key is set
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✅ OpenAI API key detected! Ready to run."
    echo "Run: python real_llm_demo.py"
elif [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "✅ Anthropic API key detected!"
    echo "Modify real_llm_demo.py to use 'claude-3-sonnet-20240229', then run it."
else
    echo "💡 No API key detected. Set one of the above environment variables."
    echo ""
    echo "🆓 Or try the mock demo: python demo.py"
fi