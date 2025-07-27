#!/bin/bash
# QNTI Trading System - Railway.app Deployment Script
# Quick cloud deployment without local setup

echo "🚂 QNTI Trading System - Railway Deployment"
echo "============================================"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "📦 Installing Railway CLI..."
    
    # Install Railway CLI
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install railway
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl -fsSL https://railway.app/install.sh | sh
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        echo "Please install Railway CLI manually: https://docs.railway.app/develop/cli"
        exit 1
    fi
fi

echo "🔐 Logging into Railway..."
railway login

echo "🚀 Creating new Railway project..."
railway init

echo "📋 Setting up environment variables..."
echo "Please configure these in Railway dashboard:"
echo "- MT5_LOGIN: Your MetaTrader 5 login"
echo "- MT5_PASSWORD: Your MetaTrader 5 password"
echo "- MT5_SERVER: Your MetaTrader 5 server"
echo "- SECRET_KEY: Random secret key for Flask"

echo "🔧 Deploying to Railway..."
railway up

echo "✅ Deployment initiated!"
echo ""
echo "📱 Next Steps:"
echo "1. Go to railway.app/dashboard"
echo "2. Find your project"
echo "3. Configure environment variables"
echo "4. Access your trading system via the generated URL"
echo ""
echo "🎯 Your QNTI system will be available at:"
echo "https://your-project-name.railway.app" 