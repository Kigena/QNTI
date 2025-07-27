@echo off
echo.
echo ===============================================
echo  QNTI Enhanced Market Intelligence Installer
echo ===============================================
echo.
echo This will install the required packages for:
echo - Real-time market data from Yahoo Finance
echo - Technical indicators (RSI, MACD, Bollinger Bands)
echo - Historical data analysis (up to 3 months)
echo - Enhanced insights for all major pairs, gold, indices, and crypto
echo.

pause

echo Installing required packages...
echo.

pip install yfinance pandas numpy requests

echo.
if %ERRORLEVEL% equ 0 (
    echo ✅ Installation completed successfully!
    echo.
    echo Your intelligence board will now provide:
    echo   📊 Real-time prices for major forex pairs
    echo   🥇 Gold, Silver, and Crude Oil data  
    echo   📈 Major indices (S&P 500, NASDAQ, Dow, FTSE, DAX)
    echo   ₿ Bitcoin and Ethereum prices
    echo   📉 Technical indicators (RSI, MACD, volatility)
    echo   🎯 Smart insights based on real market conditions
    echo.
    echo Restart your QNTI system to activate enhanced intelligence!
) else (
    echo ❌ Installation failed. Please check your internet connection.
    echo Or run manually: pip install yfinance pandas numpy requests
)

echo.
pause 