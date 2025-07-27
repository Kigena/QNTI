@echo off
echo ========================================
echo QNTI Electron Desktop App - Dev Mode
echo ========================================
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found. Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo Node.js version:
node --version
echo.

REM Install dependencies if needed
if not exist "node_modules" (
    echo Installing dependencies...
    npm install
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
    echo.
)

REM Create src directory if it doesn't exist
if not exist "src" mkdir src

REM Check if main files exist
if not exist "src\main.js" (
    echo ERROR: src\main.js not found!
    echo Please ensure all Electron source files are in place.
    pause
    exit /b 1
)

if not exist "src\preload.js" (
    echo ERROR: src\preload.js not found!
    echo Please ensure all Electron source files are in place.
    pause
    exit /b 1
)

echo Starting QNTI Electron app in development mode...
echo.
echo Features in dev mode:
echo   • Hot reload enabled
echo   • Developer tools available (F12)
echo   • Debug logging enabled
echo   • Server auto-start
echo.
echo Press Ctrl+C to stop the application
echo.

npm run dev

echo.
echo QNTI Electron app stopped.
pause 