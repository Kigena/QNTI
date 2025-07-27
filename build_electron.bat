@echo off
echo ========================================
echo QNTI Electron Desktop App Builder
echo ========================================
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found. Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

REM Check if npm is available
npm --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: npm not found. Please install Node.js which includes npm.
    pause
    exit /b 1
)

echo Node.js and npm are available
echo.

REM Install dependencies if node_modules doesn't exist
if not exist "node_modules" (
    echo Installing Node.js dependencies...
    npm install
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
    echo ✓ Dependencies installed
    echo.
) else (
    echo Dependencies already installed
    echo.
)

REM Create src directory if it doesn't exist
if not exist "src" mkdir src

REM Create build directory if it doesn't exist
if not exist "build" mkdir build

REM Build the Electron application
echo Building QNTI Electron Desktop App...
echo This may take several minutes...
echo.

npm run build:win

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    echo Check the output above for details.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.

REM Check if installer was created
if exist "dist-electron\*.exe" (
    echo Your QNTI Electron app is available in the dist-electron folder:
    dir "dist-electron\*.exe" /b
    echo.
    
    echo Distribution options:
    echo   • Windows Installer (.exe)
    echo   • Portable Version (.exe)
    echo.
    
    set /p choice="Would you like to open the dist-electron folder? (y/n): "
    if /i "%choice%"=="y" (
        echo Opening dist-electron folder...
        start "" "dist-electron"
    )
) else (
    echo WARNING: No installer found in dist-electron folder
    echo Check the build output above for any errors
)

echo.
echo Build complete!
pause 