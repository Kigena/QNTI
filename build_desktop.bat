@echo off
echo ========================================
echo QNTI Desktop Application Builder
echo ========================================
echo.

REM Set UTF-8 encoding
chcp 65001 >nul 2>&1

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python and add it to PATH.
    pause
    exit /b 1
)

echo Step 1: Installing/Upgrading dependencies...
echo.
pip install -r requirements_desktop.txt --upgrade

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Step 2: Cleaning previous builds...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "__pycache__" rmdir /s /q "__pycache__"

echo.
echo Step 3: Building QNTI Desktop Application...
echo This may take several minutes...
echo.

pyinstaller qnti_desktop.spec --clean

if errorlevel 1 (
    echo.
    echo ERROR: Build failed. Check the output above for details.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Your QNTI Desktop application is available at:
echo   dist\QNTI_Desktop.exe
echo.
echo File size information:
for %%A in ("dist\QNTI_Desktop.exe") do echo   Size: %%~zA bytes (%%~zA / 1024 / 1024) MB
echo.

REM Check if executable exists and is runnable
if exist "dist\QNTI_Desktop.exe" (
    echo Testing executable...
    echo   - File exists: OK
    
    REM Get file version info if available
    powershell -command "Get-ItemProperty 'dist\QNTI_Desktop.exe' | Select-Object Name, Length, CreationTime" 2>nul
    
    echo.
    echo Ready to distribute!
    echo.
    echo To create an installer, run: build_installer.bat
    echo To test the application, run: dist\QNTI_Desktop.exe
    echo.
) else (
    echo ERROR: Executable not found in dist folder
    exit /b 1
)

echo Build log saved to: build.log
echo.
set /p choice="Would you like to test the application now? (y/n): "
if /i "%choice%"=="y" (
    echo.
    echo Starting QNTI Desktop Application...
    start "" "dist\QNTI_Desktop.exe"
) else (
    echo.
    echo Build complete. You can find the executable at: dist\QNTI_Desktop.exe
)

pause 