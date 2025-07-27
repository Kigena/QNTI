@echo off
title QNTI Trading System Installer

echo ========================================
echo    QNTI Trading System Installer
echo ========================================
echo.

REM Check for administrator privileges
net session >nul 2>&1
if %errorlevel% == 0 (
    echo Running with administrator privileges...
    goto :install
) else (
    echo This installer requires administrator privileges.
    echo.
    echo Requesting administrator access...
    echo Right-click this file and select "Run as administrator"
    echo OR
    echo Press any key to attempt automatic elevation...
    pause >nul
    
    REM Attempt to elevate
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:install
echo.
echo Starting PowerShell installer...
echo.

REM Run the PowerShell installer
powershell -ExecutionPolicy Bypass -File "%~dp0install_qnti.ps1"

if %errorlevel% == 0 (
    echo.
    echo Installation completed successfully!
) else (
    echo.
    echo Installation failed with error code: %errorlevel%
)

echo.
pause 