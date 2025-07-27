@echo off
echo ========================================
echo QNTI Desktop Installer Builder
echo ========================================
echo.

REM Check if executable exists
if not exist "dist\QNTI_Desktop.exe" (
    echo ERROR: QNTI_Desktop.exe not found at dist\QNTI_Desktop.exe!
    echo.
    echo Available files in dist:
    dir dist 2>nul
    echo.
    echo Please run build_desktop.bat first to create the executable.
    pause
    exit /b 1
)

REM Check if Inno Setup is installed
set "INNO_PATH="
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set "INNO_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
) else if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set "INNO_PATH=C:\Program Files\Inno Setup 6\ISCC.exe"
) else (
    echo ERROR: Inno Setup not found!
    echo.
    echo Please download and install Inno Setup 6 from:
    echo https://jrsoftware.org/isdl.php
    echo.
    echo Alternative: You can distribute the executable directly from:
    echo   dist\QNTI_Desktop.exe
    echo.
    pause
    exit /b 1
)

echo Found Inno Setup at: %INNO_PATH%
echo.

REM Create installer directory
if not exist "installer" mkdir "installer"

echo Building Windows installer...
echo This may take a few minutes...
echo.

REM Build the installer
"%INNO_PATH%" "qnti_installer.iss"

if errorlevel 1 (
    echo.
    echo ERROR: Installer build failed!
    echo Check the output above for details.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installer build completed successfully!
echo ========================================
echo.

REM Check if installer was created
if exist "installer\QNTI_Desktop_Setup.exe" (
    echo Your QNTI installer is ready:
    echo   installer\QNTI_Desktop_Setup.exe
    echo.
    
    REM Get file size
    for %%A in ("installer\QNTI_Desktop_Setup.exe") do (
        set /a size_mb=%%~zA/1024/1024
        echo   Size: %%~zA bytes (~!size_mb! MB)
    )
    
    echo.
    echo Distribution checklist:
    echo   [✓] Standalone executable: dist\QNTI_Desktop.exe
    echo   [✓] Windows installer: installer\QNTI_Desktop_Setup.exe
    echo.
    echo You can now distribute either:
    echo   1. The installer (recommended for end users)
    echo   2. The standalone executable (for portable use)
    echo.
    
    set /p choice="Would you like to test the installer? (y/n): "
    if /i "!choice!"=="y" (
        echo.
        echo Starting installer...
        start "" "installer\QNTI_Desktop_Setup.exe"
    )
) else (
    echo ERROR: Installer file not found!
    echo Check for errors in the build process above.
)

echo.
pause 