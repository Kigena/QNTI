# QNTI Trading System - PowerShell Installer
# Professional installation script for Windows

param(
    [string]$InstallPath = "$env:ProgramFiles\QNTI",
    [switch]$Uninstall,
    [switch]$Silent
)

# Require administrator privileges
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    if (!$Silent) {
        Write-Host "This installer requires administrator privileges." -ForegroundColor Red
        Write-Host "Please run PowerShell as Administrator and try again." -ForegroundColor Yellow
        pause
    }
    exit 1
}

function Write-Header {
    Clear-Host
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "   QNTI Trading System Installer" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
}

function Test-Requirements {
    Write-Host "Checking system requirements..." -ForegroundColor Yellow
    
    # Check if executable exists
    if (!(Test-Path "dist\QNTI_Desktop.exe")) {
        Write-Host "ERROR: QNTI_Desktop.exe not found!" -ForegroundColor Red
        Write-Host "Please build the executable first using: .\build_desktop.bat" -ForegroundColor Yellow
        return $false
    }
    
    Write-Host "✓ QNTI Desktop executable found" -ForegroundColor Green
    return $true
}

function Install-QNTI {
    Write-Header
    
    if (!(Test-Requirements)) {
        if (!$Silent) { pause }
        exit 1
    }
    
    Write-Host "Installing QNTI Trading System..." -ForegroundColor Green
    Write-Host "Installation Path: $InstallPath" -ForegroundColor Gray
    Write-Host ""
    
    try {
        # Create installation directory
        Write-Host "Creating installation directory..." -ForegroundColor Yellow
        if (!(Test-Path $InstallPath)) {
            New-Item -Path $InstallPath -ItemType Directory -Force | Out-Null
        }
        
        # Create subdirectories
        $dirs = @("logs", "data", "config")
        foreach ($dir in $dirs) {
            $dirPath = Join-Path $InstallPath $dir
            if (!(Test-Path $dirPath)) {
                New-Item -Path $dirPath -ItemType Directory -Force | Out-Null
            }
        }
        
        Write-Host "✓ Installation directories created" -ForegroundColor Green
        
        # Copy main executable
        Write-Host "Copying QNTI Desktop application..." -ForegroundColor Yellow
        Copy-Item "dist\QNTI_Desktop.exe" -Destination "$InstallPath\QNTI_Desktop.exe" -Force
        Write-Host "✓ Application copied" -ForegroundColor Green
        
        # Copy configuration files
        Write-Host "Copying configuration files..." -ForegroundColor Yellow
        $configFiles = @("qnti_config.json", "mt5_config.json", "vision_config.json")
        foreach ($file in $configFiles) {
            if (Test-Path $file) {
                Copy-Item $file -Destination "$InstallPath\config\$file" -Force
            }
        }
        Write-Host "✓ Configuration files copied" -ForegroundColor Green
        
        # Copy dashboard and templates
        Write-Host "Copying dashboard files..." -ForegroundColor Yellow
        if (Test-Path "dashboard") {
            Copy-Item "dashboard" -Destination "$InstallPath\dashboard" -Recurse -Force
        }
        if (Test-Path "qnti_reports\templates") {
            New-Item -Path "$InstallPath\qnti_reports" -ItemType Directory -Force | Out-Null
            Copy-Item "qnti_reports\templates" -Destination "$InstallPath\qnti_reports\templates" -Recurse -Force
        }
        Write-Host "✓ Dashboard files copied" -ForegroundColor Green
        
        # Create Start Menu shortcuts
        Write-Host "Creating Start Menu shortcuts..." -ForegroundColor Yellow
        $startMenuPath = "$env:ProgramData\Microsoft\Windows\Start Menu\Programs\QNTI"
        if (!(Test-Path $startMenuPath)) {
            New-Item -Path $startMenuPath -ItemType Directory -Force | Out-Null
        }
        
        # Main application shortcut
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut("$startMenuPath\QNTI Trading System.lnk")
        $shortcut.TargetPath = "$InstallPath\QNTI_Desktop.exe"
        $shortcut.WorkingDirectory = $InstallPath
        $shortcut.Description = "QNTI Trading System"
        $shortcut.Save()
        
        # Configuration shortcut
        $configShortcut = $shell.CreateShortcut("$startMenuPath\QNTI Configuration.lnk")
        $configShortcut.TargetPath = "$InstallPath\config"
        $configShortcut.Description = "QNTI Configuration Files"
        $configShortcut.Save()
        
        # Logs shortcut
        $logsShortcut = $shell.CreateShortcut("$startMenuPath\QNTI Logs.lnk")
        $logsShortcut.TargetPath = "$InstallPath\logs"
        $logsShortcut.Description = "QNTI Log Files"
        $logsShortcut.Save()
        
        Write-Host "✓ Start Menu shortcuts created" -ForegroundColor Green
        
        # Create Desktop shortcut (optional)
        Write-Host "Creating Desktop shortcut..." -ForegroundColor Yellow
        $desktopPath = [Environment]::GetFolderPath("CommonDesktopDirectory")
        $desktopShortcut = $shell.CreateShortcut("$desktopPath\QNTI Trading System.lnk")
        $desktopShortcut.TargetPath = "$InstallPath\QNTI_Desktop.exe"
        $desktopShortcut.WorkingDirectory = $InstallPath
        $desktopShortcut.Description = "QNTI Trading System"
        $desktopShortcut.Save()
        Write-Host "✓ Desktop shortcut created" -ForegroundColor Green
        
        # Add to Windows Programs list
        Write-Host "Registering with Windows..." -ForegroundColor Yellow
        $uninstallKey = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\QNTI"
        New-Item -Path $uninstallKey -Force | Out-Null
        Set-ItemProperty -Path $uninstallKey -Name "DisplayName" -Value "QNTI Trading System"
        Set-ItemProperty -Path $uninstallKey -Name "DisplayVersion" -Value "1.0.0"
        Set-ItemProperty -Path $uninstallKey -Name "Publisher" -Value "Quantum Nexus Trading Intelligence"
        Set-ItemProperty -Path $uninstallKey -Name "InstallLocation" -Value $InstallPath
        Set-ItemProperty -Path $uninstallKey -Name "UninstallString" -Value "powershell.exe -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Uninstall"
        Set-ItemProperty -Path $uninstallKey -Name "NoModify" -Value 1
        Set-ItemProperty -Path $uninstallKey -Name "NoRepair" -Value 1
        Set-ItemProperty -Path $uninstallKey -Name "EstimatedSize" -Value 400000  # ~400MB in KB
        
        Write-Host "✓ Registered with Windows Programs list" -ForegroundColor Green
        
        # Add PATH environment variable (optional)
        Write-Host "Adding to system PATH..." -ForegroundColor Yellow
        $currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
        if ($currentPath -notlike "*$InstallPath*") {
            [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$InstallPath", "Machine")
        }
        Write-Host "✓ Added to system PATH" -ForegroundColor Green
        
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "   Installation Completed Successfully!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "QNTI Trading System has been installed to:" -ForegroundColor White
        Write-Host "  $InstallPath" -ForegroundColor Gray
        Write-Host ""
        Write-Host "You can now start QNTI from:" -ForegroundColor White
        Write-Host "  • Start Menu > QNTI Trading System" -ForegroundColor Gray
        Write-Host "  • Desktop shortcut" -ForegroundColor Gray
        Write-Host "  • Command line: qnti" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Configuration files are located at:" -ForegroundColor White
        Write-Host "  $InstallPath\config\" -ForegroundColor Gray
        Write-Host ""
        
        if (!$Silent) {
            $launch = Read-Host "Would you like to launch QNTI now? (y/n)"
            if ($launch -eq "y" -or $launch -eq "Y") {
                Start-Process "$InstallPath\QNTI_Desktop.exe"
            }
        }
        
    } catch {
        Write-Host ""
        Write-Host "ERROR: Installation failed!" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        if (!$Silent) { pause }
        exit 1
    }
}

function Uninstall-QNTI {
    Write-Header
    Write-Host "Uninstalling QNTI Trading System..." -ForegroundColor Yellow
    Write-Host ""
    
    try {
        # Remove from Programs list
        Write-Host "Removing from Windows Programs list..." -ForegroundColor Yellow
        $uninstallKey = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\QNTI"
        if (Test-Path $uninstallKey) {
            Remove-Item -Path $uninstallKey -Force
        }
        Write-Host "✓ Removed from Programs list" -ForegroundColor Green
        
        # Remove Start Menu shortcuts
        Write-Host "Removing Start Menu shortcuts..." -ForegroundColor Yellow
        $startMenuPath = "$env:ProgramData\Microsoft\Windows\Start Menu\Programs\QNTI"
        if (Test-Path $startMenuPath) {
            Remove-Item -Path $startMenuPath -Recurse -Force
        }
        Write-Host "✓ Start Menu shortcuts removed" -ForegroundColor Green
        
        # Remove Desktop shortcut
        Write-Host "Removing Desktop shortcut..." -ForegroundColor Yellow
        $desktopPath = [Environment]::GetFolderPath("CommonDesktopDirectory")
        $desktopShortcut = "$desktopPath\QNTI Trading System.lnk"
        if (Test-Path $desktopShortcut) {
            Remove-Item -Path $desktopShortcut -Force
        }
        Write-Host "✓ Desktop shortcut removed" -ForegroundColor Green
        
        # Remove from PATH
        Write-Host "Removing from system PATH..." -ForegroundColor Yellow
        $currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
        $newPath = $currentPath -replace [regex]::Escape(";$InstallPath"), ""
        $newPath = $newPath -replace [regex]::Escape("$InstallPath;"), ""
        [Environment]::SetEnvironmentVariable("PATH", $newPath, "Machine")
        Write-Host "✓ Removed from system PATH" -ForegroundColor Green
        
        # Remove installation directory
        Write-Host "Removing installation files..." -ForegroundColor Yellow
        if (Test-Path $InstallPath) {
            Remove-Item -Path $InstallPath -Recurse -Force
        }
        Write-Host "✓ Installation files removed" -ForegroundColor Green
        
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "   Uninstallation Completed!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "QNTI Trading System has been completely removed from your system." -ForegroundColor White
        Write-Host ""
        
    } catch {
        Write-Host ""
        Write-Host "ERROR: Uninstallation failed!" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        if (!$Silent) { pause }
        exit 1
    }
}

# Main execution
if ($Uninstall) {
    Uninstall-QNTI
} else {
    Install-QNTI
}

if (!$Silent) {
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
} 