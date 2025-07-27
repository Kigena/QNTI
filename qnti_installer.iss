; QNTI Desktop Application Installer Script
; Compatible with Inno Setup 6.0+

[Setup]
AppName=QNTI Trading System
AppVersion=1.0.0
AppPublisher=Quantum Nexus Trading Intelligence
AppPublisherURL=https://github.com/your-org/qnti-trading-system
AppSupportURL=https://github.com/your-org/qnti-trading-system/issues
AppUpdatesURL=https://github.com/your-org/qnti-trading-system/releases
DefaultDirName={autopf}\QNTI
DefaultGroupName=QNTI Trading System
AllowNoIcons=yes
LicenseFile=LICENSE.txt
InfoBeforeFile=README.md
OutputDir=installer
OutputBaseFilename=QNTI_Desktop_Setup
SetupIconFile=qnti_icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1
Name: "startmenu"; Description: "Create Start Menu entry"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce
Name: "autostart"; Description: "Start QNTI with Windows"; GroupDescription: "Startup Options"; Flags: unchecked

[Files]
; Main executable
Source: "dist\QNTI_Desktop.exe"; DestDir: "{app}"; Flags: ignoreversion

; Configuration files
Source: "qnti_config.json"; DestDir: "{app}"; Flags: onlyifdoesntexist
Source: "mt5_config.json"; DestDir: "{app}"; Flags: onlyifdoesntexist  
Source: "vision_config.json"; DestDir: "{app}"; Flags: onlyifdoesntexist

; Dashboard and templates
Source: "dashboard\*"; DestDir: "{app}\dashboard"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "qnti_reports\templates\*"; DestDir: "{app}\qnti_reports\templates"; Flags: ignoreversion recursesubdirs createallsubdirs

; Data directories (if they exist)
Source: "qnti_data\*"; DestDir: "{app}\qnti_data"; Flags: ignoreversion recursesubdirs createallsubdirs; Check: DirExists(ExpandConstant('{src}\qnti_data'))
Source: "ea_profiles\*"; DestDir: "{app}\ea_profiles"; Flags: ignoreversion recursesubdirs createallsubdirs; Check: DirExists(ExpandConstant('{src}\ea_profiles'))

; Documentation
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion; Check: FileExists(ExpandConstant('{src}\LICENSE.txt'))

; Visual C++ Redistributable (if needed)
; Source: "vcredist_x64.exe"; DestDir: {tmp}; Flags: deleteafterinstall; Check: VCRedistNeedsInstall

[Icons]
Name: "{group}\QNTI Trading System"; Filename: "{app}\QNTI_Desktop.exe"
Name: "{group}\QNTI Configuration"; Filename: "{app}\qnti_config.json"
Name: "{group}\QNTI Logs"; Filename: "{app}\logs"
Name: "{group}\{cm:UninstallProgram,QNTI Trading System}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\QNTI Trading System"; Filename: "{app}\QNTI_Desktop.exe"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\QNTI Trading System"; Filename: "{app}\QNTI_Desktop.exe"; Tasks: quicklaunchicon

[Registry]
; Auto-start with Windows (if selected)
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "QNTI"; ValueData: """{app}\QNTI_Desktop.exe"""; Flags: uninsdeletevalue; Tasks: autostart

; File associations for QNTI files
Root: HKCR; Subkey: ".qnti"; ValueType: string; ValueName: ""; ValueData: "QNTIFile"; Flags: uninsdeletevalue
Root: HKCR; Subkey: "QNTIFile"; ValueType: string; ValueName: ""; ValueData: "QNTI Configuration File"; Flags: uninsdeletekey
Root: HKCR; Subkey: "QNTIFile\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\QNTI_Desktop.exe,0"
Root: HKCR; Subkey: "QNTIFile\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\QNTI_Desktop.exe"" ""%1"""

[Run]
; Filename: {tmp}\vcredist_x64.exe; StatusMsg: Installing Visual C++ Redistributable...; Check: VCRedistNeedsInstall
Filename: "{app}\QNTI_Desktop.exe"; Description: "{cm:LaunchProgram,QNTI Trading System}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\qnti_data\*.db"
Type: filesandordirs; Name: "{app}\qnti_memory"

[Code]
function DirExists(DirName: String): Boolean;
begin
  Result := DirExists(DirName);
end;

function FileExists(FileName: String): Boolean;
begin
  Result := FileExists(FileName);
end;

// Check if Visual C++ Redistributable is needed
function VCRedistNeedsInstall: Boolean;
var
  Version: String;
begin
  Result := not RegQueryStringValue(HKEY_LOCAL_MACHINE,
    'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Version', Version);
end;

// Custom page for configuration
procedure InitializeWizard();
begin
  // Add custom wizard pages here if needed
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  
  // Custom validation for configuration page
  if CurPageID = wpSelectTasks then
  begin
    // Could add validation here
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // Create logs directory
    CreateDir(ExpandConstant('{app}\logs'));
    
    // Set permissions if needed
    // Could add additional post-install setup here
  end;
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := False;
  
  // Skip license page if no license file
  if (PageID = wpLicense) and not FileExists(ExpandConstant('{src}\LICENSE.txt')) then
    Result := True;
end;

function GetUninstallString(): String;
var
  sUnInstPath: String;
  sUnInstallString: String;
begin
  sUnInstPath := ExpandConstant('Software\Microsoft\Windows\CurrentVersion\Uninstall\{#emit SetupSetting("AppId")}_is1');
  sUnInstallString := '';
  if not RegQueryStringValue(HKLM, sUnInstPath, 'UninstallString', sUnInstallString) then
    RegQueryStringValue(HKCU, sUnInstPath, 'UninstallString', sUnInstallString);
  Result := sUnInstallString;
end;

function IsUpgrade(): Boolean;
begin
  Result := (GetUninstallString() <> '');
end;

function UnInstallOldVersion(): Integer;
var
  sUnInstallString: String;
  iResultCode: Integer;
begin
  Result := 0;
  sUnInstallString := GetUninstallString();
  if sUnInstallString <> '' then begin
    sUnInstallString := RemoveQuotes(sUnInstallString);
    if Exec(sUnInstallString, '/SILENT /NORESTART /SUPPRESSMSGBOXES','', SW_HIDE, ewWaitUntilTerminated, iResultCode) then
      Result := 3
    else
      Result := 2;
  end else
    Result := 1;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  // Add custom uninstall steps here
end; 